"""Konfigurasi skala, arsitektur, dan training.

Semua angka di sini adalah KEPUTUSAN DESAIN (KONSEP.md §3, D23–D34), bukan hasil.
Hyperparameter numerik (LR, warmup) adalah nilai awal yang dikunci setelah
sweep LR pada pilot S10/S25 (KONSEP.md §10D).

Cara membaca nama arsitektur:  "<campuran>-<varian KV>"
  campuran : full (semua attention) | h1 (1:1) | h3 (3:1) | h6 (6:1) | linear (semua GDN)
  varian KV: mha | gqa | mla | cla | swa (sliding window)
Alias baseline: b0=full-mha, b1=full-gqa, b2=linear, b3=h3-mha, b4=h3-swa.
"""
from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Konstanta desain
# ---------------------------------------------------------------------------
VOCAB_SIZE = 16_384          # D25: 2^14, kelipatan 128 (ramah MXU TPU); ablation 8.192 / 32.768
HEAD_DIM = 64                # D26: dimensi head attention dan head GDN di semua skala
SEQ_LEN = 2_048              # D29: konteks training sama untuk semua skala
ROPE_THETA = 10_000.0
MLA_ROPE_DIM = 32            # bagian RoPE terpisah (decoupled) pada MLA
GDN_CHUNK = 64               # panjang chunk untuk bentuk chunkwise Gated DeltaNet
CONV_SIZE = 4                # kernel konvolusi pendek kausal pada GDN
SWA_WINDOW = 512             # jendela B4 (sliding window attention)

ALIASES = {"b0": "full-mha", "b1": "full-gqa", "b2": "linear", "b3": "h3-mha", "b4": "h3-swa"}
MIXES = {"full": 0, "h1": 1, "h3": 3, "h6": 6, "linear": -1}   # nilai = r pada rasio r:1
KV_VARIANTS = ("mha", "gqa", "mla", "cla", "swa")


def round_up(x: float, multiple: int) -> int:
    return int(math.ceil(x / multiple) * multiple)


@dataclass(frozen=True)
class Scale:
    """Satu anak tangga skala (D12). Lebar FFN = 8/3·d dibulatkan ke kelipatan 128."""
    name: str
    d_model: int
    n_layers: int
    n_heads: int
    ref_tokens: int          # token referensi (≈20 token/parameter) → anchor budget FLOPs
    global_batch: int        # jumlah urutan per langkah optimizer (semua chip)
    micro_batch: int         # urutan per micro-step (akumulasi gradien bila < global_batch)
    peak_lr: float           # nilai awal; dikunci setelah sweep LR pilot
    seq_len: int = SEQ_LEN

    @property
    def ffn_dim(self) -> int:
        return round_up(8 * self.d_model / 3, 128)

    @property
    def tokens_per_step(self) -> int:
        return self.global_batch * self.seq_len


SCALES: dict[str, Scale] = {
    "s10": Scale("s10", 256, 8, 4, 200_000_000, 64, 64, 2.0e-3),
    "s25": Scale("s25", 384, 10, 6, 500_000_000, 128, 128, 1.5e-3),
    "s50": Scale("s50", 512, 14, 8, 1_000_000_000, 128, 128, 1.2e-3),
    "s125": Scale("s125", 768, 12, 12, 2_000_000_000, 256, 128, 1.0e-3),
    # Versi mini untuk debug di laptop/CPU (bukan bagian studi).
    "mini": Scale("mini", 192, 6, 3, 4_000_000, 8, 8, 3.0e-3, seq_len=256),
    # Skala sangat kecil khusus unit test (±40 langkah).
    "tiny": Scale("tiny", 128, 4, 2, 20_480, 8, 8, 3.0e-3, seq_len=64),
}


@dataclass(frozen=True)
class ModelConfig:
    name: str
    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    ffn_dim: int
    layer_types: tuple[str, ...]        # "attn" atau "gdn" per layer (L1..Ln)
    kv_heads: int                       # MHA: = n_heads; GQA: n_heads // 2
    head_dim: int = HEAD_DIM
    mla_latent: int = 0                 # 0 = MLA nonaktif; >0 = dimensi laten c_KV
    mla_rope_dim: int = MLA_ROPE_DIM
    cla_share: int = 1                  # 1 = tanpa berbagi; 2 = sepasang layer attention berbagi K/V
    window: int = 0                     # 0 = attention penuh; >0 = sliding window
    rope_theta: float = ROPE_THETA
    gdn_chunk: int = GDN_CHUNK
    conv_size: int = CONV_SIZE
    norm_eps: float = 1e-6
    init_std: float = 0.02
    seq_len: int = SEQ_LEN
    # Pilihan implementasi (tidak mengubah matematika model):
    attn_impl: str = "auto"             # auto | xla | pallas
    gdn_solver: str = "solve"           # solve (triangular solve) | product (perkalian matriks)
    gdn_precision: str = "float32"      # presisi matmul inti GDN; di TPU default fp32 = pass bf16 (diuji di pilot)
    remat: bool = True                  # rematerialisasi per blok: hemat memori, +komputasi
    compute_dtype: str = "bfloat16"

    @property
    def attn_layers(self) -> list[int]:
        return [i for i, t in enumerate(self.layer_types) if t == "attn"]

    @property
    def gdn_layers(self) -> list[int]:
        return [i for i, t in enumerate(self.layer_types) if t == "gdn"]

    def kv_source(self) -> dict[int, int]:
        """CLA: peta layer attention -> layer yang menghasilkan K/V-nya (dirinya sendiri bila produsen)."""
        src = {}
        for j, i in enumerate(self.attn_layers):
            group_start = (j // self.cla_share) * self.cla_share
            src[i] = self.attn_layers[group_start]
        return src

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["layer_types"] = list(self.layer_types)
        return d


def hybrid_pattern(n_layers: int, r: int) -> tuple[str, ...]:
    """Penempatan layer attention untuk rasio nominal r:1 (D28).

    Aturan tetap (ditetapkan sebelum eksperimen):
      n_attn = max(1, floor(L/(r+1) + 0,5))           # pembulatan setengah ke atas
      posisi  = ceil(j·L/n_attn), j = 1..n_attn (1-indexed)
    Akibatnya layer terakhir selalu attention dan layer pertama tidak pernah attention.
    Rasio aktual dilaporkan karena tidak selalu sama dengan rasio nominal.
    """
    if r == 0:
        return ("attn",) * n_layers
    if r < 0:
        return ("gdn",) * n_layers
    n_attn = max(1, int(math.floor(n_layers / (r + 1) + 0.5)))
    pos = {int(math.ceil(j * n_layers / n_attn)) for j in range(1, n_attn + 1)}
    return tuple("attn" if (i + 1) in pos else "gdn" for i in range(n_layers))


def build_model_config(arch: str, scale: str | Scale, vocab_size: int = VOCAB_SIZE, **overrides) -> ModelConfig:
    s = SCALES[scale] if isinstance(scale, str) else scale
    canonical = ALIASES.get(arch, arch)
    if canonical == "linear":
        mix, kv = "linear", "mha"
    else:
        mix, kv = canonical.split("-")
    if mix not in MIXES or kv not in KV_VARIANTS:
        raise ValueError(f"arsitektur tidak dikenal: {arch}")
    layer_types = hybrid_pattern(s.n_layers, MIXES[mix])
    kw = dict(kv_heads=s.n_heads)
    if kv == "gqa":                      # D27: anggaran cache = ½ MHA
        kw["kv_heads"] = s.n_heads // 2
    elif kv == "mla":                    # cache per token per layer = latent + rope = d_model (½ MHA)
        kw["mla_latent"] = s.d_model - MLA_ROPE_DIM
    elif kv == "cla":                    # sepasang layer attention berbagi K/V (½ MHA bila jumlah layer genap)
        kw["cla_share"] = 2
    elif kv == "swa":
        kw["window"] = SWA_WINDOW
    kw.update(overrides)
    kw.setdefault("seq_len", s.seq_len)
    return ModelConfig(
        name=f"{canonical}-{s.name}", vocab_size=vocab_size, d_model=s.d_model, n_layers=s.n_layers,
        n_heads=s.n_heads, ffn_dim=s.ffn_dim, layer_types=layer_types, **kw)


def grid_archs() -> list[str]:
    """17 konfigurasi grid utama (KONSEP §10): 4 campuran × 4 varian KV + linear murni."""
    return [f"{m}-{kv}" for m in ("full", "h1", "h3", "h6") for kv in ("mha", "gqa", "mla", "cla")] + ["linear"]


@dataclass
class TrainConfig:
    arch: str = "b0"
    scale: str = "s10"
    seed: int = 0
    budget_mode: str = "flops"          # flops (compute-matched, utama) | tokens (kontrol token-matched)
    anchor_arch: str = "b0"             # D30: budget FLOPs = FLOPs/token anchor × token referensi
    peak_lr: float | None = None        # None = pakai nilai skala
    lr_multiplier: float = 1.0          # untuk sweep LR {0,5; 1; 2}
    min_lr_ratio: float = 0.1           # cosine turun ke 10% LR puncak
    warmup_frac: float = 0.03
    min_warmup_steps: int = 50
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    z_loss: float = 0.0
    global_batch: int | None = None
    micro_batch: int | None = None
    max_steps: int | None = None        # pemotong untuk smoke test / benchmark
    eval_points: int = 10               # evaluasi di 10%, 20%, … budget (grid compute bersama)
    eval_tokens: int = 4_194_304        # ±4,2M token validasi per evaluasi
    log_every: int = 10
    ckpt_every_minutes: float = 30.0
    data_dir: str = ""
    synthetic_data: bool = False        # data acak (benchmark/smoke test tanpa korpus)
    out_dir: str = "runs"
    # N2 (linear-first warm-start)
    n2_tau: float = 0.0                 # 0 = nonaktif; 0,10/0,25/0,50 = titik alih (fraksi langkah referensi)
    n2_target: str = "h3-mha"           # arsitektur akhir setelah konversi
    n2_keep_moments: bool = True        # momen AdamW tensor yang ditransfer dipertahankan
    n2_transfer: bool = True            # False = ablation: layer attention baru diinisialisasi acak
    task: str = "lm"                    # lm | mqar
    model_overrides: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)


def lr_at(progress: float, peak: float, warmup_frac: float, min_ratio: float) -> float:
    """Jadwal LR sebagai fungsi progres budget p∈[0,1] (fraksi FLOPs kumulatif).

    Warmup linear sampai p_w, lalu cosine turun ke min_ratio·peak di p=1.
    Memakai p (bukan nomor langkah) agar jadwal N2 tetap benar walau biaya per
    langkah berubah setelah konversi.
    """
    p = min(max(progress, 0.0), 1.0)
    if p < warmup_frac:
        return peak * p / warmup_frac
    t = (p - warmup_frac) / max(1e-12, 1.0 - warmup_frac)
    return peak * (min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * t)))
