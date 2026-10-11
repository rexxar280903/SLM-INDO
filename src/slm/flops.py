"""Akuntansi parameter, FLOPs, dan memori inferensi (analitik).

Konvensi (D30, paper §3.6):
- 1 multiply–add = 2 FLOPs. Hanya perkalian matriks/einsum yang dihitung; norm,
  aktivasi, softmax, RoPE diabaikan (<1%).
- FLOPs training "model" = 3 × forward (forward + backward 2×). Rematerialisasi
  TIDAK dihitung (itu biaya implementasi, bukan model) → dipakai untuk budget & MFU.
- Attention kausal: rata-rata (T+1)/2 key per query (biaya algoritmik, bukan kernel
  yang menghitung blok penuh). Sliding window: min(posisi, W).
- GDN chunkwise: biaya algoritmik bentuk chunkwise dengan triangular solve.
Angka ini divalidasi silang dengan cost analysis XLA pada pilot (lihat train.py).
"""
from __future__ import annotations

from .config import ModelConfig


def mixer_params(cfg: ModelConfig, layer_idx: int) -> int:
    d, h, e = cfg.d_model, cfg.n_heads, cfg.head_dim
    if cfg.layer_types[layer_idx] == "gdn":
        # q,k,v,g (d×He), o (He×d), a,b (d×H), conv q/k/v (K×He), A_log, dt_bias (H), o_norm (e)
        return 5 * d * h * e + 2 * d * h + 3 * cfg.conv_size * h * e + 2 * h + e
    if cfg.mla_latent > 0:
        c, r = cfg.mla_latent, cfg.mla_rope_dim
        return d * h * e + d * c + c + c * h * (e - r) + c * h * e + d * r + h * e * d
    p = 2 * d * h * e                                      # wq, wo
    if cfg.kv_source()[layer_idx] == layer_idx:
        p += 2 * d * cfg.kv_heads * e                      # wk, wv
    return p


def param_count(cfg: ModelConfig) -> dict:
    d, f = cfg.d_model, cfg.ffn_dim
    emb = cfg.vocab_size * d
    per_layer = [mixer_params(cfg, i) + 3 * d * f + 2 * d for i in range(cfg.n_layers)]
    non_emb = sum(per_layer) + d                           # + norm akhir
    return {"total": emb + non_emb, "embedding": emb, "non_embedding": non_emb}


def mixer_flops_per_token(cfg: ModelConfig, layer_idx: int, seq_len: int) -> float:
    """FLOPs forward per token untuk satu mixer pada urutan sepanjang seq_len."""
    d, h, e = cfg.d_model, cfg.n_heads, cfg.head_dim
    t = seq_len
    if cfg.layer_types[layer_idx] == "gdn":
        proj = 2 * d * (5 * h * e + 2 * h)
        conv = 2 * 3 * cfg.conv_size * h * e
        c = cfg.gdn_chunk
        # per head per token: KKᵀ (2Ce) + solve (C·2e) + QKᵀ (2Ce) + attn·U (2Ce) + 3 kali e×e (U, Q·S, update S)
        core = h * (2 * c * e + c * 2 * e + 2 * c * e + 2 * c * e + 3 * 2 * e * e)
        return proj + conv + core
    keys = (t + 1) / 2 if cfg.window == 0 else _mean_window(t, cfg.window)
    core = 2 * 2 * h * e * keys                            # QKᵀ dan probs·V
    if cfg.mla_latent > 0:
        c, r = cfg.mla_latent, cfg.mla_rope_dim
        proj = 2 * (d * h * e + d * c + c * h * (e - r) + c * h * e + d * r + h * e * d)
        return proj + core
    proj = 2 * d * h * e * 2                               # wq, wo
    if cfg.kv_source()[layer_idx] == layer_idx:
        proj += 2 * d * cfg.kv_heads * e * 2               # wk, wv
    return proj + core


def _mean_window(t: int, w: int) -> float:
    if t <= w:
        return (t + 1) / 2
    return (w * (w + 1) / 2 + (t - w) * w) / t


def forward_flops_per_token(cfg: ModelConfig, seq_len: int | None = None) -> dict:
    t = seq_len or cfg.seq_len
    d, f = cfg.d_model, cfg.ffn_dim
    mixer = [mixer_flops_per_token(cfg, i, t) for i in range(cfg.n_layers)]
    ffn = cfg.n_layers * 2 * 3 * d * f
    head = 2 * d * cfg.vocab_size
    total = sum(mixer) + ffn + head
    return {"total": total, "mixers": sum(mixer), "ffn": ffn, "lm_head": head, "per_layer_mixer": mixer}


def train_flops_per_token(cfg: ModelConfig, seq_len: int | None = None) -> float:
    return 3.0 * forward_flops_per_token(cfg, seq_len)["total"]


def inference_memory(cfg: ModelConfig, context: int, batch: int = 1, kv_bytes: int = 2, state_bytes: int = 4) -> dict:
    """Memori tambahan saat decode (paper Pers. 5–6): KV-cache + state rekuren + buffer conv.

    MHA/GQA: 2·G·E per token per layer produsen.  MLA: (c + r) per token per layer.
    CLA: cache layer penerima tidak dihitung.  Sliding window: konteks dibatasi W.
    """
    e = cfg.head_dim
    kv_tokens = context if cfg.window == 0 else min(context, cfg.window)
    sources = cfg.kv_source()
    per_token = 0
    for i in cfg.attn_layers:
        if cfg.mla_latent > 0:
            per_token += cfg.mla_latent + cfg.mla_rope_dim
        elif sources[i] == i:
            per_token += 2 * cfg.kv_heads * e
    kv = batch * kv_tokens * per_token * kv_bytes
    n_gdn = len(cfg.gdn_layers)
    state = batch * n_gdn * cfg.n_heads * e * e * state_bytes
    conv = batch * n_gdn * (cfg.conv_size - 1) * 3 * cfg.n_heads * e * kv_bytes
    return {"kv": kv, "state": state, "conv": conv, "total": kv + state + conv,
            "kv_per_token": per_token * kv_bytes}
