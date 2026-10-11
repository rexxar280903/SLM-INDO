"""Lapisan dasar dalam gaya JAX fungsional.

Catatan JAX untuk pembaca baru:
- Parameter disimpan sebagai dict biasa berisi array (disebut *pytree*). Tidak ada
  objek "layer" yang menyimpan state; setiap fungsi menerima `p` (parameter) dan `x`
  (input) lalu mengembalikan output. Fungsi murni seperti ini bisa di-`jax.jit`
  (dikompilasi XLA) dan di-`jax.grad` (diturunkan otomatis).
- Bentuk tensor ditulis di komentar: B=batch, T=panjang urutan, D=d_model,
  H=jumlah head query, G=jumlah head K/V, E=head_dim.
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from .config import ModelConfig

# ---------------------------------------------------------------------------
# Normalisasi, aktivasi, RoPE
# ---------------------------------------------------------------------------

def rms_norm(x, weight, eps: float = 1e-6):
    """RMSNorm: x / sqrt(mean(x²) + eps) * weight. Dihitung di fp32 agar stabil."""
    x32 = x.astype(jnp.float32)
    y = x32 * jax.lax.rsqrt(jnp.mean(x32 * x32, axis=-1, keepdims=True) + eps)
    return (y * weight.astype(jnp.float32)).astype(x.dtype)


def l2_norm(x, eps: float = 1e-6):
    """Normalisasi L2 per vektor (dipakai pada q dan k Gated DeltaNet)."""
    x32 = x.astype(jnp.float32)
    return x32 * jax.lax.rsqrt(jnp.sum(x32 * x32, axis=-1, keepdims=True) + eps)


def rope_tables(seq_len: int, dim: int, theta: float):
    """Tabel cos/sin RoPE berbentuk (T, dim/2), fp32."""
    inv_freq = 1.0 / (theta ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))
    angles = jnp.arange(seq_len, dtype=jnp.float32)[:, None] * inv_freq[None, :]
    return jnp.cos(angles), jnp.sin(angles)


def apply_rope(x, cos, sin):
    """Rotasi pasangan kanal (gaya LLaMA 'rotate half'). x: (B,T,H,E) dengan E genap."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half].astype(jnp.float32), x[..., half:].astype(jnp.float32)
    c, s = cos[None, :, None, :], sin[None, :, None, :]
    out = jnp.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], axis=-1)
    return out.astype(x.dtype)


def swiglu_ffn(p, x):
    """FFN SwiGLU: W_down( SiLU(x·W_gate) ⊙ (x·W_up) ). Tanpa bias."""
    return (jax.nn.silu(x @ p["w_gate"]) * (x @ p["w_up"])) @ p["w_down"]


# ---------------------------------------------------------------------------
# Inti attention
# ---------------------------------------------------------------------------

def _attention_xla(q, k, v, window: int):
    """Attention kausal eksplisit (mudah dibaca, materialisasi matriks skor T×T).

    q: (B,T,H,E)  k: (B,T,G,E)  v: (B,T,G,Ev); H harus kelipatan G (GQA).
    """
    b, t, h, e = q.shape
    g = k.shape[2]
    q = q.reshape(b, t, g, h // g, e)
    scores = jnp.einsum("btgqe,bsge->bgqts", q, k, preferred_element_type=jnp.float32)
    scores = scores / math.sqrt(e)
    i = jnp.arange(t)[:, None]
    j = jnp.arange(t)[None, :]
    allowed = j <= i
    if window > 0:
        allowed = allowed & (i - j < window)
    scores = jnp.where(allowed[None, None, None], scores, jnp.finfo(jnp.float32).min)
    probs = jax.nn.softmax(scores, axis=-1).astype(v.dtype)
    out = jnp.einsum("bgqts,bsge->btgqe", probs, v)
    return out.reshape(b, t, h, v.shape[-1])


def _attention_pallas(q, k, v):
    """Flash attention Pallas untuk TPU (tidak menyimpan matriks T×T).

    Kernel ini memakai jumlah head yang sama untuk Q dan K/V, jadi K/V GQA diulang
    (biaya training sama dengan MHA; penghematan GQA ada pada cache inferensi).
    """
    from jax.experimental.pallas.ops.tpu.flash_attention import flash_attention

    h, g = q.shape[2], k.shape[2]
    if g != h:
        k = jnp.repeat(k, h // g, axis=2)
        v = jnp.repeat(v, h // g, axis=2)
    tr = lambda a: jnp.transpose(a, (0, 2, 1, 3))
    out = flash_attention(tr(q), tr(k), tr(v), causal=True, sm_scale=1.0 / math.sqrt(q.shape[-1]))
    return tr(out)


def attention_core(q, k, v, cfg: ModelConfig):
    impl = cfg.attn_impl
    if impl == "auto":
        impl = "pallas" if jax.default_backend() == "tpu" else "xla"
    same_dim = q.shape[-1] == v.shape[-1]
    if impl == "pallas" and cfg.window == 0 and same_dim:
        return _attention_pallas(q, k, v)
    return _attention_xla(q, k, v, cfg.window)


# ---------------------------------------------------------------------------
# Mixer attention: MHA / GQA / CLA / sliding window / MLA
# ---------------------------------------------------------------------------

def attention_mixer(p, x, cfg: ModelConfig, rope, shared_kv=None):
    """Mixer full attention. Mengembalikan (output, kv) agar CLA bisa memakai ulang K/V.

    - MHA/GQA: K/V diproyeksikan dari x (G head K/V).
    - CLA: layer 'penerima' tidak punya W_K/W_V; memakai `shared_kv` dari layer sumber.
    - MLA: K/V direkonstruksi dari laten c_KV (yang disimpan di cache saat inferensi)
      ditambah satu kunci RoPE bersama (decoupled RoPE, gaya DeepSeek-V2).
    """
    b, t, _ = x.shape
    h, e = cfg.n_heads, cfg.head_dim
    cos, sin = rope
    if cfg.mla_latent > 0:
        return _mla_mixer(p, x, cfg, rope), None
    q = (x @ p["wq"]).reshape(b, t, h, e)
    q = apply_rope(q, cos, sin)
    if shared_kv is None:
        g = cfg.kv_heads
        k = (x @ p["wk"]).reshape(b, t, g, e)
        v = (x @ p["wv"]).reshape(b, t, g, e)
        k = apply_rope(k, cos, sin)
    else:
        k, v = shared_kv
    out = attention_core(q, k, v, cfg).reshape(b, t, h * e)
    return out @ p["wo"], (k, v)


def _mla_mixer(p, x, cfg: ModelConfig, rope):
    b, t, _ = x.shape
    h, e, r = cfg.n_heads, cfg.head_dim, cfg.mla_rope_dim
    nope = e - r                                   # bagian q·k tanpa posisi
    cos, sin = rope                                # tabel untuk dimensi r
    q = (x @ p["wq"]).reshape(b, t, h, e)
    q_nope, q_rope = q[..., :nope], apply_rope(q[..., nope:], cos, sin)
    c_kv = rms_norm(x @ p["w_dkv"], p["kv_norm"], cfg.norm_eps)      # (B,T,C) ← yang di-cache
    k_rope = apply_rope((x @ p["w_kr"]).reshape(b, t, 1, r), cos, sin)  # (B,T,1,r) ← di-cache
    k_nope = (c_kv @ p["w_uk"]).reshape(b, t, h, nope)
    v = (c_kv @ p["w_uv"]).reshape(b, t, h, e)
    k = jnp.concatenate([k_nope, jnp.broadcast_to(k_rope, (b, t, h, r))], axis=-1)
    qf = jnp.concatenate([q_nope, q_rope], axis=-1)
    out = attention_core(qf, k, v, cfg).reshape(b, t, h * e)
    return out @ p["wo"]


def causal_conv(x, w):
    """Konvolusi depthwise kausal. x: (B,T,C), w: (K,C). y_t = Σ_i w_i · x_{t-K+1+i}."""
    k = w.shape[0]
    xp = jnp.pad(x, ((0, 0), (k - 1, 0), (0, 0)))
    t = x.shape[1]
    return sum(xp[:, i:i + t, :] * w[i] for i in range(k))
