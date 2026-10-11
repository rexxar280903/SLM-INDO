"""Model bahasa decoder: embedding → L blok (attention atau Gated DeltaNet) → LM head.

Struktur parameter (pytree dict):
    params = {
      "embed": (V, D),                       # di-tie dengan LM head
      "layers": [ {"norm1": (D,), "mixer": {...}, "norm2": (D,), "ffn": {...}}, ... ],
      "norm_f": (D,),
    }
Isi "mixer" bergantung pada jenis layer (lihat `init_attention` / `init_gdn`).
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

from .config import ModelConfig
from .gdn import gdn_mixer
from .layers import attention_mixer, rms_norm, rope_tables, swiglu_ffn

# ---------------------------------------------------------------------------
# Inisialisasi
# ---------------------------------------------------------------------------

def _normal(key, shape, std):
    return std * jax.random.normal(key, shape, jnp.float32)


def init_attention(key, cfg: ModelConfig, layer_idx: int, out_std: float):
    d, h, e = cfg.d_model, cfg.n_heads, cfg.head_dim
    ks = jax.random.split(key, 6)
    std = cfg.init_std
    if cfg.mla_latent > 0:
        c, r = cfg.mla_latent, cfg.mla_rope_dim
        return {
            "wq": _normal(ks[0], (d, h * e), std),
            "w_dkv": _normal(ks[1], (d, c), std),
            "kv_norm": jnp.ones((c,), jnp.float32),
            "w_uk": _normal(ks[2], (c, h * (e - r)), std),
            "w_uv": _normal(ks[3], (c, h * e), std),
            "w_kr": _normal(ks[4], (d, r), std),
            "wo": _normal(ks[5], (h * e, d), out_std),
        }
    p = {"wq": _normal(ks[0], (d, h * e), std), "wo": _normal(ks[3], (h * e, d), out_std)}
    if cfg.kv_source()[layer_idx] == layer_idx:          # produsen K/V (bukan penerima CLA)
        g = cfg.kv_heads
        p["wk"] = _normal(ks[1], (d, g * e), std)
        p["wv"] = _normal(ks[2], (d, g * e), std)
    return p


def init_gdn(key, cfg: ModelConfig, out_std: float):
    d, h, e, kc = cfg.d_model, cfg.n_heads, cfg.head_dim, cfg.conv_size
    ks = jax.random.split(key, 12)
    std = cfg.init_std
    conv_bound = 1.0 / math.sqrt(kc)                     # default Conv1d depthwise (fan_in = kernel)
    conv = lambda k_: jax.random.uniform(k_, (kc, h * e), jnp.float32, -conv_bound, conv_bound)
    # dt ~ log-uniform[1e-3, 1e-1]; dt_bias = softplus⁻¹(dt) (inisialisasi Mamba-2)
    dt = jnp.exp(jax.random.uniform(ks[9], (h,), jnp.float32, math.log(1e-3), math.log(1e-1)))
    dt = jnp.maximum(dt, 1e-4)
    return {
        "wq": _normal(ks[0], (d, h * e), std),
        "wk": _normal(ks[1], (d, h * e), std),
        "wv": _normal(ks[2], (d, h * e), std),
        "wg": _normal(ks[3], (d, h * e), std),
        "wa": _normal(ks[4], (d, h), std),
        "wb": _normal(ks[5], (d, h), std),
        "conv_q": conv(ks[6]), "conv_k": conv(ks[7]), "conv_v": conv(ks[8]),
        "A_log": jnp.log(jax.random.uniform(ks[10], (h,), jnp.float32, 1.0, 16.0)),
        "dt_bias": dt + jnp.log(-jnp.expm1(-dt)),
        "o_norm": jnp.ones((e,), jnp.float32),
        "wo": _normal(ks[11], (h * e, d), out_std),
    }


def init_layer(key, cfg: ModelConfig, i: int):
    d, f = cfg.d_model, cfg.ffn_dim
    out_std = cfg.init_std / math.sqrt(2 * cfg.n_layers)  # proyeksi keluar diperkecil (gaya GPT-2)
    k_mix, k1, k2, k3 = jax.random.split(key, 4)
    mixer = init_attention(k_mix, cfg, i, out_std) if cfg.layer_types[i] == "attn" else init_gdn(k_mix, cfg, out_std)
    return {
        "norm1": jnp.ones((d,), jnp.float32),
        "mixer": mixer,
        "norm2": jnp.ones((d,), jnp.float32),
        "ffn": {
            "w_gate": _normal(k1, (d, f), cfg.init_std),
            "w_up": _normal(k2, (d, f), cfg.init_std),
            "w_down": _normal(k3, (f, d), out_std),
        },
    }


def init_params(key, cfg: ModelConfig):
    k_emb, k_layers = jax.random.split(key)
    layer_keys = jax.random.split(k_layers, cfg.n_layers)
    return {
        "embed": _normal(k_emb, (cfg.vocab_size, cfg.d_model), cfg.init_std),
        "layers": [init_layer(layer_keys[i], cfg, i) for i in range(cfg.n_layers)],
        "norm_f": jnp.ones((cfg.d_model,), jnp.float32),
    }


def count_params(params) -> int:
    return int(sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(params)))


# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------

KEEP_FP32 = ("A_log", "dt_bias")


def cast_params(params, dtype):
    """Salinan parameter dalam dtype komputasi (bf16); master weight tetap fp32."""
    def cast(path, x):
        name = str(path[-1].key) if hasattr(path[-1], "key") else ""
        return x if name in KEEP_FP32 else x.astype(dtype)
    return jax.tree_util.tree_map_with_path(cast, params)


def _block(p, x, cfg: ModelConfig, kind: str, rope, shared_kv):
    h = rms_norm(x, p["norm1"], cfg.norm_eps)
    if kind == "attn":
        mix, kv = attention_mixer(p["mixer"], h, cfg, rope, shared_kv)
    else:
        mix, kv = gdn_mixer(p["mixer"], h, cfg), None
    x = x + mix
    x = x + swiglu_ffn(p["ffn"], rms_norm(x, p["norm2"], cfg.norm_eps))
    return x, kv


def forward_hidden(params, tokens, cfg: ModelConfig):
    """tokens: (B,T) int32 → hidden state terakhir setelah norm akhir, (B,T,D)."""
    dtype = jnp.dtype(cfg.compute_dtype)
    p = cast_params(params, dtype)
    t = tokens.shape[1]
    rope_dim = cfg.mla_rope_dim if cfg.mla_latent > 0 else cfg.head_dim
    rope = rope_tables(t, rope_dim, cfg.rope_theta)
    x = p["embed"][tokens]
    sources = cfg.kv_source()
    kv_store: dict[int, tuple] = {}
    for i, kind in enumerate(cfg.layer_types):
        shared = None
        if kind == "attn" and sources[i] != i:
            shared = kv_store[sources[i]]                 # CLA: pakai K/V layer sumber
        block = lambda p_, x_, s_, kind=kind: _block(p_, x_, cfg, kind, rope, s_)
        if cfg.remat:
            # jax.checkpoint: aktivasi di dalam blok tidak disimpan untuk backward,
            # tetapi dihitung ulang. Memori turun drastis, compute naik ±1/3.
            block = jax.checkpoint(block)
        x, kv = block(p["layers"][i], x, shared)
        if kv is not None and sources.get(i) == i:
            kv_store[i] = kv
    return rms_norm(x, p["norm_f"], cfg.norm_eps), p["embed"]


def logits_fn(params, tokens, cfg: ModelConfig):
    """Logit penuh (B,T,V) — hanya untuk tes/inferensi kecil; training memakai `lm_loss`."""
    h, emb = forward_hidden(params, tokens, cfg)
    return jnp.einsum("btd,vd->btv", h, emb, preferred_element_type=jnp.float32)


def chunked_cross_entropy(h, emb, labels, chunk_tokens: int = 8192, z_loss: float = 0.0):
    """Cross-entropy tanpa membentuk seluruh logit fp32 sekaligus.

    Logit S125 untuk 32 urutan × 2048 token × 16.384 vocab = 4,3 GB di fp32.
    Token dipotong per `chunk_tokens`; tiap potongan dihitung lalu dibuang
    (jax.checkpoint menghitung ulang saat backward).
    Label < 0 diabaikan (dipakai MQAR). Mengembalikan (jumlah NLL, jumlah token dinilai, nll per token).
    """
    d = h.shape[-1]
    hf = h.reshape(-1, d)
    lf = labels.reshape(-1)
    n = hf.shape[0]
    chunk = min(chunk_tokens, n)
    pad = (-n) % chunk
    if pad:
        hf = jnp.pad(hf, ((0, pad), (0, 0)))
        lf = jnp.pad(lf, (0, pad), constant_values=-1)
    hc = hf.reshape(-1, chunk, d)
    lc = lf.reshape(-1, chunk)

    @jax.checkpoint
    def piece(hx, lx):
        logits = jnp.einsum("nd,vd->nv", hx, emb, preferred_element_type=jnp.float32)
        lse = jax.nn.logsumexp(logits, axis=-1)
        tgt = jnp.take_along_axis(logits, jnp.maximum(lx, 0)[:, None], axis=-1)[:, 0]
        valid = (lx >= 0).astype(jnp.float32)
        nll = (lse - tgt) * valid
        zl = z_loss * (lse ** 2) * valid
        return nll, zl

    nll, zl = jax.lax.map(lambda a: piece(*a), (hc, lc))
    nll = nll.reshape(-1)[:n]
    zl = zl.reshape(-1)[:n]
    valid = (labels.reshape(-1) >= 0).astype(jnp.float32)
    return jnp.sum(nll), jnp.sum(valid), jnp.sum(zl), nll.reshape(labels.shape)


def lm_loss(params, batch, cfg: ModelConfig, z_loss: float = 0.0):
    """batch: (B, T+1) token. Input = token[:, :-1], target = token[:, 1:]."""
    tokens, labels = batch[:, :-1], batch[:, 1:]
    h, emb = forward_hidden(params, tokens, cfg)
    total, count, zl, _ = chunked_cross_entropy(h, emb, labels, z_loss=z_loss)
    loss = total / jnp.maximum(count, 1.0)
    return loss + zl / jnp.maximum(count, 1.0), (total, count)


def masked_loss(params, inputs, labels, cfg: ModelConfig):
    """Loss untuk tugas dengan label ber-mask (MQAR): label -1 diabaikan."""
    h, emb = forward_hidden(params, inputs, cfg)
    total, count, _, nll = chunked_cross_entropy(h, emb, labels)
    return total / jnp.maximum(count, 1.0), (total, count)
