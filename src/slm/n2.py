"""N2 · linear-first warm-start: konversi layer Gated DeltaNet → full attention di titik alih τ.

Aturan transfer (paper §3.5, keputusan D32):
- Layer yang TIDAK dikonversi, embedding, FFN, dan norm: disalin apa adanya (fungsi identik).
- Layer yang dikonversi, target MHA: W_Q, W_K, W_V, W_O GDN dipakai ulang langsung
  (bentuk d×(H·E) sama). Fungsinya TIDAK sama: di GDN q/k/v melewati conv pendek,
  SiLU, dan L2-norm; di attention melewati RoPE. Karena itu transfer vs inisialisasi
  acak adalah ablation (`transfer=False`).
- Target GQA: W_K/W_V di-*mean-pool* per kelompok head (resep uptraining GQA, Ainslie dkk. 2023).
- Target MLA: tidak ada pemetaan yang wajar → W_Q/W_O ditransfer, sisanya diinisialisasi baru.
- Parameter khusus GDN (conv, W_a, W_b, A_log, dt_bias, W_g, o_norm) dibuang.
- Momen AdamW: tensor yang tidak berubah selalu dilanjutkan; tensor hasil transfer
  dilanjutkan bila keep_moments=True; tensor baru/hasil transformasi direset ke nol.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .config import ModelConfig
from .model import init_attention


def _mean_pool_heads(w, n_heads: int, kv_heads: int, head_dim: int):
    d = w.shape[0]
    w = w.reshape(d, kv_heads, n_heads // kv_heads, head_dim).mean(axis=2)
    return w.reshape(d, kv_heads * head_dim)


def convert_params(params, old_cfg: ModelConfig, new_cfg: ModelConfig, key, transfer: bool = True):
    """Mengembalikan (params_baru, laporan) dengan laporan: path → 'copy'|'transfer'|'transform'|'new'."""
    assert old_cfg.n_layers == new_cfg.n_layers and old_cfg.d_model == new_cfg.d_model
    out_std = new_cfg.init_std / np.sqrt(2 * new_cfg.n_layers)
    report = {}
    layers = []
    for i, (old_t, new_t) in enumerate(zip(old_cfg.layer_types, new_cfg.layer_types)):
        lp = params["layers"][i]
        if old_t == new_t:
            layers.append(lp)
            report[f"layers/{i}"] = "copy"
            continue
        if not (old_t == "gdn" and new_t == "attn"):
            raise ValueError("N2 hanya mendukung konversi gdn → attn")
        fresh = init_attention(jax.random.fold_in(key, i), new_cfg, i, out_std)
        mixer = dict(fresh)
        src = lp["mixer"]
        tag = {name: "new" for name in fresh}
        if transfer:
            for name in ("wq", "wo"):
                if name in fresh and fresh[name].shape == src[name].shape:
                    mixer[name], tag[name] = src[name], "transfer"
            for name in ("wk", "wv"):
                if name not in fresh:
                    continue
                if fresh[name].shape == src[name].shape:
                    mixer[name], tag[name] = src[name], "transfer"
                elif new_cfg.mla_latent == 0:
                    mixer[name] = _mean_pool_heads(src[name], new_cfg.n_heads, new_cfg.kv_heads, new_cfg.head_dim)
                    tag[name] = "transform"
        layers.append({"norm1": lp["norm1"], "mixer": mixer, "norm2": lp["norm2"], "ffn": lp["ffn"]})
        report[f"layers/{i}"] = "copy(norm,ffn)"
        for name, t in tag.items():
            report[f"layers/{i}/mixer/{name}"] = t
    new_params = {"embed": params["embed"], "layers": layers, "norm_f": params["norm_f"]}
    return new_params, report


def _path_str(path) -> str:
    parts = []
    for p in path:
        parts.append(str(p.key) if hasattr(p, "key") else str(p.idx) if hasattr(p, "idx") else str(p))
    return "/".join(parts)


def _flat(tree) -> dict:
    leaves, _ = jax.tree_util.tree_flatten_with_path(tree)
    return {_path_str(p): v for p, v in leaves}


def convert_opt_state(old_state, new_params, tx, report: dict, keep_moments: bool):
    """Membangun state optimizer untuk parameter baru dan menyalin momen yang boleh dilanjutkan."""
    new_state = tx.init(new_params)

    def keep(path: str) -> bool:
        parts = path.split("/")
        layer_key = "/".join(parts[:2])
        status = report.get(f"{layer_key}/mixer/{parts[-1]}") if len(parts) > 3 and parts[2] == "mixer" else None
        if status is None:
            return True                       # embed, norm, FFN, layer yang tidak dikonversi
        return status == "transfer" and keep_moments

    def merge(old_adam, new_adam):
        olds = {"mu": _flat(old_adam.mu), "nu": _flat(old_adam.nu)}
        out = {}
        for field in ("mu", "nu"):
            def pick(path, new_leaf, field=field):
                ps = _path_str(path)
                old_leaf = olds[field].get(ps)
                if old_leaf is not None and old_leaf.shape == new_leaf.shape and keep(ps):
                    return old_leaf
                return jnp.zeros_like(new_leaf)
            out[field] = jax.tree_util.tree_map_with_path(pick, getattr(new_adam, field))
        return new_adam._replace(count=old_adam.count, mu=out["mu"], nu=out["nu"])

    def walk(old, new):
        if isinstance(new, optax.ScaleByAdamState):
            return merge(old, new)
        if isinstance(new, tuple) and not hasattr(new, "_fields"):
            return tuple(walk(o, n) for o, n in zip(old, new))
        return new

    return walk(old_state, new_state)
