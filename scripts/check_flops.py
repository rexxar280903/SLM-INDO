"""Validasi silang FLOPs analitik (slm.flops) dengan cost analysis XLA.

XLA menghitung badan loop (lax.scan) hanya sekali, jadi perbandingan dilakukan pada
konfigurasi tanpa loop: forward saja, satu chunk GDN sepanjang urutan (chunk = T),
dan attention eksplisit. Untuk attention, rumus analitik memakai (T+1)/2 key per
query, sedangkan kernel eksplisit menghitung T×T penuh; skrip ini mengoreksinya.

    python scripts/check_flops.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from slm.config import build_model_config  # noqa: E402
from slm.flops import forward_flops_per_token  # noqa: E402
from slm.model import forward_hidden, init_params  # noqa: E402


def xla_flops(cfg, t):
    params = jax.eval_shape(lambda k: init_params(k, cfg), jax.random.PRNGKey(0))
    tok = jax.ShapeDtypeStruct((1, t), jnp.int32)

    def f(p, x):
        h, emb = forward_hidden(p, x, cfg)
        return jnp.einsum("btd,vd->btv", h, emb)

    cost = jax.jit(f).lower(params, tok).compile().cost_analysis()
    cost = cost[0] if isinstance(cost, (list, tuple)) else cost
    return cost.get("flops", float("nan")) / t


def main():
    t = 256
    print(f"{'arsitektur':<12}{'analitik':>14}{'XLA':>14}{'selisih':>10}")
    for arch in ("b0", "b1", "b2", "b3", "full-mla", "full-cla"):
        cfg = build_model_config(arch, "s10", attn_impl="xla", remat=False, gdn_chunk=t, compute_dtype="float32",
                                 seq_len=t)
        ana = forward_flops_per_token(cfg, t)
        # Koreksi attention: kernel eksplisit menghitung seluruh T key (bukan (T+1)/2)
        n_attn = len(cfg.attn_layers)
        corr = n_attn * 2 * 2 * cfg.n_heads * cfg.head_dim * (t - (t + 1) / 2)
        # Koreksi GDN: rumus memakai biaya solve C·2e; kernel XLA triangular solve tidak dihitung sebagai flops
        expected = ana["total"] + corr
        x = xla_flops(cfg, t)
        print(f"{arch:<12}{expected/1e6:>13.2f}M{x/1e6:>13.2f}M{(x/expected-1)*100:>9.1f}%")


if __name__ == "__main__":
    main()
