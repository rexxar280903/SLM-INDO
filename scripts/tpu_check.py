"""Uji kelayakan TPU (Keputusan B di peta alur): jalankan di notebook Kaggle TPU v5e-8 SEBELUM run pilot.

Memeriksa:
 1. perangkat: jumlah chip, jenis, versi JAX;
 2. numerik GDN di TPU: galat chunkwise vs rekuren pada presisi matmul "default" dan "float32";
 3. waktu kompilasi + throughput (token/detik, MFU) beberapa langkah training data sintetis
    untuk B0/B1/B2/B3 di S10 dan S25, dengan dua pilihan attention (pallas vs xla) dan
    dua pilihan presisi GDN. Hasil → tpu_check.json untuk mengkalibrasi anggaran TPU.

    python scripts/tpu_check.py --out /kaggle/working/tpu_check.json --steps 20
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P  # noqa: E402

from slm.config import SCALES, TrainConfig, build_model_config  # noqa: E402
from slm.flops import train_flops_per_token  # noqa: E402
from slm.gdn import gated_delta_chunkwise, gated_delta_recurrent  # noqa: E402
from slm.layers import l2_norm  # noqa: E402
from slm.model import init_params, lm_loss  # noqa: E402
from slm.train import make_optimizer, make_train_step, peak_flops  # noqa: E402


def gdn_numerics(precision: str) -> dict:
    key = jax.random.PRNGKey(0)
    ks = jax.random.split(key, 5)
    b, t, h, d = 2, 2048, 4, 64
    q = l2_norm(jax.random.normal(ks[0], (b, t, h, d))) * d ** -0.5
    k = l2_norm(jax.random.normal(ks[1], (b, t, h, d)))
    v = jax.random.normal(ks[2], (b, t, h, d))
    g = -jax.nn.softplus(jax.random.normal(ks[3], (b, t, h))) * 0.1
    beta = jax.nn.sigmoid(jax.random.normal(ks[4], (b, t, h)))
    with jax.default_matmul_precision("float32"):
        ref, _ = jax.jit(gated_delta_recurrent)(q, k, v, g, beta)
    with jax.default_matmul_precision(precision):
        out, _ = jax.jit(lambda *a: gated_delta_chunkwise(*a, chunk=64))(q, k, v, g, beta)
    err = jnp.abs(out - ref)
    return {"precision": precision, "max_abs_err": float(err.max()), "mean_abs_err": float(err.mean()),
            "ref_rms": float(jnp.sqrt(jnp.mean(ref ** 2)))}


def bench(arch: str, scale: str, steps: int, overrides: dict) -> dict:
    s = SCALES[scale]
    cfg = build_model_config(arch, scale, **overrides)
    tc = TrainConfig()
    tx = make_optimizer(tc)
    mesh = Mesh(np.array(jax.devices()), ("data",))
    rep, bsh = NamedSharding(mesh, P()), NamedSharding(mesh, P(None, "data"))
    params = jax.device_put(init_params(jax.random.PRNGKey(0), cfg), rep)
    opt = jax.device_put(tx.init(params), rep)
    step = make_train_step(lambda p, mb: lm_loss(p, mb, cfg), tx)
    micro = min(s.micro_batch, s.global_batch)
    accum = s.global_batch // micro
    rng = np.random.default_rng(0)
    batch = jax.device_put(rng.integers(0, cfg.vocab_size, (accum, micro, s.seq_len + 1), dtype=np.int32), bsh)
    t0 = time.time()
    params, opt, m = step(params, opt, batch, jnp.float32(1e-4))
    jax.block_until_ready(m["loss"])
    compile_s = time.time() - t0
    t0 = time.time()
    for _ in range(steps):
        params, opt, m = step(params, opt, batch, jnp.float32(1e-4))
    jax.block_until_ready(m["loss"])
    dt = (time.time() - t0) / steps
    tok_s = s.global_batch * s.seq_len / dt
    pk = peak_flops()
    return {"arch": arch, "scale": scale, **{k: str(v) for k, v in overrides.items()}, "compile_s": compile_s,
            "step_s": dt, "tok_per_s": tok_s, "mfu": tok_s * train_flops_per_token(cfg) / pk if pk else None,
            "loss": float(m["loss"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tpu_check.json")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--scales", nargs="*", default=["s10", "s25"])
    ap.add_argument("--archs", nargs="*", default=["b0", "b1", "b2", "b3"])
    a = ap.parse_args()
    dev = jax.devices()
    report = {"jax": jax.__version__, "backend": jax.default_backend(), "n_devices": len(dev),
              "device_kind": dev[0].device_kind, "gdn_numerics": [], "bench": []}
    print(json.dumps({k: report[k] for k in ("jax", "backend", "n_devices", "device_kind")}))
    for prec in ("default", "float32"):
        report["gdn_numerics"].append(gdn_numerics(prec))
        print(report["gdn_numerics"][-1])
    variants = [{"attn_impl": "auto", "gdn_precision": "float32"},
                {"attn_impl": "xla", "gdn_precision": "float32"},
                {"attn_impl": "auto", "gdn_precision": "default"}]
    for scale in a.scales:
        for arch in a.archs:
            for ov in variants:
                if ov["attn_impl"] == "xla" and arch == "b2":
                    continue
                if ov["gdn_precision"] == "default" and arch in ("b0", "b1"):
                    continue
                try:
                    r = bench(arch, scale, a.steps, ov)
                except Exception as e:      # catat kegagalan (mis. OOM, kernel tidak tersedia), lanjutkan
                    r = {"arch": arch, "scale": scale, **ov, "error": repr(e)[:300]}
                report["bench"].append(r)
                print(r, flush=True)
                json.dump(report, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
