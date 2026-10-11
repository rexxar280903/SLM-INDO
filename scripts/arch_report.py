"""Laporan arsitektur: parameter, FLOPs/token, cache inferensi, dan rencana run per skala.

    python scripts/arch_report.py               # tabel markdown ke stdout
    python scripts/arch_report.py --json out.json

Semua angka ANALITIK (dari konfigurasi), bukan hasil training/benchmark.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from slm.config import ALIASES, SCALES, TrainConfig, build_model_config, grid_archs  # noqa: E402
from slm.flops import forward_flops_per_token, inference_memory, param_count, train_flops_per_token  # noqa: E402
from slm.train import plan_run  # noqa: E402

MAIN_SCALES = ("s10", "s25", "s50", "s125")


def row(arch: str, scale: str) -> dict:
    cfg = build_model_config(arch, scale)
    pc = param_count(cfg)
    b0 = build_model_config("b0", scale)
    plan = plan_run(TrainConfig(arch=arch, scale=scale))
    mem = inference_memory(cfg, 2048)
    mem8k = inference_memory(cfg, 8192)
    attn = [i + 1 for i in cfg.attn_layers]
    n_a, n_g = len(cfg.attn_layers), len(cfg.gdn_layers)
    fwd = forward_flops_per_token(cfg)
    return {
        "arch": arch, "canonical": ALIASES.get(arch, arch), "scale": scale,
        "d_model": cfg.d_model, "n_layers": cfg.n_layers, "n_heads": cfg.n_heads, "kv_heads": cfg.kv_heads,
        "ffn_dim": cfg.ffn_dim, "mla_latent": cfg.mla_latent, "cla_share": cfg.cla_share, "window": cfg.window,
        "attn_positions": attn, "n_attn": n_a, "n_gdn": n_g,
        "actual_ratio": (f"{n_g}:{n_a}" if n_a else "linear") if n_g else "full",
        "params_total": pc["total"], "params_non_embedding": pc["non_embedding"],
        "fwd_flops_per_token": fwd["total"], "fwd_mixer_share": fwd["mixers"] / fwd["total"],
        "train_flops_per_token": train_flops_per_token(cfg),
        "rel_cost_vs_b0": train_flops_per_token(cfg) / train_flops_per_token(b0),
        "kv_bytes_per_token": mem["kv_per_token"], "kv_2048_mb": mem["kv"] / 1e6,
        "state_mb": mem["state"] / 1e6, "conv_mb": mem["conv"] / 1e6,
        "total_cache_2048_mb": mem["total"] / 1e6, "total_cache_8192_mb": mem8k["total"] / 1e6,
        "budget_flops": plan["budget_flops"], "steps": plan["total_steps"], "tokens": plan["total_tokens"],
        "global_batch": plan["global_batch"], "peak_lr": plan["peak_lr"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    archs = ["b0", "b1", "b2", "b3", "b4"] + [x for x in grid_archs() if x not in ALIASES.values()]
    rows = [row(arch, s) for s in MAIN_SCALES for arch in archs]
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump({"scales": {k: vars(v) | {"ffn_dim": v.ffn_dim} for k, v in SCALES.items() if k in MAIN_SCALES},
                       "rows": rows}, f, indent=1)
    print("| skala | arsitektur | attention di | rasio aktual | param total (M) | non-emb (M) | FLOPs latih/token | relatif B0 | KV/token (B) | cache 2k (MB) | langkah | token (M) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['scale']} | {r['arch']} | {','.join(map(str, r['attn_positions'])) or '-'} | {r['actual_ratio']} | "
              f"{r['params_total']/1e6:.2f} | {r['params_non_embedding']/1e6:.2f} | {r['train_flops_per_token']/1e6:.1f}M | "
              f"{r['rel_cost_vs_b0']:.3f} | {r['kv_bytes_per_token']} | {r['total_cache_2048_mb']:.2f} | {r['steps']} | {r['tokens']/1e6:.0f} |")


if __name__ == "__main__":
    main()
