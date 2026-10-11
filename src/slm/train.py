"""Training loop (TPU v5e-8 / CPU) — satu run = satu (arsitektur, skala, seed).

Contoh:
    python -m slm.train --arch b0 --scale s10 --seed 0 --data-dir /kaggle/input/slm-data
    python -m slm.train --arch b0 --scale mini --synthetic --max-steps 50        # smoke test
    python -m slm.train --arch h3-mha --scale s25 --n2-tau 0.25 --data-dir ...   # N2
    python -m slm.train --task mqar --arch b3 --scale s10 --lr 1e-3              # probe MQAR

Alur satu langkah (lihat dashboard bagian "Satu run"):
  batch (akumulasi × micro × T+1) → jax.value_and_grad(loss) per micro-batch (lax.scan)
  → rata-rata gradien → clip norm 1,0 → AdamW (decoupled) → parameter baru.
JAX: `jax.jit` mengompilasi fungsi langkah sekali (beberapa detik–menit), lalu setiap
panggilan berikutnya langsung jalan di TPU. `NamedSharding` membagi batch ke 8 chip
(data parallel); parameter direplikasi; XLA menyisipkan all-reduce gradien otomatis.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import platform
import subprocess
import sys
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from . import checkpoint as ckpt
from .config import SCALES, TrainConfig, build_model_config, lr_at
from .data import BatchSchedule, SyntheticData, TokenData, eval_batches, load_token_bytes
from .flops import inference_memory, param_count, train_flops_per_token
from .model import chunked_cross_entropy, count_params, forward_hidden, init_params, lm_loss, masked_loss

PEAK_FLOPS = {"TPU v5 lite": 197e12, "TPU v5e": 197e12, "TPU v4": 275e12, "Tesla T4": 65e12}

# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

NO_DECAY = ("embed", "norm1", "norm2", "norm_f", "o_norm", "kv_norm", "A_log", "dt_bias",
            "conv_q", "conv_k", "conv_v")


def decay_mask(params):
    """Weight decay hanya untuk matriks proyeksi (bukan embedding, norm, gerbang GDN, conv)."""
    def rule(path, x):
        name = str(path[-1].key) if hasattr(path[-1], "key") else ""
        return x.ndim >= 2 and name not in NO_DECAY
    return jax.tree_util.tree_map_with_path(rule, params)


def make_optimizer(tc: TrainConfig) -> optax.GradientTransformation:
    """Arah update AdamW; LR dikalikan di luar agar jadwal bisa mengikuti progres FLOPs."""
    return optax.chain(
        optax.clip_by_global_norm(tc.grad_clip),
        optax.scale_by_adam(b1=tc.beta1, b2=tc.beta2, eps=tc.eps),
        optax.add_decayed_weights(tc.weight_decay, mask=decay_mask),
    )


# ---------------------------------------------------------------------------
# Rencana budget (D30): FLOPs budget = FLOPs/token arsitektur anchor (B0) × token referensi
# ---------------------------------------------------------------------------

def plan_run(tc: TrainConfig, overrides: dict | None = None) -> dict:
    s = SCALES[tc.scale]
    overrides = overrides or {}
    batch = tc.global_batch or s.global_batch
    tps = batch * s.seq_len
    ref_steps = math.ceil(s.ref_tokens / tps)
    anchor = build_model_config(tc.anchor_arch, s, **overrides)
    budget = train_flops_per_token(anchor) * tps * ref_steps
    phases = []
    if tc.n2_tau > 0:
        lin = build_model_config("linear", s, **overrides)
        tgt = build_model_config(tc.n2_target, s, **overrides)
        f_lin, f_tgt = train_flops_per_token(lin) * tps, train_flops_per_token(tgt) * tps
        steps1 = int(round(tc.n2_tau * ref_steps))
        if tc.budget_mode == "flops":
            steps2 = int((budget - steps1 * f_lin) // f_tgt)
        else:
            steps2 = ref_steps - steps1
            budget = steps1 * f_lin + steps2 * f_tgt
        phases = [{"arch": "linear", "steps": steps1, "flops_per_step": f_lin},
                  {"arch": tc.n2_target, "steps": steps2, "flops_per_step": f_tgt}]
    else:
        mcfg = build_model_config(tc.arch, s, **overrides)
        f = train_flops_per_token(mcfg) * tps
        steps = int(budget // f) if tc.budget_mode == "flops" else ref_steps
        if tc.budget_mode != "flops":
            budget = steps * f
        phases = [{"arch": tc.arch, "steps": steps, "flops_per_step": f}]
    per_step = np.concatenate([np.full(ph["steps"], ph["flops_per_step"]) for ph in phases])
    cum_after = np.cumsum(per_step)
    total_steps = len(per_step)
    warm = max(tc.warmup_frac, tc.min_warmup_steps / max(1, total_steps))
    peak = (tc.peak_lr or s.peak_lr) * tc.lr_multiplier
    # LR langkah i memakai progres FLOPs di AKHIR langkah i → langkah pertama sudah > 0,
    # langkah terakhir tepat di min_lr_ratio · peak.
    lr = np.array([lr_at(c / cum_after[-1], peak, warm, tc.min_lr_ratio) for c in cum_after])
    eval_steps = sorted({int(np.searchsorted(cum_after, k / tc.eval_points * cum_after[-1] * (1 - 1e-9)))
                         for k in range(1, tc.eval_points + 1)})
    return {"scale": s.name, "global_batch": batch, "tokens_per_step": tps, "ref_steps": ref_steps,
            "budget_flops": budget, "anchor_arch": tc.anchor_arch, "phases": phases,
            "total_steps": total_steps, "total_tokens": total_steps * tps, "peak_lr": peak,
            "warmup_frac": warm, "lr": lr, "cum_flops_after": cum_after, "eval_steps": eval_steps}


# ---------------------------------------------------------------------------
# Fungsi langkah (dikompilasi jit)
# ---------------------------------------------------------------------------

def make_train_step(loss_fn, tx):
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    @partial(jax.jit, donate_argnums=(0, 1))
    def step(params, opt_state, batch, lr):
        def micro(carry, mb):
            g_acc, loss_acc = carry
            (loss, _), g = grad_fn(params, mb)
            return (jax.tree_util.tree_map(jnp.add, g_acc, g), loss_acc + loss), None

        zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
        (grads, loss_sum), _ = jax.lax.scan(micro, (zeros, jnp.float32(0.0)), batch)
        n = jax.tree_util.tree_leaves(batch)[0].shape[0]
        grads = jax.tree_util.tree_map(lambda g: g / n, grads)
        gnorm = jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(grads)))
        updates, opt_state = tx.update(grads, opt_state, params)
        params = jax.tree_util.tree_map(lambda p, u: p - lr * u, params, updates)
        return params, opt_state, {"loss": loss_sum / n, "grad_norm": gnorm}

    return step


def make_eval_step(mcfg):
    @jax.jit
    def step(params, batch, token_bytes):
        tokens, labels = batch[:, :-1], batch[:, 1:]
        h, emb = forward_hidden(params, tokens, mcfg)
        total, count, _, _ = chunked_cross_entropy(h, emb, labels)
        return total, count, jnp.sum(token_bytes[labels].astype(jnp.float32))
    return step


# ---------------------------------------------------------------------------
# Utilitas
# ---------------------------------------------------------------------------

def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
                                       cwd=os.path.dirname(__file__)).decode().strip()
    except Exception:
        return "unknown"


def environment() -> dict:
    devs = jax.devices()
    return {"jax": jax.__version__, "optax": optax.__version__, "python": platform.python_version(),
            "backend": jax.default_backend(), "device_kind": devs[0].device_kind, "n_devices": len(devs),
            "git_commit": git_commit()}


def peak_flops() -> float | None:
    kind = jax.devices()[0].device_kind
    for k, v in PEAK_FLOPS.items():
        if k.lower() in kind.lower():
            return v * jax.device_count()
    return None


class Logger:
    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)

    def write(self, name: str, record: dict):
        with open(os.path.join(self.run_dir, name), "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _to_jsonable(plan: dict) -> dict:
    out = {k: v for k, v in plan.items() if k not in ("lr", "cum_flops_after")}
    out["lr_first_last"] = [float(plan["lr"][0]), float(plan["lr"][-1])]
    return out


# ---------------------------------------------------------------------------
# Training bahasa (tugas utama)
# ---------------------------------------------------------------------------

def run_lm(tc: TrainConfig) -> dict:
    s = SCALES[tc.scale]
    ov = dict(tc.model_overrides)
    batch_size = tc.global_batch or s.global_batch

    # Data (dibaca dulu: ukuran vocab data menentukan FLOPs LM head dalam rencana budget)
    if tc.synthetic_data:
        vocab = int(ov.get("vocab_size", 16_384))
        train_data = SyntheticData(vocab, s.seq_len, batch_size, tc.seed)
        val_data, token_bytes = None, None
    else:
        train_data = BatchSchedule(TokenData(tc.data_dir, "train", s.seq_len), batch_size, tc.seed)
        val_data = TokenData(tc.data_dir, "val", s.seq_len)
        token_bytes = load_token_bytes(tc.data_dir)
        vocab = int(train_data.data.meta.get("vocab_size", 16_384))
    ov["vocab_size"] = vocab

    plan = plan_run(tc, ov)
    phases = plan["phases"]
    switch_step = phases[0]["steps"] if len(phases) == 2 else None
    total_steps = plan["total_steps"] if tc.max_steps is None else min(plan["total_steps"], tc.max_steps)

    run_id = f"{tc.arch if tc.n2_tau == 0 else 'n2-' + tc.n2_target + f'-tau{tc.n2_tau:g}'}-{tc.scale}-seed{tc.seed}-{tc.budget_mode}"
    if tc.lr_multiplier != 1.0:
        run_id += f"-lr{tc.lr_multiplier:g}"
    run_dir = os.path.join(tc.out_dir, run_id)
    log = Logger(run_dir)
    ckpt_dir = os.path.join(run_dir, "ckpt")

    n_dev = jax.device_count()
    mesh = Mesh(np.array(jax.devices()), ("data",))
    replicated = NamedSharding(mesh, P())
    batch_sharding = NamedSharding(mesh, P(None, "data"))
    micro = tc.micro_batch or min(s.micro_batch, batch_size)
    assert batch_size % micro == 0 and micro % n_dev == 0, "micro batch harus habis dibagi jumlah chip"
    accum = batch_size // micro

    def model_cfg(arch):
        return build_model_config(arch, s, **ov)

    tx = make_optimizer(tc)
    start_step, cum_flops = 0, 0.0
    arch_now = phases[0]["arch"]
    mcfg = model_cfg(arch_now)
    params = init_params(jax.random.PRNGKey(tc.seed), mcfg)
    opt_state = tx.init(params)

    latest = ckpt.latest(ckpt_dir)
    if latest:
        meta = ckpt.read_meta(latest)
        arch_now = meta["arch_now"]
        mcfg = model_cfg(arch_now)
        params = init_params(jax.random.PRNGKey(tc.seed), mcfg)
        params, opt_state, meta = ckpt.restore(latest, params, tx.init(params))
        start_step, cum_flops = meta["step"], meta["cum_flops"]
        print(f"[resume] {latest} langkah {start_step}, arsitektur {arch_now}")
    else:
        with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump({"train": tc.to_dict(), "model": mcfg.to_dict(), "plan": _to_jsonable(plan),
                       "params": param_count(mcfg), "env": environment(),
                       "data_meta": getattr(getattr(train_data, "data", None), "meta", {})},
                      f, indent=2, ensure_ascii=False)

    params = jax.device_put(params, replicated)
    opt_state = jax.device_put(opt_state, replicated)
    loss_fn = lambda p, mb, cfg=mcfg: lm_loss(p, mb, cfg, tc.z_loss)
    train_step = make_train_step(loss_fn, tx)
    eval_step = make_eval_step(mcfg)
    tb = jnp.asarray(token_bytes if token_bytes is not None else np.ones(mcfg.vocab_size, np.float32))

    def evaluate(step):
        if val_data is None:
            return None
        tot = cnt = nb = 0.0
        for vb in eval_batches(val_data, micro, tc.eval_tokens):
            a, b, c = eval_step(params, jax.device_put(vb, NamedSharding(mesh, P("data"))), tb)
            tot, cnt, nb = tot + float(a), cnt + float(b), nb + float(c)
        rec = {"step": step, "cum_flops": cum_flops, "progress": cum_flops / plan["budget_flops"],
               "val_loss": tot / cnt, "val_bpb": tot / math.log(2) / nb, "val_tokens": cnt}
        log.write("eval.jsonl", rec)
        print(f"[eval] langkah {step}: val_loss {rec['val_loss']:.4f} · bpb {rec['val_bpb']:.4f}")
        return rec

    peak = peak_flops()
    t_last, last_ckpt = time.time(), time.time()
    tokens_since = 0
    eval_steps = set(plan["eval_steps"])
    print(f"[run] {run_id}: {total_steps} langkah · batch {batch_size}×{s.seq_len} (akumulasi {accum}) · "
          f"{param_count(mcfg)['total']/1e6:.2f}M param · budget {plan['budget_flops']:.3e} FLOPs · {n_dev} perangkat")

    for step in range(start_step, total_steps):
        if switch_step is not None and step == switch_step and arch_now == "linear":
            from .n2 import convert_opt_state, convert_params
            ckpt.save(ckpt_dir, step, params, opt_state, {"arch_now": arch_now, "cum_flops": cum_flops,
                                                          "note": "sebelum konversi N2"})
            new_cfg = model_cfg(phases[1]["arch"])
            host_p, host_o = jax.device_get(params), jax.device_get(opt_state)
            new_p, report = convert_params(host_p, mcfg, new_cfg, jax.random.PRNGKey(10_000 + tc.seed),
                                           transfer=tc.n2_transfer)
            new_o = convert_opt_state(host_o, new_p, tx, report, tc.n2_keep_moments)
            log.write("n2.jsonl", {"step": step, "report": report})
            arch_now, mcfg = phases[1]["arch"], new_cfg
            params, opt_state = jax.device_put(new_p, replicated), jax.device_put(new_o, replicated)
            loss_fn = lambda p, mb, cfg=mcfg: lm_loss(p, mb, cfg, tc.z_loss)
            train_step, eval_step = make_train_step(loss_fn, tx), make_eval_step(mcfg)
            print(f"[n2] konversi linear → {arch_now} pada langkah {step}")

        batch = train_data.batch(step).reshape(accum, micro, -1)
        batch = jax.device_put(batch, batch_sharding)
        params, opt_state, metrics = train_step(params, opt_state, batch, jnp.float32(plan["lr"][step]))
        cum_flops = float(plan["cum_flops_after"][step])
        tokens_since += batch_size * s.seq_len

        if (step + 1) % tc.log_every == 0 or step == start_step or step + 1 == total_steps:
            m = jax.device_get(metrics)
            now = time.time()
            dt = now - t_last
            tok_s = tokens_since / dt if step != start_step else float("nan")
            rec = {"step": step + 1, "loss": float(m["loss"]), "grad_norm": float(m["grad_norm"]),
                   "lr": float(plan["lr"][step]), "cum_flops": cum_flops,
                   "progress": cum_flops / plan["budget_flops"], "tokens": (step + 1) * batch_size * s.seq_len,
                   "tok_per_s": tok_s, "arch_now": arch_now,
                   "epoch": train_data.epoch_of(step)}
            if peak and step != start_step:
                rec["mfu"] = tok_s * train_flops_per_token(mcfg) / peak
            log.write("metrics.jsonl", rec)
            print(f"langkah {step+1}/{total_steps} loss {rec['loss']:.4f} gnorm {rec['grad_norm']:.3f} "
                  f"lr {rec['lr']:.2e} tok/s {tok_s:,.0f}")
            if not math.isfinite(rec["loss"]):
                raise FloatingPointError(f"loss tidak hingga pada langkah {step+1} — hentikan run (kriteria divergensi)")
            t_last, tokens_since = now, 0

        if step in eval_steps:
            evaluate(step + 1)
        if time.time() - last_ckpt > tc.ckpt_every_minutes * 60:
            ckpt.save(ckpt_dir, step + 1, params, opt_state, {"arch_now": arch_now, "cum_flops": cum_flops})
            last_ckpt = time.time()

    final_eval = evaluate(total_steps)
    ckpt.save(ckpt_dir, total_steps, params, opt_state, {"arch_now": arch_now, "cum_flops": cum_flops, "final": True})
    summary = {"run_id": run_id, "steps": total_steps, "cum_flops": cum_flops, "arch_final": arch_now,
               "params": param_count(mcfg), "params_actual": count_params(params),
               "cache_2048": inference_memory(mcfg, 2048), "final_eval": final_eval}
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


# ---------------------------------------------------------------------------
# Probe MQAR
# ---------------------------------------------------------------------------

def run_mqar(tc: TrainConfig, steps: int = 6000, batch: int = 128, lr: float = 1e-3) -> dict:
    from .mqar import EVAL_GRID, MQAR_VOCAB, TRAIN_MIX, make_mqar_batch

    s = SCALES[tc.scale]
    ov = dict(tc.model_overrides)
    mcfg = build_model_config(tc.arch, s, vocab_size=MQAR_VOCAB, seq_len=1024, **ov)
    run_dir = os.path.join(tc.out_dir, f"mqar-{tc.arch}-{tc.scale}-seed{tc.seed}-lr{lr:g}")
    log = Logger(run_dir)
    tx = make_optimizer(dataclasses.replace(tc, weight_decay=0.1))
    params = init_params(jax.random.PRNGKey(tc.seed), mcfg)
    opt_state = tx.init(params)
    loss_fn = lambda p, mb: masked_loss(p, mb[0], mb[1], mcfg)
    train_step = make_train_step(loss_fn, tx)
    rng = np.random.default_rng(tc.seed)
    steps = tc.max_steps or steps
    for step in range(steps):
        seq, nkv = TRAIN_MIX[rng.integers(len(TRAIN_MIX))]
        x, y = make_mqar_batch(rng, batch, seq, nkv)
        lr_now = lr_at(step / steps, lr, 0.05, 0.0)
        params, opt_state, m = train_step(params, opt_state, (x[None], y[None]), jnp.float32(lr_now))
        if (step + 1) % 200 == 0:
            log.write("metrics.jsonl", {"step": step + 1, "loss": float(m["loss"])})

    @jax.jit
    def predict(p, x):
        h, emb = forward_hidden(p, x, mcfg)
        return jnp.argmax(jnp.einsum("btd,vd->btv", h, emb), axis=-1)

    eval_rng = np.random.default_rng(12345)          # set uji MQAR identik untuk semua arsitektur
    results = []
    for seq, nkv in EVAL_GRID:
        x, y = make_mqar_batch(eval_rng, 256, seq, nkv)
        pred = np.asarray(predict(params, x))
        mask = y >= 0
        results.append({"seq_len": seq, "num_kv": nkv, "acc": float((pred[mask] == y[mask]).mean())})
    summary = {"arch": tc.arch, "lr": lr, "steps": steps, "results": results,
               "mean_acc": float(np.mean([r["acc"] for r in results]))}
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_value(v: str):
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return {"true": True, "false": False}.get(v.lower(), v)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Training SLM-INDO")
    ap.add_argument("--arch", default="b0")
    ap.add_argument("--scale", default="s10")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget-mode", default="flops", choices=["flops", "tokens"])
    ap.add_argument("--lr-mult", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=None, help="LR puncak eksplisit (sweep/MQAR)")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--data-dir", default="")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--out-dir", default="runs")
    ap.add_argument("--n2-tau", type=float, default=0.0)
    ap.add_argument("--n2-target", default="h3-mha")
    ap.add_argument("--n2-reset-moments", action="store_true")
    ap.add_argument("--n2-no-transfer", action="store_true", help="ablation: layer attention baru diinisialisasi acak")
    ap.add_argument("--task", default="lm", choices=["lm", "mqar"])
    ap.add_argument("--global-batch", type=int, default=None)
    ap.add_argument("--micro-batch", type=int, default=None)
    ap.add_argument("--eval-tokens", type=int, default=4_194_304)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--set", nargs="*", default=[], help="override model, mis. attn_impl=xla remat=false")
    a = ap.parse_args(argv)
    overrides = dict(kv.split("=", 1) for kv in a.set)
    overrides = {k: _parse_value(v) for k, v in overrides.items()}
    tc = TrainConfig(arch=a.arch, scale=a.scale, seed=a.seed, budget_mode=a.budget_mode, lr_multiplier=a.lr_mult,
                     peak_lr=a.lr, max_steps=a.max_steps, data_dir=a.data_dir, synthetic_data=a.synthetic,
                     out_dir=a.out_dir, n2_tau=a.n2_tau, n2_target=a.n2_target,
                     n2_keep_moments=not a.n2_reset_moments, n2_transfer=not a.n2_no_transfer, task=a.task, global_batch=a.global_batch,
                     micro_batch=a.micro_batch, eval_tokens=a.eval_tokens, log_every=a.log_every,
                     model_overrides=overrides)
    if tc.task == "mqar":
        return run_mqar(tc, lr=a.lr or 1e-3)
    return run_lm(tc)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
