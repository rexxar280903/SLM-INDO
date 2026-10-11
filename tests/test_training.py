"""Uji loop training: overfit satu batch, N2, checkpoint, data deterministik, rencana budget."""
import json
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from slm import checkpoint as ckpt
from slm.config import TrainConfig, build_model_config
from slm.data import BatchSchedule, TokenData
from slm.flops import train_flops_per_token
from slm.model import init_params, lm_loss
from slm.n2 import convert_opt_state, convert_params
from slm.train import make_optimizer, make_train_step, plan_run, run_lm

KW = dict(attn_impl="xla", gdn_chunk=16)


@pytest.mark.parametrize("arch", ["b0", "b2", "b3"])
def test_overfit_single_batch(arch):
    cfg = build_model_config(arch, "mini", **KW)
    tc = TrainConfig(weight_decay=0.0)
    tx = make_optimizer(tc)
    params = init_params(jax.random.PRNGKey(0), cfg)
    opt = tx.init(params)
    step = make_train_step(lambda p, mb: lm_loss(p, mb, cfg), tx)
    batch = jax.random.randint(jax.random.PRNGKey(1), (1, 4, 65), 0, 200)
    first = None
    for i in range(60):
        params, opt, m = step(params, opt, batch, jnp.float32(3e-3))
        first = first if first is not None else float(m["loss"])
    assert float(m["loss"]) < 0.3 * first, (first, float(m["loss"]))


def test_n2_conversion_keeps_and_transfers():
    lin = build_model_config("linear", "mini", **KW)
    tgt = build_model_config("h3-gqa", "mini", **KW)
    tc = TrainConfig()
    tx = make_optimizer(tc)
    params = init_params(jax.random.PRNGKey(0), lin)
    opt = tx.init(params)
    step = make_train_step(lambda p, mb: lm_loss(p, mb, lin), tx)
    batch = jax.random.randint(jax.random.PRNGKey(1), (1, 2, 33), 0, 500)
    params, opt, _ = step(params, opt, batch, jnp.float32(1e-3))
    new_p, report = convert_params(jax.device_get(params), lin, tgt, jax.random.PRNGKey(5))
    conv = tgt.attn_layers[0]
    assert np.array_equal(new_p["layers"][conv]["mixer"]["wq"], params["layers"][conv]["mixer"]["wq"])
    assert report[f"layers/{conv}/mixer/wk"] == "transform"            # mean-pool GQA
    keep = tgt.gdn_layers[0]
    assert np.array_equal(new_p["layers"][keep]["mixer"]["wg"], params["layers"][keep]["mixer"]["wg"])
    new_o = convert_opt_state(jax.device_get(opt), new_p, tx, report, keep_moments=True)
    adam = new_o[1]
    assert np.allclose(adam.mu["layers"][conv]["mixer"]["wq"], opt[1].mu["layers"][conv]["mixer"]["wq"])
    assert not np.any(adam.mu["layers"][conv]["mixer"]["wk"])           # tensor hasil transformasi direset
    step2 = make_train_step(lambda p, mb: lm_loss(p, mb, tgt), tx)
    p2, o2, m = step2(new_p, new_o, batch, jnp.float32(1e-3))
    assert np.isfinite(float(m["loss"]))


def test_checkpoint_roundtrip(tmp_path):
    cfg = build_model_config("b3", "mini", **KW)
    tx = make_optimizer(TrainConfig())
    params = init_params(jax.random.PRNGKey(0), cfg)
    opt = tx.init(params)
    base = ckpt.save(str(tmp_path), 7, params, opt, {"arch_now": "b3", "cum_flops": 1.0})
    p2, o2, meta = ckpt.restore(base, init_params(jax.random.PRNGKey(9), cfg), tx.init(params))
    assert meta["step"] == 7
    for a, b in zip(jax.tree_util.tree_leaves(params), jax.tree_util.tree_leaves(p2)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def _write_data(d, n_tokens=40_000, vocab=300):
    os.makedirs(d, exist_ok=True)
    rng = np.random.default_rng(0)
    for split, n in (("train", n_tokens), ("val", n_tokens // 4)):
        rng.integers(0, vocab, n, dtype=np.uint16).tofile(os.path.join(d, f"{split}.bin"))
    np.save(os.path.join(d, "token_bytes.npy"), np.full(vocab, 3, np.int32))
    json.dump({"vocab_size": vocab}, open(os.path.join(d, "meta.json"), "w"))


def test_batch_schedule_deterministic(tmp_path):
    _write_data(str(tmp_path))
    data = TokenData(str(tmp_path), "train", 64)
    a = BatchSchedule(data, 8, seed=3)
    b = BatchSchedule(data, 8, seed=3)
    np.testing.assert_array_equal(a.batch(5), b.batch(5))
    assert not np.array_equal(a.batch(5), BatchSchedule(data, 8, seed=4).batch(5))
    assert a.batch(5).shape == (8, 65)


def test_plan_compute_matched():
    tc = TrainConfig(arch="b3", scale="s25")
    plan = plan_run(tc)
    b0 = plan_run(TrainConfig(arch="b0", scale="s25"))
    assert abs(plan["budget_flops"] - b0["budget_flops"]) < 1
    assert plan["cum_flops_after"][-1] <= plan["budget_flops"] + 1
    n2 = plan_run(TrainConfig(arch="b3", scale="s25", n2_tau=0.25))
    assert [p["arch"] for p in n2["phases"]] == ["linear", "h3-mha"]
    assert n2["cum_flops_after"][-1] <= n2["budget_flops"] + 1
    assert len(n2["eval_steps"]) == 10


def test_run_lm_end_to_end_with_resume(tmp_path):
    _write_data(str(tmp_path / "data"))
    common = dict(arch="b3", scale="tiny", data_dir=str(tmp_path / "data"), out_dir=str(tmp_path / "runs"),
                  eval_tokens=4096, log_every=5, model_overrides=dict(KW))
    s1 = run_lm(TrainConfig(max_steps=12, **common))
    assert s1["final_eval"]["val_bpb"] > 0
    s2 = run_lm(TrainConfig(max_steps=20, **common))                   # melanjutkan dari checkpoint langkah 12
    lines = open(tmp_path / "runs" / s2["run_id"] / "metrics.jsonl").read().strip().splitlines()
    steps = [json.loads(l)["step"] for l in lines]
    assert 20 in steps and steps.count(13) <= 1


def test_run_lm_n2_switch(tmp_path):
    _write_data(str(tmp_path / "data"))
    tc = TrainConfig(arch="b3", scale="tiny", n2_tau=0.25, data_dir=str(tmp_path / "data"),
                     out_dir=str(tmp_path / "runs"), eval_tokens=4096, log_every=5, model_overrides=dict(KW))
    plan = plan_run(tc, {"vocab_size": 300, **KW})
    tc.max_steps = plan["phases"][0]["steps"] + 3
    s = run_lm(tc)
    assert s["arch_final"] == "h3-mha"


def test_mqar_generator_labels_are_values_of_queried_keys():
    from slm.mqar import make_mqar_batch
    rng = np.random.default_rng(0)
    x, y = make_mqar_batch(rng, 4, 256, 16)
    for b in range(4):
        ctx = dict(zip(x[b, 0:32:2].tolist(), x[b, 1:32:2].tolist()))
        pos = np.where(y[b] >= 0)[0]
        assert len(pos) == 16 and np.all(pos >= 32)
        for p in pos:
            assert ctx[int(x[b, p])] == y[b, p] == x[b, p + 1]


def test_dataprep_filters_and_ngrams():
    from slm.dataprep import NgramIndex, quality_reason, split_of, doc_hash
    spam = "Daftar slot gacor hari ini maxwin, situs slot terpercaya dengan link alternatif dan deposit pulsa. " * 3
    assert quality_reason(spam) == "spam_judi"
    news = ("Pemerintah Kota Bandar Lampung meresmikan perpustakaan baru di pusat kota pada hari Senin "
            "untuk meningkatkan minat baca warga dan pelajar di wilayah tersebut.")
    assert quality_reason(news) is None
    ev = "kereta tiba di stasiun pada sore hari setelah perjalanan panjang melewati sawah dan sungai yang luas"
    idx = NgramIndex.build([ev])
    assert idx.matches("Berita: " + ev + " lalu penumpang turun.")
    assert not idx.matches(news)
    assert split_of(doc_hash(news)) in ("train", "val")
