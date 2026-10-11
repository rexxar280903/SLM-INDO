"""Uji model: hitung parameter, kausalitas, varian KV, dan pola hybrid."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from slm.config import ALIASES, SCALES, build_model_config, grid_archs, hybrid_pattern
from slm.flops import inference_memory, param_count
from slm.model import count_params, init_params, logits_fn

ALL_ARCHS = sorted(set(grid_archs()) | set(ALIASES))


@pytest.mark.parametrize("scale", ["s10", "s25", "s50", "s125"])
@pytest.mark.parametrize("arch", ALL_ARCHS)
def test_param_count_matches_analytic(arch, scale):
    cfg = build_model_config(arch, scale)
    shapes = jax.eval_shape(lambda k: init_params(k, cfg), jax.random.PRNGKey(0))
    assert count_params(shapes) == param_count(cfg)["total"]


def test_reference_sizes():
    # Tabel KONSEP §8: B0 S10≈11,0M, S25≈24,0M, S50≈53,4M, S125≈97,5M (vocab 16.384)
    expect = {"s10": 11.0, "s25": 24.0, "s50": 53.4, "s125": 97.5}
    for s, m in expect.items():
        assert abs(param_count(build_model_config("b0", s))["total"] / 1e6 - m) < 0.15


@pytest.mark.parametrize("arch", ["b0", "b1", "b2", "b3", "b4", "h1-mla", "full-cla", "h3-cla", "h6-gqa"])
def test_causal_and_finite(arch):
    cfg = build_model_config(arch, "mini", attn_impl="xla", compute_dtype="float32", remat=False, gdn_chunk=16)
    params = init_params(jax.random.PRNGKey(1), cfg)
    tok = jax.random.randint(jax.random.PRNGKey(2), (2, 40), 0, cfg.vocab_size)
    tok2 = tok.at[:, 25:].set((tok[:, 25:] + 7) % cfg.vocab_size)   # ubah token masa depan
    l1, l2 = logits_fn(params, tok, cfg), logits_fn(params, tok2, cfg)
    assert np.all(np.isfinite(np.asarray(l1)))
    np.testing.assert_allclose(np.asarray(l1[:, :25]), np.asarray(l2[:, :25]), atol=1e-5)
    assert not np.allclose(np.asarray(l1[:, 25:]), np.asarray(l2[:, 25:]))


def test_remat_and_bf16_same_shapes():
    cfg = build_model_config("b3", "mini", attn_impl="xla", gdn_chunk=16)
    params = init_params(jax.random.PRNGKey(0), cfg)
    out = logits_fn(params, jnp.zeros((1, 32), jnp.int32), cfg)
    assert out.shape == (1, 32, cfg.vocab_size) and np.all(np.isfinite(np.asarray(out)))


def test_hybrid_patterns_documented():
    # Tabel penempatan di KONSEP §8 (posisi 1-indexed)
    pos = lambda L, r: [i + 1 for i, t in enumerate(hybrid_pattern(L, r)) if t == "attn"]
    assert pos(12, 3) == [4, 8, 12]
    assert pos(8, 3) == [4, 8]
    assert pos(10, 3) == [4, 7, 10]
    assert pos(14, 3) == [4, 7, 11, 14]
    assert pos(12, 6) == [6, 12]
    assert pos(10, 6) == [10]
    assert pos(12, 1) == [2, 4, 6, 8, 10, 12]


def test_kv_variants_half_cache():
    for scale in ("s10", "s25", "s50", "s125"):
        mha = inference_memory(build_model_config("full-mha", scale), 2048)["kv"]
        for v in ("gqa", "mla", "cla"):
            kv = inference_memory(build_model_config(f"full-{v}", scale), 2048)["kv"]
            assert kv == mha // 2, (scale, v)


def test_cla_receivers_have_no_kv_projection():
    cfg = build_model_config("full-cla", "s10")
    p = jax.eval_shape(lambda k: init_params(k, cfg), jax.random.PRNGKey(0))
    has_kv = ["wk" in p["layers"][i]["mixer"] for i in range(cfg.n_layers)]
    assert has_kv == [True, False] * (cfg.n_layers // 2)
