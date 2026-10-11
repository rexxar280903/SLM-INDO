"""Bentuk chunkwise Gated DeltaNet harus sama dengan bentuk rekuren (KONSEP §11, risiko kernel TPU)."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from slm.gdn import _unit_lower_inverse_product, gated_delta_chunkwise, gated_delta_recurrent
from slm.layers import l2_norm

jax.config.update("jax_enable_x64", False)


def _inputs(key, b=2, t=100, h=3, d=16):
    ks = jax.random.split(key, 5)
    q = l2_norm(jax.random.normal(ks[0], (b, t, h, d))) * d ** -0.5
    k = l2_norm(jax.random.normal(ks[1], (b, t, h, d)))
    v = jax.random.normal(ks[2], (b, t, h, d))
    g = -jax.nn.softplus(jax.random.normal(ks[3], (b, t, h)))          # log-peluruhan ≤ 0
    beta = jax.nn.sigmoid(jax.random.normal(ks[4], (b, t, h)))
    return q, k, v, g, beta


@pytest.mark.parametrize("solver", ["solve", "product"])
@pytest.mark.parametrize("t,chunk", [(64, 16), (100, 16), (37, 64), (128, 32)])
def test_chunkwise_matches_recurrent(solver, t, chunk):
    q, k, v, g, beta = _inputs(jax.random.PRNGKey(t + chunk), t=t)
    o_ref, s_ref = gated_delta_recurrent(q, k, v, g, beta)
    o, s = gated_delta_chunkwise(q, k, v, g, beta, chunk=chunk, solver=solver)
    np.testing.assert_allclose(np.asarray(o), np.asarray(o_ref), atol=2e-4, rtol=2e-4)
    if t % chunk == 0:  # dengan padding, state akhir tetap sama karena β=0, g=0 di padding
        np.testing.assert_allclose(np.asarray(s), np.asarray(s_ref), atol=2e-4, rtol=2e-4)


def test_initial_state_carries_over():
    q, k, v, g, beta = _inputs(jax.random.PRNGKey(7), t=96)
    o_full, s_full = gated_delta_recurrent(q, k, v, g, beta)
    o1, s1 = gated_delta_chunkwise(q[:, :48], k[:, :48], v[:, :48], g[:, :48], beta[:, :48], chunk=16)
    o2, s2 = gated_delta_chunkwise(q[:, 48:], k[:, 48:], v[:, 48:], g[:, 48:], beta[:, 48:], chunk=16, initial_state=s1)
    np.testing.assert_allclose(np.asarray(jnp.concatenate([o1, o2], 1)), np.asarray(o_full), atol=2e-4)
    np.testing.assert_allclose(np.asarray(s2), np.asarray(s_full), atol=2e-4)


def test_gradients_match():
    q, k, v, g, beta = _inputs(jax.random.PRNGKey(3), t=48)
    f_ref = lambda *a: jnp.sum(gated_delta_recurrent(*a)[0] ** 2)
    f_chk = lambda *a: jnp.sum(gated_delta_chunkwise(*a, chunk=16)[0] ** 2)
    g_ref = jax.grad(f_ref, argnums=(0, 1, 2, 3, 4))(q, k, v, g, beta)
    g_chk = jax.grad(f_chk, argnums=(0, 1, 2, 3, 4))(q, k, v, g, beta)
    for a, b in zip(g_chk, g_ref):
        assert np.all(np.isfinite(np.asarray(a)))
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=5e-4, rtol=5e-3)


def test_unit_lower_inverse_product():
    c = 64
    a = jnp.tril(jax.random.normal(jax.random.PRNGKey(0), (c, c)) * 0.1, k=-1)
    inv = _unit_lower_inverse_product(a)
    np.testing.assert_allclose(np.asarray(inv @ (jnp.eye(c) + a)), np.eye(c), atol=1e-4)
