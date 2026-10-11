"""Gated DeltaNet (Yang, Kautz & Hatamizadeh, ICLR 2025) dalam JAX murni.

Aturan update per head (state S berukuran dk × dv, output baris o_t = q_tᵀ S_t):

    S_t = α_t (I − β_t k_t k_tᵀ) S_{t−1} + β_t k_t v_tᵀ          α_t = exp(g_t), g_t ≤ 0
    o_t = S_tᵀ q_t

Bentuk ini setara dengan persamaan paper (yang memakai S transpos). Ditulis ulang:

    u_t = β_t (v_t − α_t S_{t−1}ᵀ k_t)      ("galat" yang ditulis ke memori)
    S_t = α_t S_{t−1} + k_t u_tᵀ

Dua implementasi:
1. `gated_delta_recurrent` : loop token demi token (lax.scan). Lambat, tetapi jelas
   benar → dipakai sebagai ACUAN di unit test dan untuk decode satu token.
2. `gated_delta_chunkwise` : token dibagi menjadi chunk berukuran C. Di dalam chunk
   semuanya perkalian matriks (cepat di TPU); antarchunk state diteruskan dengan
   lax.scan sepanjang T/C langkah saja.

Derivasi chunkwise (posisi r = 1..C dalam satu chunk, S_0 = state awal chunk):
    γ_r = Σ_{j≤r} g_j,  Γ_r = exp(γ_r)
    U = (I + A)⁻¹ [ diag(β) V − diag(βΓ) K S_0 ],   A_rj = β_r (Γ_r/Γ_j)(k_r·k_j) untuk j<r
      = W − K_d S_0  dengan W = (I+A)⁻¹ diag(β) V,  K_d = (I+A)⁻¹ diag(βΓ) K
    O = diag(Γ) Q S_0 + [(Q Kᵀ) ⊙ D] U,             D_rj = Γ_r/Γ_j untuk j≤r
    S_C = Γ_C S_0 + (diag(Γ_C/Γ) K)ᵀ U
Semua faktor peluruhan berbentuk exp(selisih γ) ≤ 1 sehingga stabil secara numerik.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from .config import ModelConfig
from .layers import causal_conv, l2_norm, rms_norm


def gated_delta_recurrent(q, k, v, g, beta, initial_state=None):
    """Acuan rekuren. q,k: (B,T,H,dk)  v: (B,T,H,dv)  g,beta: (B,T,H). Semua fp32.

    Mengembalikan (o: (B,T,H,dv), state akhir: (B,H,dk,dv)).
    """
    b, t, h, dk = k.shape
    dv = v.shape[-1]
    s0 = jnp.zeros((b, h, dk, dv), jnp.float32) if initial_state is None else initial_state

    def step(s, inp):
        q_t, k_t, v_t, g_t, b_t = inp                       # (B,H,dk) … (B,H)
        s = s * jnp.exp(g_t)[..., None, None]               # peluruhan α_t
        pred = jnp.einsum("bhk,bhkv->bhv", k_t, s)          # Sᵀk: nilai yang "diingat" untuk k_t
        u = b_t[..., None] * (v_t - pred)                    # aturan delta
        s = s + jnp.einsum("bhk,bhv->bhkv", k_t, u)
        o_t = jnp.einsum("bhk,bhkv->bhv", q_t, s)
        return s, o_t

    xs = tuple(jnp.moveaxis(a, 1, 0) for a in (q, k, v, g, beta))
    s_final, o = jax.lax.scan(step, s0, xs)
    return jnp.moveaxis(o, 0, 1), s_final


def _unit_lower_inverse_product(a):
    """(I + A)⁻¹ untuk A segitiga bawah murni (nilpoten) lewat perkalian matriks saja.

    1/(1+x) = (1−x)(1+x²)(1+x⁴)… berlaku tepat karena A^C = 0. Butuh ±2·log2(C)
    perkalian C×C. Alternatif dari triangular solve bila kernel itu lambat di TPU.
    """
    c = a.shape[-1]
    eye = jnp.eye(c, dtype=a.dtype)
    inv = eye - a
    power = a
    n = 1
    while n * 2 < c:
        power = power @ power                                 # A^(2^m)
        inv = inv @ (eye + power)
        n *= 2
    return inv


def gated_delta_chunkwise(q, k, v, g, beta, chunk: int = 64, initial_state=None, solver: str = "solve"):
    """Bentuk chunkwise; hasilnya harus sama dengan `gated_delta_recurrent` (diuji)."""
    b, t, h, dk = k.shape
    dv = v.shape[-1]
    pad = (-t) % chunk
    if pad:
        # Padding di akhir: g=0 (tanpa peluruhan), β=0 (tanpa update), q=0 → output dibuang.
        padf = lambda a: jnp.pad(a, ((0, 0), (0, pad)) + ((0, 0),) * (a.ndim - 2))
        q, k, v, g, beta = map(padf, (q, k, v, g, beta))
    n = (t + pad) // chunk

    def to_chunks(a):  # (B,T,H,…) → (B,H,N,C,…)
        a = a.reshape((b, n, chunk, h) + a.shape[3:])
        return jnp.moveaxis(a, 3, 1)

    qc, kc, vc = to_chunks(q), to_chunks(k), to_chunks(v)
    gc, bc = to_chunks(g), to_chunks(beta)                    # (B,H,N,C)
    gam = jnp.cumsum(gc, axis=-1)                             # γ_r
    idx = jnp.arange(chunk)
    incl = idx[:, None] >= idx[None, :]                       # j ≤ r
    strict = idx[:, None] > idx[None, :]                      # j < r
    diff = gam[..., :, None] - gam[..., None, :]              # γ_r − γ_j
    decay = jnp.exp(jnp.where(incl, diff, -jnp.inf))          # D_rj (0 di atas diagonal)

    kk = jnp.einsum("bhnrd,bhnjd->bhnrj", kc, kc)
    a = jnp.where(strict, bc[..., :, None] * kk * decay, 0.0)
    rhs = jnp.concatenate([vc * bc[..., None], kc * (bc * jnp.exp(gam))[..., None]], axis=-1)
    if solver == "product":
        sol = _unit_lower_inverse_product(a) @ rhs
    else:
        eye = jnp.eye(chunk, dtype=a.dtype)
        mat = jnp.broadcast_to(eye, a.shape) + a
        sol = jax.lax.linalg.triangular_solve(mat, rhs, left_side=True, lower=True, unit_diagonal=True)
    w, kd = sol[..., :dv], sol[..., dv:]                      # W dan K_d
    qk = jnp.einsum("bhnrd,bhnjd->bhnrj", qc, kc) * decay      # (QKᵀ) ⊙ D
    q_dec = qc * jnp.exp(gam)[..., None]                       # diag(Γ) Q
    g_last = gam[..., -1]                                      # γ_C
    k_dec = kc * jnp.exp(g_last[..., None] - gam)[..., None]   # diag(Γ_C/Γ) K

    s0 = jnp.zeros((b, h, dk, dv), jnp.float32) if initial_state is None else initial_state

    def step(s, inp):
        w_i, kd_i, q_i, qk_i, kdec_i, gl_i = inp
        u = w_i - kd_i @ s                                     # (B,H,C,dv)
        o = q_i @ s + qk_i @ u
        s = s * jnp.exp(gl_i)[..., None, None] + jnp.swapaxes(kdec_i, -1, -2) @ u
        return s, o

    xs = tuple(jnp.moveaxis(x_, 2, 0) for x_ in (w, kd, q_dec, qk, k_dec, g_last))
    s_final, o = jax.lax.scan(step, s0, xs)                    # o: (N,B,H,C,dv)
    o = jnp.moveaxis(o, 0, 2).reshape(b, h, n * chunk, dv)
    o = jnp.moveaxis(o, 1, 2)[:, :t]
    return o, s_final


def gdn_gates(p, x):
    """Gerbang α (lewat log-peluruhan g) dan β, dihitung di fp32.

    g = −exp(A_log) · softplus(x·W_a + dt_bias)  ≤ 0   (parameterisasi gaya Mamba-2)
    β = sigmoid(x·W_b) ∈ (0,1)
    """
    a = (x @ p["wa"]).astype(jnp.float32) + p["dt_bias"].astype(jnp.float32)
    g = -jnp.exp(p["A_log"].astype(jnp.float32)) * jax.nn.softplus(a)
    beta = jax.nn.sigmoid((x @ p["wb"]).astype(jnp.float32))
    return g, beta


def gdn_mixer(p, x, cfg: ModelConfig):
    """Mixer Gated DeltaNet lengkap (mengikuti desain referensi flash-linear-attention):

    x → proyeksi q,k,v → conv pendek kausal + SiLU → L2-norm q,k → gerbang g, β
      → gated delta rule (chunkwise) → RMSNorm per head ⊙ SiLU(x·W_g) → proyeksi O
    """
    b, t, _ = x.shape
    h, e = cfg.n_heads, cfg.head_dim
    q = jax.nn.silu(causal_conv(x @ p["wq"], p["conv_q"]))
    k = jax.nn.silu(causal_conv(x @ p["wk"], p["conv_k"]))
    v = jax.nn.silu(causal_conv(x @ p["wv"], p["conv_v"]))
    q = l2_norm(q.reshape(b, t, h, e)) * (e ** -0.5)
    k = l2_norm(k.reshape(b, t, h, e))
    v = v.reshape(b, t, h, e).astype(jnp.float32)
    g, beta = gdn_gates(p, x)
    # Di TPU, matmul fp32 dengan presisi "default" dihitung dengan pass bf16. State GDN
    # terakumulasi lintas chunk, jadi intinya dipaksa float32 (biaya diukur di pilot).
    with jax.default_matmul_precision(cfg.gdn_precision):
        o, _ = gated_delta_chunkwise(q, k, v, g, beta, cfg.gdn_chunk, solver=cfg.gdn_solver)
    o = rms_norm(o, p["o_norm"], cfg.norm_eps)                 # norm per head (dimensi dv)
    gate = jax.nn.silu((x @ p["wg"]).astype(jnp.float32)).reshape(b, t, h, e)
    o = (o * gate).astype(x.dtype).reshape(b, t, h * e)
    return o @ p["wo"]
