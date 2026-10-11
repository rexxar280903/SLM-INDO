"""SLM-INDO: kode riset SLM efisien bahasa Indonesia (JAX murni + Optax).

Modul utama:
- config   : skala S10–S125, parser arsitektur (B0–B4 + grid), hyperparameter
- layers   : RMSNorm, RoPE, SwiGLU, attention (MHA/GQA/MLA/CLA/sliding window)
- gdn      : Gated DeltaNet (rekuren acuan + chunkwise untuk training)
- model    : inisialisasi parameter dan forward pass model hybrid
- flops    : hitung parameter, FLOPs per token, KV-cache, dan state
- data     : pembaca shard uint16 + urutan batch deterministik + bpb
- mqar     : generator tugas sintetis MQAR (probe recall)
- n2       : konversi linear-first warm-start (GDN -> attention)
- checkpoint, train, evaluate
"""

__version__ = "0.1.0"
