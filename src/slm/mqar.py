"""Generator MQAR (Multi-Query Associative Recall), mengikuti protokol Zoology (Arora dkk., 2023).

Satu contoh:
    [k1 v1 k2 v2 … kN vN] [0 0 k3 v3 0 0 0 k1 v1 0 …]
     ^ konteks pasangan     ^ kueri: key muncul lagi, model harus menebak value-nya
Key diambil dari paruh bawah vocab, value dari paruh atas; token 0 = pengisi.
Label hanya ada di posisi kueri (label = value), posisi lain −1 (diabaikan loss).
Jarak kueri dari konteks diambil dengan distribusi pangkat (power law, α = 0,01 di Zoology).

Akurasi = proporsi posisi kueri yang argmax-nya tepat. Probe dilatih terpisah dari LM
(paper §4.3); skornya adalah kemampuan arsitektur, bukan kemampuan checkpoint Indonesia.
"""
from __future__ import annotations

import numpy as np

MQAR_VOCAB = 8_192


def make_mqar_batch(rng: np.random.Generator, batch: int, seq_len: int, num_kv: int,
                    vocab_size: int = MQAR_VOCAB, power_a: float = 0.01):
    assert seq_len >= 4 * num_kv, "seq_len harus ≥ 4 × jumlah pasangan"
    half = vocab_size // 2
    ctx = 2 * num_kv
    inputs = np.zeros((batch, seq_len), np.int32)
    labels = np.full((batch, seq_len), -1, np.int32)
    # Peluang posisi kueri ∝ pangkat jarak (lebih sering dekat konteks).
    n_slots = (seq_len - ctx) // 2
    p = power_a * np.arange(1, n_slots + 1, dtype=np.float64) ** (power_a - 1)
    p = p / p.sum()
    for b in range(batch):
        keys = rng.choice(np.arange(1, half), size=num_kv, replace=False)
        vals = rng.choice(np.arange(half, vocab_size), size=num_kv, replace=True)
        inputs[b, 0:ctx:2] = keys
        inputs[b, 1:ctx:2] = vals
        slots = rng.choice(n_slots, size=num_kv, replace=False, p=p)
        for key, val, slot in zip(keys, vals, slots):
            pos = ctx + 2 * slot
            inputs[b, pos] = key
            inputs[b, pos + 1] = val
            labels[b, pos] = val       # di posisi key, target berikutnya adalah value
    return inputs, labels


# Grid evaluasi (paper §4.3): panjang 128–1.024 dengan beban pasangan meningkat.
EVAL_GRID = [(128, 8), (256, 16), (512, 32), (512, 64), (1024, 64), (1024, 128)]
TRAIN_MIX = [(128, 8), (256, 16), (512, 32), (512, 64), (1024, 64), (1024, 128)]
