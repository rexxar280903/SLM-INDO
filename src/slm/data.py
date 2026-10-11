"""Pembacaan data training/evaluasi.

Format direktori data (hasil scripts/data/04_tokenize.py):
    train.bin, val.bin, test.bin   token uint16 berurutan; dokumen dipisah token EOS
    token_bytes.npy                (V,) jumlah byte UTF-8 tiap ID token (EOS dihitung 1)
    meta.json                      jumlah token/byte/dokumen, versi tokenizer, sumber data

Urutan batch (D33): jendela tak-tumpang-tindih sepanjang T+1 token (langkah T),
diacak per epoch dengan seed. Batch ke-i hanya bergantung pada (seed, i) sehingga
(1) semua arsitektur dengan seed yang sama melihat urutan data identik, dan
(2) resume cukup menyimpan nomor langkah.
"""
from __future__ import annotations

import json
import os

import numpy as np


class TokenData:
    def __init__(self, data_dir: str, split: str, seq_len: int):
        path = os.path.join(data_dir, f"{split}.bin")
        self.tokens = np.memmap(path, dtype=np.uint16, mode="r")
        self.seq_len = seq_len
        self.n_windows = (len(self.tokens) - 1) // seq_len
        meta_path = os.path.join(data_dir, "meta.json")
        self.meta = json.load(open(meta_path, encoding="utf-8")) if os.path.exists(meta_path) else {}
        if self.n_windows <= 0:
            raise ValueError(f"{path}: token terlalu sedikit untuk seq_len={seq_len}")

    def window(self, idx: int) -> np.ndarray:
        s = idx * self.seq_len
        return np.asarray(self.tokens[s:s + self.seq_len + 1], dtype=np.int32)


class BatchSchedule:
    """Batch training deterministik: batch(i) → array (B, T+1) int32."""

    def __init__(self, data: TokenData, batch_size: int, seed: int):
        self.data, self.batch_size, self.seed = data, batch_size, seed
        self._perm_epoch, self._perm = -1, None
        self.windows_per_epoch = (data.n_windows // batch_size) * batch_size

    def _permutation(self, epoch: int) -> np.ndarray:
        if epoch != self._perm_epoch:
            rng = np.random.default_rng([self.seed, epoch])
            self._perm = rng.permutation(self.data.n_windows)
            self._perm_epoch = epoch
        return self._perm

    def epoch_of(self, step: int) -> int:
        return (step * self.batch_size) // self.windows_per_epoch

    def batch(self, step: int) -> np.ndarray:
        start = step * self.batch_size
        epoch, offset = divmod(start, self.windows_per_epoch)
        idx = self._permutation(epoch)[offset:offset + self.batch_size]
        return np.stack([self.data.window(int(i)) for i in idx])


def eval_batches(data: TokenData, batch_size: int, max_tokens: int):
    """Jendela evaluasi tetap (urutan asli, tanpa acak): setiap token target dinilai sekali."""
    n = min(data.n_windows, max(1, max_tokens // data.seq_len))
    n = (n // batch_size) * batch_size or min(batch_size, data.n_windows)
    for s in range(0, n, batch_size):
        yield np.stack([data.window(i) for i in range(s, min(s + batch_size, n))])


def load_token_bytes(data_dir: str) -> np.ndarray | None:
    path = os.path.join(data_dir, "token_bytes.npy")
    return np.load(path) if os.path.exists(path) else None


class SyntheticData:
    """Token acak seragam untuk smoke test dan benchmark throughput (tanpa korpus)."""

    def __init__(self, vocab_size: int, seq_len: int, batch_size: int, seed: int):
        self.vocab_size, self.seq_len, self.batch_size, self.seed = vocab_size, seq_len, batch_size, seed

    def batch(self, step: int) -> np.ndarray:
        rng = np.random.default_rng([self.seed, step])
        return rng.integers(0, self.vocab_size, (self.batch_size, self.seq_len + 1), dtype=np.int32)

    def epoch_of(self, step: int) -> int:
        return 0
