"""Checkpoint sederhana: semua array disimpan ke satu .npz (kunci = path pytree) + meta JSON.

Isi checkpoint (paper Tabel 6): bobot, state optimizer (momen AdamW + count), langkah,
FLOPs kumulatif, fase N2 (jenis layer saat ini), seed, dan konfigurasi. Posisi data
cukup = nomor langkah karena urutan batch deterministik (data.py).
Penulisan atomik (tulis ke .tmp lalu rename) agar sesi Kaggle yang terputus tidak
meninggalkan checkpoint rusak.
"""
from __future__ import annotations

import glob
import json
import os

import jax
import numpy as np


def _key(path) -> str:
    out = []
    for p in path:
        for attr in ("key", "idx", "name"):
            if hasattr(p, attr):
                out.append(str(getattr(p, attr)))
                break
        else:
            out.append(str(p))
    return "|".join(out)


def save(ckpt_dir: str, step: int, params, opt_state, meta: dict, keep: int = 2) -> str:
    os.makedirs(ckpt_dir, exist_ok=True)
    arrays = {}
    for prefix, tree in (("params", params), ("opt", opt_state)):
        leaves, _ = jax.tree_util.tree_flatten_with_path(jax.device_get(tree))
        for path, leaf in leaves:
            arrays[f"{prefix}|{_key(path)}"] = np.asarray(leaf)
    base = os.path.join(ckpt_dir, f"step_{step:08d}")
    tmp = base + ".tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, base + ".npz")
    with open(base + ".json.tmp", "w", encoding="utf-8") as f:
        json.dump({**meta, "step": step}, f, indent=2, ensure_ascii=False)
    os.replace(base + ".json.tmp", base + ".json")
    for old in sorted(glob.glob(os.path.join(ckpt_dir, "step_*.npz")))[:-keep]:
        os.remove(old)
        if os.path.exists(old[:-4] + ".json"):
            os.remove(old[:-4] + ".json")
    return base


def latest(ckpt_dir: str) -> str | None:
    files = sorted(glob.glob(os.path.join(ckpt_dir, "step_*.npz")))
    return files[-1][:-4] if files else None


def read_meta(base: str) -> dict:
    with open(base + ".json", encoding="utf-8") as f:
        return json.load(f)


def restore(base: str, params_template, opt_template):
    """Mengisi pytree template (bentuk sama) dengan array dari checkpoint."""
    data = np.load(base + ".npz")

    def fill(prefix, tree):
        def get(path, leaf):
            arr = data[f"{prefix}|{_key(path)}"]
            if arr.shape != np.shape(leaf):
                raise ValueError(f"bentuk tidak cocok untuk {_key(path)}: {arr.shape} vs {np.shape(leaf)}")
            return arr
        return jax.tree_util.tree_map_with_path(get, tree)

    return fill("params", params_template), fill("opt", opt_template), read_meta(base)
