"""Evaluasi akhir bits-per-byte dari checkpoint (paper Pers. 8).

    python -m slm.evaluate --run-dir runs/b0-s10-seed0-flops --data-dir /kaggle/input/slm-data \
        --eval-dir /kaggle/input/slm-eval --sets val test flores_devtest wiki_eval

- val / test       : token .bin dari 04_tokenize.py (test = split test resmi FineWeb-2)
- flores_devtest, flores_dev, wiki_eval : file .jsonl dari 00_eval_sets.py, ditokenisasi di sini
Setiap token target dinilai sekali (jendela tak tumpang tindih sepanjang konteks training).
bpb = Σ NLL / (ln 2 · Σ byte UTF-8 token target); EOS dihitung 1 byte (D31).
Set uji hanya dipakai di akhir — tidak untuk memilih konfigurasi atau hyperparameter.
"""
from __future__ import annotations

import argparse
import json
import math
import os

import jax
import jax.numpy as jnp
import numpy as np

from . import checkpoint as ckpt
from .config import SCALES, build_model_config
from .model import chunked_cross_entropy, forward_hidden, init_params


def windows_from_stream(tokens: np.ndarray, seq_len: int) -> np.ndarray:
    n = (len(tokens) - 1) // seq_len
    return np.stack([tokens[i * seq_len:i * seq_len + seq_len + 1] for i in range(n)]).astype(np.int32)


def tokenize_jsonl(path: str, sp) -> np.ndarray:
    texts = [json.loads(l)["text"] for l in open(path, encoding="utf-8")]
    eos = sp.eos_id()
    return np.concatenate([np.asarray(ids + [eos], np.int32) for ids in sp.encode(texts)])


def bpb(params, mcfg, windows: np.ndarray, token_bytes: np.ndarray, batch: int) -> dict:
    tb = jnp.asarray(token_bytes)

    @jax.jit
    def step(p, b):
        h, emb = forward_hidden(p, b[:, :-1], mcfg)
        total, count, _, _ = chunked_cross_entropy(h, emb, b[:, 1:])
        return total, count, jnp.sum(tb[b[:, 1:]].astype(jnp.float32))

    tot = cnt = nb = 0.0
    for s in range(0, len(windows), batch):
        a, b, c = step(params, windows[s:s + batch])   # batch terakhir boleh lebih kecil (kompilasi ulang sekali)
        tot, cnt, nb = tot + float(a), cnt + float(b), nb + float(c)
    return {"bpb": tot / math.log(2) / nb, "loss_per_token": tot / cnt, "tokens": cnt, "bytes": nb,
            "bytes_per_token": nb / cnt}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--eval-dir", default="")
    ap.add_argument("--sets", nargs="*", default=["val", "test", "flores_devtest"])
    ap.add_argument("--max-tokens", type=int, default=20_000_000)
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args(argv)
    cfg = json.load(open(os.path.join(a.run_dir, "config.json"), encoding="utf-8"))
    base = ckpt.latest(os.path.join(a.run_dir, "ckpt"))
    meta = ckpt.read_meta(base)
    m = cfg["model"]
    overrides = {k: m[k] for k in ("vocab_size", "attn_impl", "gdn_solver", "remat", "compute_dtype", "gdn_chunk")}
    arch = meta.get("arch_now", cfg["train"]["arch"])
    mcfg = build_model_config(arch, SCALES[cfg["train"]["scale"]], **overrides)
    params = init_params(jax.random.PRNGKey(0), mcfg)
    params = _restore_params(base, params)
    token_bytes = np.load(os.path.join(a.data_dir, "token_bytes.npy"))
    sp = None
    results = {"run": a.run_dir, "checkpoint": base, "arch": arch}
    for name in a.sets:
        if name in ("val", "test"):
            toks = np.memmap(os.path.join(a.data_dir, f"{name}.bin"), dtype=np.uint16, mode="r")
            toks = np.asarray(toks[:a.max_tokens + 1], np.int32)
        else:
            if sp is None:
                import sentencepiece as spm
                sp = spm.SentencePieceProcessor(model_file=os.path.join(a.data_dir, "spm.model"))
            toks = tokenize_jsonl(os.path.join(a.eval_dir, f"{name}.jsonl"), sp)[:a.max_tokens + 1]
        win = windows_from_stream(toks, mcfg.seq_len)
        results[name] = bpb(params, mcfg, win, token_bytes, a.batch)
        print(name, results[name])
    with open(os.path.join(a.run_dir, "eval_final.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return results


def _restore_params(base: str, template):
    data = np.load(base + ".npz")
    return jax.tree_util.tree_map_with_path(lambda p, _: data["params|" + ckpt._key(p)], template)


if __name__ == "__main__":
    main()
