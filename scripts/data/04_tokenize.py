"""Langkah 4 · Tokenisasi korpus bersih → train.bin / val.bin / test.bin (uint16) + meta.json.

Setiap dokumen = encode(teks) + [EOS]. Token ditulis berurutan (packing tanpa padding).
test.bin berasal dari split test resmi FineWeb-2 (eval/fineweb2_test.jsonl).
Hasil siap diunggah sebagai Kaggle Dataset privat untuk notebook TPU.

    python scripts/data/04_tokenize.py --clean slm-data/clean --eval-dir slm-data/eval \
        --tok slm-data/tok16k --out slm-data/tokens16k --workers 4
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import shutil
import time
from multiprocessing import Pool

import numpy as np
import pyarrow.parquet as pq

_SP = None


def _init(model_path):
    global _SP
    import sentencepiece as spm
    _SP = spm.SentencePieceProcessor(model_file=model_path)


def _encode_file(path):
    texts = pq.read_table(path, columns=["text"])["text"].to_pylist()
    eos = _SP.eos_id()
    ids = _SP.encode(texts)
    out = np.concatenate([np.asarray(doc + [eos], dtype=np.uint16) for doc in ids]) if ids else np.empty(0, np.uint16)
    return out, len(texts), sum(len(t.encode("utf-8")) for t in texts)


def _encode_jsonl(path, model_path):
    _init(model_path)
    texts = [json.loads(l)["text"] for l in open(path, encoding="utf-8")]
    eos = _SP.eos_id()
    ids = _SP.encode(texts)
    out = np.concatenate([np.asarray(doc + [eos], dtype=np.uint16) for doc in ids]) if ids else np.empty(0, np.uint16)
    return out, len(texts), sum(len(t.encode("utf-8")) for t in texts)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", default="slm-data/clean")
    ap.add_argument("--eval-dir", default="slm-data/eval")
    ap.add_argument("--tok", default="slm-data/tok16k")
    ap.add_argument("--out", default="slm-data/tokens16k")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    model = os.path.join(a.tok, "spm.model")
    vocab = len(np.load(os.path.join(a.tok, "token_bytes.npy")))
    assert vocab <= 65536, "uint16 hanya cukup untuk vocab ≤ 65.536"
    meta = {"vocab_size": vocab, "tokenizer_sha256": sha256(model), "dtype": "uint16", "splits": {}}
    t0 = time.time()
    for split in ("train", "val"):
        files = sorted(glob.glob(os.path.join(a.clean, split, "*.parquet")))
        out_path = os.path.join(a.out, f"{split}.bin")
        n_tok = n_docs = n_bytes = 0
        with open(out_path, "wb") as f, Pool(a.workers, initializer=_init, initargs=(model,)) as pool:
            for arr, d, b in pool.imap(_encode_file, files):
                arr.tofile(f)
                n_tok, n_docs, n_bytes = n_tok + len(arr), n_docs + d, n_bytes + b
                print(f"{split}: {n_tok/1e6:.0f}M token · {time.time() - t0:.0f}s", flush=True)
        meta["splits"][split] = {"tokens": n_tok, "docs": n_docs, "bytes": n_bytes, "sha256": sha256(out_path)}
    test_src = os.path.join(a.eval_dir, "fineweb2_test.jsonl")
    if os.path.exists(test_src):
        arr, d, b = _encode_jsonl(test_src, model)
        arr.tofile(os.path.join(a.out, "test.bin"))
        meta["splits"]["test"] = {"tokens": len(arr), "docs": d, "bytes": b, "source": "FineWeb-2 test resmi"}
    shutil.copy(model, os.path.join(a.out, "spm.model"))
    shutil.copy(os.path.join(a.tok, "token_bytes.npy"), os.path.join(a.out, "token_bytes.npy"))
    for extra in ("tokenizer_stats.json",):
        if os.path.exists(os.path.join(a.tok, extra)):
            shutil.copy(os.path.join(a.tok, extra), os.path.join(a.out, extra))
    for extra in ("sample_report.json",):
        if os.path.exists(os.path.join(a.clean, extra)):
            meta[extra.replace(".json", "")] = json.load(open(os.path.join(a.clean, extra)))
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=2)
    print(json.dumps(meta["splits"], indent=2))


if __name__ == "__main__":
    main()
