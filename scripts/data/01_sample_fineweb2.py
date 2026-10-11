"""Langkah 1 · Sampel FineWeb-2 ind_Latn → korpus bersih (parquet) untuk tokenizer & tokenisasi.

Strategi sampling (D23): file parquet FineWeb-2 diurutkan per dump Common Crawl (2013 → 2024),
jadi mengambil file pertama saja akan bias ke tahun lama. Skrip ini memilih ROW GROUP acak
(1.000 dokumen per row group) dari SEMUA file train dengan seed tetap, lalu membaca hanya
row group itu lewat HTTP range request. Berhenti saat byte teks train ≥ --target-gb.

Row group diproses paralel (--workers); urutan hasil tetap mengikuti urutan acak (deterministik).
Setiap dokumen: filter kualitas (spam judi, terlalu pendek) → dedup exact (termasuk terhadap
test resmi) → dekontaminasi n-gram → split hash (99% train / 1% val) → ditulis ke shard parquet.

    python scripts/data/01_sample_fineweb2.py --eval-dir slm-data/eval --out slm-data/clean --target-gb 14
    python scripts/data/01_sample_fineweb2.py ... --max-row-groups 3      # uji cepat
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from multiprocessing import Pool

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from slm.dataprep import WEBTEST_OVERLAP, NgramIndex, doc_hash, quality_reason, split_of  # noqa: E402

REPO = "HuggingFaceFW/fineweb-2"
REV = "af9c13333eb981300149d5ca60a8e9d659b276b9"
COLUMNS = ["text", "id", "url", "dump", "language_score", "minhash_cluster_size"]


class ShardWriter:
    def __init__(self, out_dir: str, split: str, docs_per_shard: int = 100_000):
        self.dir = os.path.join(out_dir, split)
        os.makedirs(self.dir, exist_ok=True)
        self.rows, self.shard, self.limit = [], 0, docs_per_shard
        self.n_docs = self.n_bytes = 0

    def add(self, row: dict):
        self.rows.append(row)
        self.n_docs += 1
        self.n_bytes += row["n_bytes"]
        if len(self.rows) >= self.limit:
            self.flush()

    def flush(self):
        if not self.rows:
            return
        table = pa.Table.from_pylist(self.rows)
        pq.write_table(table, os.path.join(self.dir, f"part-{self.shard:05d}.parquet"), compression="zstd")
        self.rows, self.shard = [], self.shard + 1


def list_row_groups(fs, files):
    groups = []
    for f in files:
        with fs.open(f, "rb") as fh:
            n = pq.ParquetFile(fh).metadata.num_row_groups
        groups += [(f, g) for g in range(n)]
    return groups


# --- worker: baca satu row group, filter kualitas + dekontaminasi (dedup & split di proses utama) ---
_W = {}


def _init_worker(eval_dir: str):
    from huggingface_hub import HfFileSystem
    _W["fs"] = HfFileSystem()
    _W["bench"] = NgramIndex.load(os.path.join(eval_dir, "decontam_bench.npz"))
    _W["webtest"] = NgramIndex.load(os.path.join(eval_dir, "decontam_webtest.npz"))


def _process_group(task):
    f, g = task
    for attempt in range(5):                        # jaringan bisa putus: coba ulang
        try:
            # Satu handle per file per worker: metadata/footer dibaca sekali, lalu hanya
            # range request ke CDN (menghindari batas laju API Hugging Face anonim).
            pf = _W.setdefault("pf", {}).get(f)
            if pf is None:
                pf = pq.ParquetFile(_W["fs"].open(f, "rb", block_size=8 * 2**20))
                _W["pf"][f] = pf
            rows = pf.read_row_group(g, columns=COLUMNS).to_pylist()
            break
        except Exception:
            _W.get("pf", {}).pop(f, None)
            time.sleep(5 * (attempt + 1))
    else:
        return f, g, [], Counter({"row_group_gagal": 1})
    stats, keep = Counter(), []
    for row in rows:
        stats["dibaca"] += 1
        text = row["text"]
        reason = quality_reason(text)
        if reason:
            stats[reason] += 1
            continue
        if _W["bench"].matches(text):
            stats["kontaminasi_benchmark"] += 1
            continue
        if _W["webtest"].overlap_fraction(text) >= WEBTEST_OVERLAP:
            stats["near_duplicate_test_web"] += 1
            continue
        keep.append({"id": row["id"], "text": text, "url": row["url"], "dump": row["dump"],
                     "n_bytes": len(text.encode("utf-8")), "hash": doc_hash(text)})
    return f, g, keep, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default="slm-data/eval")
    ap.add_argument("--out", default="slm-data/clean")
    ap.add_argument("--target-gb", type=float, default=14.0, help="byte teks train yang dikumpulkan")
    ap.add_argument("--seed", type=int, default=20261011)
    ap.add_argument("--max-row-groups", type=int, default=None)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    a = ap.parse_args()

    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()
    files = sorted(fs.glob(f"datasets/{REPO}@{REV}/data/ind_Latn/train/*.parquet"))
    print(f"{len(files)} file train; membaca metadata row group …")
    groups = list_row_groups(fs, files)
    rng = np.random.default_rng(a.seed)
    order = rng.permutation(len(groups))
    if a.max_row_groups:
        order = order[:a.max_row_groups]
    print(f"{len(groups)} row group tersedia (±{len(groups) * 1000 / 1e6:.0f}M dokumen)")

    test_hashes = set()
    test_path = os.path.join(a.eval_dir, "fineweb2_test.jsonl")
    if os.path.exists(test_path):
        test_hashes = {doc_hash(json.loads(l)["text"]) for l in open(test_path, encoding="utf-8")}

    writers = {s: ShardWriter(a.out, s) for s in ("train", "val")}
    seen: set[str] = set()
    stats = Counter()
    years = Counter()
    used = []
    t0 = time.time()
    target = a.target_gb * 1e9
    with Pool(a.workers, initializer=_init_worker, initargs=(a.eval_dir,)) as pool:
        # imap menjaga urutan hasil = urutan row group acak → korpus deterministik untuk seed yang sama
        for k, (f, g, keep, st) in enumerate(pool.imap(_process_group, (groups[i] for i in order), chunksize=1)):
            if writers["train"].n_bytes >= target:
                pool.terminate()
                break
            used.append([os.path.basename(f), g])
            stats.update(st)
            for row in keep:
                h = row["hash"]
                if h in seen or h in test_hashes:
                    stats["duplikat_exact"] += 1
                    continue
                seen.add(h)
                split = split_of(h)
                writers[split].add(row)
                stats[f"lolos_{split}"] += 1
                years[row["dump"][8:12]] += 1
            if k % 50 == 0:
                print(f"row group {k}: train {writers['train'].n_bytes / 1e9:.2f} GB · {dict(stats)} · "
                      f"{time.time() - t0:.0f}s", flush=True)
    for w in writers.values():
        w.flush()
    report = {"repo": REPO, "revision": REV, "seed": a.seed, "row_groups_used": len(used), "stats": dict(stats),
              "splits": {s: {"docs": w.n_docs, "bytes": w.n_bytes, "shards": w.shard} for s, w in writers.items()},
              "dump_year_counts": dict(sorted(years.items())), "seconds": time.time() - t0}
    json.dump(report, open(os.path.join(a.out, "sample_report.json"), "w"), indent=2)
    json.dump(used, open(os.path.join(a.out, "row_groups_used.json"), "w"))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
