"""Langkah 0 · Bangun set evaluasi + indeks n-gram untuk dekontaminasi (jalankan SEBELUM sampling korpus).

Keluaran di --out (default ./slm-data/eval):
    flores_dev.jsonl, flores_devtest.jsonl     FLORES-200 ind_Latn (devtest = evaluasi final)
    smsa_{train,valid,test}.jsonl               teks IndoNLU SmSA (label disertakan bila tersedia)
    wiki_eval.jsonl                             1.000 artikel Wikipedia Indonesia (bpb pelengkap)
    fineweb2_test.jsonl                         split test resmi FineWeb-2 ind_Latn (bpb web held-out)
    decontam_index.npz                          n-gram 13 kata (+ item pendek 8–12 kata)
    manifest.json                               sumber, revisi, jumlah item

    python scripts/data/00_eval_sets.py --out /kaggle/working/slm-data/eval
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
import tarfile
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from slm.dataprep import NgramIndex  # noqa: E402

FINEWEB2_REPO = "HuggingFaceFW/fineweb-2"
FINEWEB2_REV = "af9c13333eb981300149d5ca60a8e9d659b276b9"   # dipatok 2026-10-11
FLORES_URL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"
SMSA_BASE = "https://raw.githubusercontent.com/IndoNLP/indonlu/master/dataset/smsa_doc-sentiment-prosa/"
SMSA_FILES = {"train": "train_preprocess.tsv", "valid": "valid_preprocess.tsv", "test": "test_preprocess.tsv"}
WIKI_REPO, WIKI_CONFIG = "wikimedia/wikipedia", "20231101.id"


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read()


def flores(out: str) -> dict:
    try:
        tar = tarfile.open(fileobj=io.BytesIO(fetch(FLORES_URL)), mode="r:gz")
        res = {}
        for split in ("dev", "devtest"):
            member = next(m for m in tar.getmembers() if m.name.endswith(f"{split}/ind_Latn.{split}"))
            lines = tar.extractfile(member).read().decode("utf-8").splitlines()
            res[split] = write_jsonl(os.path.join(out, f"flores_{split}.jsonl"),
                                     [{"id": f"flores-{split}-{i}", "text": t} for i, t in enumerate(lines)])
        return {"source": FLORES_URL, **res}
    except Exception as e:  # sumber cadangan: openlanguagedata/flores_plus (gated, perlu login HF)
        print(f"[flores] gagal mengunduh {FLORES_URL}: {e}. Coba openlanguagedata/flores_plus (ind_Latn).")
        from datasets import load_dataset
        res = {}
        for split in ("dev", "devtest"):
            ds = load_dataset("openlanguagedata/flores_plus", "ind_Latn", split=split)
            res[split] = write_jsonl(os.path.join(out, f"flores_{split}.jsonl"),
                                     [{"id": f"flores-{split}-{i}", "text": r["text"]} for i, r in enumerate(ds)])
        return {"source": "openlanguagedata/flores_plus", **res}


def smsa(out: str) -> dict:
    res = {}
    for split, name in SMSA_FILES.items():
        try:
            raw = fetch(SMSA_BASE + name).decode("utf-8")
        except Exception:
            if split != "test":
                raise
            name = "test_preprocess_masked_label.tsv"     # label test publik mungkin disamarkan
            raw = fetch(SMSA_BASE + name).decode("utf-8")
        rows = [{"id": f"smsa-{split}-{i}", "text": r[0], "label": r[1] if len(r) > 1 else None}
                for i, r in enumerate(csv.reader(io.StringIO(raw), delimiter="\t")) if r]
        res[split] = {"file": name, "n": write_jsonl(os.path.join(out, f"smsa_{split}.jsonl"), rows)}
    return {"source": SMSA_BASE, **res}


def wikipedia(out: str, n: int, seed: int) -> dict:
    import numpy as np
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()
    files = sorted(fs.glob(f"datasets/{WIKI_REPO}/{WIKI_CONFIG}/*.parquet"))
    rng = np.random.default_rng(seed)
    with fs.open(files[0], "rb") as f:
        pf = pq.ParquetFile(f)
        groups = rng.choice(pf.metadata.num_row_groups, size=min(4, pf.metadata.num_row_groups), replace=False)
        table = pf.read_row_groups(sorted(groups.tolist()), columns=["id", "title", "text"])
    texts = table["text"].to_pylist()
    eligible = [i for i, t in enumerate(texts) if len(t) >= 500]      # artikel sangat pendek dilewati
    idx = rng.choice(eligible, size=min(n, len(eligible)), replace=False)
    rows = [{"id": f"wiki-{table['id'][int(i)].as_py()}", "title": table["title"][int(i)].as_py(),
             "text": texts[int(i)]} for i in sorted(idx)]
    return {"source": f"{WIKI_REPO}/{WIKI_CONFIG}", "file": files[0], "n": write_jsonl(os.path.join(out, "wiki_eval.jsonl"), rows)}


def fineweb2_test(out: str) -> dict:
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    path = f"datasets/{FINEWEB2_REPO}@{FINEWEB2_REV}/data/ind_Latn/test/000_00000.parquet"
    with HfFileSystem().open(path, "rb") as f:
        t = pq.read_table(f, columns=["id", "text", "url", "dump"])
    rows = [{"id": i, "text": x, "url": u, "dump": d} for i, x, u, d in
            zip(t["id"].to_pylist(), t["text"].to_pylist(), t["url"].to_pylist(), t["dump"].to_pylist())]
    return {"source": path, "n": write_jsonl(os.path.join(out, "fineweb2_test.jsonl"), rows)}


def read_texts(out: str, names) -> list[str]:
    texts = []
    for name in names:
        p = os.path.join(out, name)
        if os.path.exists(p):
            texts += [json.loads(l)["text"] for l in open(p, encoding="utf-8")]
    return texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="slm-data/eval")
    ap.add_argument("--wiki-n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip", nargs="*", default=[], help="lewati sumber: flores smsa wiki fineweb2_test")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    manifest = {"fineweb2_revision": FINEWEB2_REV}
    for name, fn in (("flores", lambda: flores(a.out)), ("smsa", lambda: smsa(a.out)),
                     ("wiki", lambda: wikipedia(a.out, a.wiki_n, a.seed)), ("fineweb2_test", lambda: fineweb2_test(a.out))):
        if name in a.skip:
            continue
        manifest[name] = fn()
        print(name, manifest[name])
    # Benchmark: satu n-gram cocok sudah cukup untuk membuang dokumen.
    bench = read_texts(a.out, ["flores_dev.jsonl", "flores_devtest.jsonl", "smsa_train.jsonl", "smsa_valid.jsonl",
                                "smsa_test.jsonl", "wiki_eval.jsonl"])
    index = NgramIndex.build(bench)
    index.save(os.path.join(a.out, "decontam_bench.npz"))
    # Test web: dipakai sebagai kriteria near-duplicate (proporsi overlap), bukan satu n-gram.
    web = NgramIndex.build(read_texts(a.out, ["fineweb2_test.jsonl"]))
    web.save(os.path.join(a.out, "decontam_webtest.npz"))
    manifest["decontam_ngrams"] = {"bench": {str(k): int(len(v)) for k, v in index.sets.items()},
                                   "webtest": {str(k): int(len(v)) for k, v in web.sets.items()}}
    json.dump(manifest, open(os.path.join(a.out, "manifest.json"), "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("indeks dekontaminasi:", manifest["decontam_ngrams"])


if __name__ == "__main__":
    main()
