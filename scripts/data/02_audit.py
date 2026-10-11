"""Langkah 2 · Audit kualitas sampel mentah FineWeb-2 ind_Latn (cek manusia sebelum ambang dibekukan).

Mengambil N row group acak (seed berbeda dari sampling korpus), menghitung statistik filter,
dan menulis laporan markdown berisi contoh dokumen yang dibuang/dipertahankan.

    python scripts/data/02_audit.py --row-groups 2 --out notes/audit-fineweb2.md
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from slm.dataprep import _WORD, gambling_hits, gambling_word_rate, quality_reason  # noqa: E402

REPO = "HuggingFaceFW/fineweb-2"
REV = "af9c13333eb981300149d5ca60a8e9d659b276b9"


def snippet(t: str, n: int = 280) -> str:
    return t[:n].replace("\n", " ⏎ ").replace("|", "/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--row-groups", type=int, default=2)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="notes/audit-fineweb2.md")
    a = ap.parse_args()
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()
    files = sorted(fs.glob(f"datasets/{REPO}@{REV}/data/ind_Latn/train/*.parquet"))
    rng = np.random.default_rng(a.seed)
    rows = []
    picks = []
    for _ in range(a.row_groups):
        f = files[rng.integers(len(files))]
        with fs.open(f, "rb") as fh:
            pf = pq.ParquetFile(fh)
            g = int(rng.integers(pf.metadata.num_row_groups))
            rows += pf.read_row_group(g, columns=["text", "url", "dump", "language_score"]).to_pylist()
        picks.append(f"{os.path.basename(f)}#rg{g}")
    reasons = Counter()
    gamble_any = 0
    lengths = []
    removed, kept, borderline = [], [], []
    for r in rows:
        t = r["text"]
        lengths.append(len(t))
        reason = quality_reason(t)
        reasons[reason or "lolos"] += 1
        n, dens = gambling_hits(t)
        nw, _ = gambling_word_rate(_WORD.findall(t.lower()))
        n, dens = n + nw, dens
        gamble_any += n > 0
        if reason:
            removed.append((reason, n, r))
        elif n > 0:
            borderline.append((n, dens, r))
        else:
            kept.append(r)
    lengths = np.array(lengths)
    years = Counter(r["dump"][8:12] for r in rows)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("# Audit sampel FineWeb-2 ind_Latn\n\n")
        f.write(f"Revisi `{REV}` · row group: {', '.join(picks)} · {len(rows)} dokumen · seed {a.seed}.\n")
        f.write("Hasil audit ini adalah pengamatan pada sampel kecil, bukan statistik seluruh korpus.\n\n")
        f.write("## Statistik\n\n| Ukuran | Nilai |\n|---|---|\n")
        for k, v in reasons.most_common():
            f.write(f"| {k} | {v} ({100 * v / len(rows):.2f}%) |\n")
        f.write(f"| memuat ≥1 istilah judi | {gamble_any} ({100 * gamble_any / len(rows):.2f}%) |\n")
        f.write(f"| panjang karakter median / p10 / p90 | {np.median(lengths):.0f} / {np.percentile(lengths, 10):.0f} / {np.percentile(lengths, 90):.0f} |\n")
        f.write(f"| sebaran tahun dump | {dict(sorted(years.items()))} |\n\n")
        f.write("## Dibuang (maks. 15)\n\n| alasan | istilah judi | url | cuplikan |\n|---|---|---|---|\n")
        for reason, n, r in removed[:15]:
            f.write(f"| {reason} | {n} | {r['url'][:60]} | {snippet(r['text'])} |\n")
        f.write("\n## Lolos tetapi memuat istilah judi (1 frasa/kata judi atau lebih; cek false negative, maks. 15)\n\n| hit | densitas | url | cuplikan |\n|---|---|---|---|\n")
        for n, dens, r in borderline[:15]:
            f.write(f"| {n} | {dens:.4f} | {r['url'][:60]} | {snippet(r['text'])} |\n")
        f.write("\n## Contoh lolos acak (maks. 10)\n\n| url | cuplikan |\n|---|---|\n")
        for i in rng.choice(len(kept), size=min(10, len(kept)), replace=False):
            f.write(f"| {kept[i]['url'][:60]} | {snippet(kept[i]['text'])} |\n")
    print(f"laporan ditulis ke {a.out}: {dict(reasons)}")


if __name__ == "__main__":
    main()
