"""Langkah 3 · Latih tokenizer SentencePiece BPE dari data TRAIN bersih saja (D25).

Pengaturan (dibekukan sebelum eksperimen):
  model_type=bpe, vocab 16.384 (ablation 8.192 / 32.768), byte_fallback=True,
  normalization=identity (teks tidak diubah), add_dummy_prefix=False,
  remove_extra_whitespaces=False, split_digits=True, allow_whitespace_only_pieces=True,
  user_defined_symbols=["\n"], ID khusus: unk=0, bos=1, eos=2 (pad tidak dipakai).
Dengan identity + tanpa dummy prefix, decode(encode(teks)) == teks persis, sehingga
jumlah byte per token bisa dijumlahkan menjadi byte teks asli (syarat bpb yang tepat).

    python scripts/data/03_train_tokenizer.py --clean slm-data/clean --out slm-data/tok16k --vocab 16384
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))


def token_byte_table(sp) -> np.ndarray:
    """Jumlah byte UTF-8 yang direpresentasikan tiap ID token (EOS/BOS/UNK dihitung 1)."""
    out = np.zeros(sp.get_piece_size(), np.int32)
    for i in range(sp.get_piece_size()):
        piece = sp.id_to_piece(i)
        if sp.is_byte(i):
            out[i] = 1
        elif sp.is_control(i) or sp.is_unknown(i):
            out[i] = 1
        else:
            out[i] = len(piece.replace("▁", " ").encode("utf-8"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", default="slm-data/clean")
    ap.add_argument("--out", default="slm-data/tok16k")
    ap.add_argument("--vocab", type=int, default=16384)
    ap.add_argument("--sample-gb", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    a = ap.parse_args()
    import sentencepiece as spm

    os.makedirs(a.out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.clean, "train", "*.parquet")))
    rnd = random.Random(a.seed)
    rnd.shuffle(files)
    sample_path = os.path.join(a.out, "tokenizer_sample.txt")
    budget, written = a.sample_gb * 1e9, 0
    with open(sample_path, "w", encoding="utf-8") as f:
        for path in files:
            texts = pq.read_table(path, columns=["text"])["text"].to_pylist()
            rnd.shuffle(texts)
            for t in texts:
                # SentencePiece membaca per baris: dokumen dipecah per baris (baris kosong dilewati).
                for line in t.split("\n"):
                    if line.strip():
                        f.write(line + "\n")
                        written += len(line.encode("utf-8")) + 1
                if written >= budget:
                    break
            if written >= budget:
                break
    prefix = os.path.join(a.out, "spm")
    spm.SentencePieceTrainer.train(
        input=sample_path, model_prefix=prefix, vocab_size=a.vocab, model_type="bpe",
        character_coverage=0.9995, byte_fallback=True, normalization_rule_name="identity",
        add_dummy_prefix=False, remove_extra_whitespaces=False, split_digits=True,
        allow_whitespace_only_pieces=True, user_defined_symbols=["\n"], max_sentence_length=16384,
        unk_id=0, bos_id=1, eos_id=2, pad_id=-1, num_threads=a.threads, input_sentence_size=0,
        shuffle_input_sentence=True, train_extremely_large_corpus=False, seed_sentencepiece_size=1_000_000)
    sp = spm.SentencePieceProcessor(model_file=prefix + ".model")
    table = token_byte_table(sp)
    np.save(os.path.join(a.out, "token_bytes.npy"), table)

    # Uji round-trip & statistik pada dokumen validasi
    val_files = sorted(glob.glob(os.path.join(a.clean, "val", "*.parquet")))
    val = pq.read_table(val_files[0], columns=["text"])["text"].to_pylist()[:2000] if val_files else []
    exact = n_tok = n_bytes = n_words = 0
    for t in val:
        ids = sp.encode(t)
        exact += sp.decode(ids) == t
        n_tok += len(ids)
        n_bytes += len(t.encode("utf-8"))
        n_words += len(t.split())
        assert int(table[ids].sum()) == len(t.encode("utf-8")), "jumlah byte per token ≠ byte teks"
    stats = {"vocab": sp.get_piece_size(), "sample_bytes": written, "val_docs": len(val),
             "roundtrip_exact": exact, "bytes_per_token": n_bytes / max(1, n_tok),
             "fertility_tokens_per_word": n_tok / max(1, n_words), "eos_id": sp.eos_id()}
    json.dump(stats, open(os.path.join(a.out, "tokenizer_stats.json"), "w"), indent=2)
    os.remove(sample_path)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
