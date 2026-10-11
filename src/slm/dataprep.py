"""Fungsi pemrosesan korpus (dipakai scripts/data/*.py dan diuji di tests/).

Keputusan D23 (KONSEP §9): sumber pretraining = FineWeb-2 `ind_Latn` (revisi dipatok).
FineWeb-2 sudah menerapkan identifikasi bahasa (GlotLID), filter kualitas, dan
deduplikasi MinHash global per bahasa. Kita menambahkan:
  1. filter spam judi online (khas web Indonesia),
  2. filter dokumen sangat pendek,
  3. deduplikasi exact (hash teks ternormalisasi),
  4. dekontaminasi: dokumen dibuang bila memuat ≥1 n-gram benchmark (FLORES, SmSA, Wikipedia uji)
     atau bila ≥50% n-gram-nya ada di test web (near-duplicate; overlap template situs dibiarkan),
  5. split dokumen berbasis hash: 99% train, 1% val (test = split test resmi FineWeb-2).
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
import zlib

import numpy as np

# ---------------------------------------------------------------------------
# Normalisasi & hash
# ---------------------------------------------------------------------------
_WS = re.compile(r"\s+")
_WORD = re.compile(r"\w+", re.UNICODE)


def normalize_for_hash(text: str) -> str:
    return _WS.sub(" ", unicodedata.normalize("NFKC", text).lower()).strip()


def doc_hash(text: str) -> str:
    return hashlib.sha1(normalize_for_hash(text).encode("utf-8")).hexdigest()


def split_of(h: str, val_per_mille: int = 10) -> str:
    """Split deterministik dari hash teks: duplikat exact selalu jatuh di split yang sama."""
    return "val" if int(h[:8], 16) % 1000 < val_per_mille else "train"


# ---------------------------------------------------------------------------
# Filter kualitas
# ---------------------------------------------------------------------------
GAMBLING_TERMS = [
    "slot gacor", "gacor hari ini", "maxwin", "rtp live", "rtp slot", "situs slot", "slot online", "judi slot",
    "slot88", "slot 88", "link alternatif", "deposit pulsa", "depo pulsa", "togel online", "bandar togel",
    "situs togel", "agen judi", "judi online", "sbobet", "pragmatic play", "scatter hitam", "pola gacor",
    "bocoran slot", "mahjong ways", "gates of olympus", "sweet bonanza", "zeus slot", "slot demo",
    "bonus new member", "daftar slot", "agen slot", "bandar slot", "casino online", "poker online",
    "judi bola", "live casino", "toto macau", "togel hongkong", "togel singapore", "prediksi togel",
    "angka jitu", "slot thailand", "server thailand", "anti rungkad", "situs judi", "wd terbesar",
]
_GAMBLING = re.compile("|".join(re.escape(t) for t in GAMBLING_TERMS))
# Tingkat kedua: kata tunggal yang umum di spam judi. Satu kata saja tidak cukup
# ("slot waktu", "poker" di berita olahraga), jadi yang dinilai adalah kepadatannya.
GAMBLING_WORDS = {"judi", "perjudian", "penjudi", "taruhan", "betaruh", "bettor", "kasino", "casino", "togel",
                  "gacor", "jackpot", "sicbo", "poker", "slot", "toto", "maxwin", "scatter", "rtp", "qq",
                  "depo", "freebet"}   # "bandar" (Bandar Lampung, bandar udara), "spin", "wd" terlalu ambigu

MIN_WORDS = 20
GAMBLING_MAX_HITS = 3          # 🟡 frasa kuat; dikalibrasi lewat audit sampel (notes/audit-fineweb2.md)
GAMBLING_MAX_DENSITY = 0.01    # 🟡 proporsi karakter teks yang berupa frasa kuat
GAMBLING_WORD_RATE = 0.01      # 🟡 ≥1% kata adalah kata judi (dan ≥4 kata) → spam


def gambling_hits(text: str) -> tuple[int, float]:
    low = text.lower()
    hits = _GAMBLING.findall(low)
    density = sum(len(h) for h in hits) / max(1, len(low))
    return len(hits), density


def gambling_word_rate(words: list[str]) -> tuple[int, float]:
    n = sum(1 for w in words if w in GAMBLING_WORDS)
    return n, n / max(1, len(words))


def quality_reason(text: str) -> str | None:
    """None = lolos; selain itu alasan dokumen dibuang."""
    words = _WORD.findall(text.lower())
    if len(words) < MIN_WORDS:
        return "terlalu_pendek"
    n, dens = gambling_hits(text)
    if n >= GAMBLING_MAX_HITS or (n > 0 and dens >= GAMBLING_MAX_DENSITY):
        return "spam_judi"
    nw, rate = gambling_word_rate(words)
    if nw >= 4 and rate >= GAMBLING_WORD_RATE:
        return "spam_judi"
    return None


# ---------------------------------------------------------------------------
# Dekontaminasi n-gram kata
# ---------------------------------------------------------------------------
NGRAM = 13          # standar GPT-3 untuk teks panjang
SHORT_MIN = 8       # item evaluasi 8–12 kata dicocokkan utuh (n = panjang item)
WEBTEST_OVERLAP = 0.5  # ambang near-duplicate terhadap test web (diagnosis: median overlap template 6,8%)
_MUL = np.uint64(0x9E3779B97F4A7C15)


def word_ids(text: str) -> np.ndarray:
    words = _WORD.findall(text.lower())
    return np.fromiter((zlib.crc32(w.encode("utf-8")) for w in words), dtype=np.uint64, count=len(words))


def ngram_hashes(ids: np.ndarray, n: int) -> np.ndarray:
    """Hash polinomial (mod 2^64) untuk setiap n-gram berurutan."""
    if len(ids) < n:
        return np.empty(0, np.uint64)
    with np.errstate(over="ignore"):
        h = np.zeros(len(ids) - n + 1, np.uint64)
        for i in range(n):
            h = h * _MUL + ids[i:len(ids) - n + 1 + i] + np.uint64(i + 1)
    return h


class NgramIndex:
    """Indeks n-gram dari teks evaluasi. `matches(text)` → True bila dokumen memuat isi evaluasi."""

    def __init__(self):
        self.sets: dict[int, np.ndarray] = {}

    @classmethod
    def build(cls, texts, n: int = NGRAM, max_doc_freq: int | None = None):
        idx = cls()
        buckets: dict[int, list] = {}
        for t in texts:
            ids = word_ids(t)
            if len(ids) >= n:
                buckets.setdefault(n, []).append(np.unique(ngram_hashes(ids, n)))
            elif len(ids) >= SHORT_MIN:
                buckets.setdefault(len(ids), []).append(ngram_hashes(ids, len(ids)))
        for k, arrs in buckets.items():
            allh = np.concatenate(arrs) if arrs else np.empty(0, np.uint64)
            if max_doc_freq is not None:
                # Buang n-gram yang muncul di banyak dokumen evaluasi (boilerplate, bukan konten uji)
                u, c = np.unique(allh, return_counts=True)
                idx.sets[k] = u[c <= max_doc_freq]
            else:
                idx.sets[k] = np.unique(allh)
        return idx

    def merge(self, other: "NgramIndex") -> "NgramIndex":
        for k, v in other.sets.items():
            self.sets[k] = np.union1d(self.sets.get(k, np.empty(0, np.uint64)), v)
        return self

    def matches(self, text: str) -> bool:
        ids = word_ids(text)
        for n, s in self.sets.items():
            if len(s) == 0 or len(ids) < n:
                continue
            h = ngram_hashes(ids, n)
            pos = np.searchsorted(s, h)
            pos[pos == len(s)] = 0
            if np.any(s[pos] == h):
                return True
        return False

    def overlap_fraction(self, text: str, n: int = NGRAM) -> float:
        """Proporsi n-gram dokumen yang ada di indeks (dipakai untuk near-duplicate terhadap test web)."""
        s = self.sets.get(n)
        ids = word_ids(text)
        if s is None or len(s) == 0 or len(ids) < n:
            return 0.0
        h = ngram_hashes(ids, n)
        pos = np.searchsorted(s, h)
        pos[pos == len(s)] = 0
        return float(np.mean(s[pos] == h))

    def save(self, path: str):
        np.savez(path, **{str(k): v for k, v in self.sets.items()})

    @classmethod
    def load(cls, path: str) -> "NgramIndex":
        idx = cls()
        data = np.load(path)
        idx.sets = {int(k): data[k] for k in data.files}
        return idx

    def size(self) -> int:
        return int(sum(len(v) for v in self.sets.values()))
