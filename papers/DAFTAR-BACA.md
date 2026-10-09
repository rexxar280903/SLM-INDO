# Daftar Bacaan — Riset SLM

PDF diunduh dengan `unduh_paper.ps1` (jalankan sekali di PC). File PDF tidak di-commit ke git.
Centang `[x]` setelah dibaca, dan tulis 2–3 baris catatan di bawah tiap paper (atau minta Claude merangkumnya).

## 🔴 Wajib — minggu 1

| # | Status | Paper | Kenapa dibaca | Fokus saat membaca |
|---|---|---|---|---|
| W1 | [ ] | **Training-time Selection of Linear Vs. Softmax Attention in Layer-based Hybrid Transformers** — [OpenReview](https://openreview.net/forum?id=4cP69sGjiG) | Berpotensi tumpang-tindih dengan N2 (*linear-first warm-start*) | Apakah mereka mulai dari semua linear lalu beralih ke softmax? Skala? Hemat compute? |
| W2 | [ ] | ✅ diunduh · **A Systematic Analysis of Hybrid Linear Attention** — [arXiv 2507.06457](https://arxiv.org/abs/2507.06457) | Rujukan utama rasio hybrid (3:1–6:1) | Tabel rasio vs kualitas/recall, setting training, keterbatasan |
| W3 | [ ] | ✅ diunduh · **Gated Delta Networks: Improving Mamba2 with Delta Rule** — [arXiv 2412.06464](https://arxiv.org/abs/2412.06464) | Mixer linear utama kita | Rumus Gated DeltaNet, bentuk chunkwise (yang akan kita tulis di JAX) |
| W4 | [ ] | **Hybrid Architectures for Language Models: Systematic Analysis and Design Insights** — [arXiv 2510.04800](https://arxiv.org/abs/2510.04800) | Studi hybrid terbaru (Meta), termasuk titik 100M | Rasio 1:5, penempatan layer attention, hasil skala 100M |
| W5 | [ ] | **IndoNLG (IndoGPT)** (= I1) — [arXiv 2104.08200](https://arxiv.org/abs/2104.08200) | Pembanding Indonesia dengan backbone sama (768×12) | Data Indo4B-Plus, tokenizer 40k, setting pretraining, PPL |

## 🇮🇩 Model & benchmark bahasa Indonesia — rujukan & pembanding (folder `indonesia/`)

| # | Status | Paper | Peran untuk kita | Fokus saat membaca |
|---|---|---|---|---|
| I1 | [ ] | **IndoNLG (IndoGPT)** — [arXiv 2104.08200](https://arxiv.org/abs/2104.08200) ✅ sudah diunduh | **Pembanding utama**: IndoGPT ±117M, backbone 768×12 = S125 | Data Indo4B-Plus, tokenizer 40k, setting pretraining, PPL validasi |
| I2 | [ ] | **Goldfish: Monolingual LMs for 350 Languages** — [arXiv 2408.10441](https://arxiv.org/abs/2408.10441) | **Pembanding**: LM Indonesia kecil (≤125M) dari nol | Ukuran model per data, tokenizer, protokol evaluasi FLORES |
| I3 | [ ] | **IndoNLU** — [arXiv 2009.05387](https://arxiv.org/abs/2009.05387) | Sumber tugas downstream **SmSA** & benchmark NLU | Deskripsi SmSA, split data, metrik (F1) |
| I4 | [ ] | **IndoLEM & IndoBERT** — [arXiv 2011.00677](https://arxiv.org/abs/2011.00677) | Konteks LM Indonesia & data pretraining | Korpus Indonesia yang dipakai, ukuran data |
| I5 | [ ] | **Cendol** — [arXiv 2404.06138](https://arxiv.org/abs/2404.06138) | Konteks LLM Indonesia (fine-tune, bukan dari nol) | Bagaimana mereka mengevaluasi kemampuan bahasa Indonesia |
| I6 | [ ] | **Sailor: Open LMs for South-East Asia** — [arXiv 2404.03608](https://arxiv.org/abs/2404.03608) | Model SEA kecil (0,5B+) termasuk Indonesia; rujukan tokenizer & data | Pembersihan data Indonesia, efisiensi tokenizer |
| I7 | [ ] | **SEA-LION** — [arXiv 2504.05747](https://arxiv.org/abs/2504.05747) | Model regional SEA; rujukan data & evaluasi | Komposisi data Indonesia, benchmark yang dipakai |
| I8 | [ ] | **IndoMMLU** — [arXiv 2310.04928](https://arxiv.org/abs/2310.04928) | Benchmark pengetahuan Indonesia (terlalu sulit untuk ≤125M, konteks saja) | Mengapa tidak kita pakai sebagai metrik utama |
| I9 | [ ] | **NusaCrowd** — [arXiv 2212.09648](https://arxiv.org/abs/2212.09648) | Katalog dataset NLP Indonesia | Sumber data latih/evaluasi tambahan |

Pembanding tanpa paper: `cahya/gpt2-small-indonesian-522M` dan `w11wo/indo-gpt2-small` (hanya model card Hugging Face).

## 🟡 Penting — minggu 2–3

| # | Status | Paper | Kenapa dibaca |
|---|---|---|---|
| P1 | [ ] | **Parallelizing Linear Transformers with the Delta Rule over Sequence Length** — [arXiv 2406.06484](https://arxiv.org/abs/2406.06484) | Algoritma chunkwise DeltaNet (dasar implementasi JAX) |
| P2 | [ ] | **Zoology: Measuring and Improving Recall in Efficient Language Models** — [arXiv 2312.04927](https://arxiv.org/abs/2312.04927) | Definisi & protokol MQAR |
| P3 | [ ] | **Based: Simple linear attention LMs balance the recall-throughput tradeoff** — [arXiv 2402.18668](https://arxiv.org/abs/2402.18668) | Teori recall vs ukuran state; hybrid linear + sliding window |
| P4 | [ ] | **Reducing Transformer KV Cache Size with Cross-Layer Attention** — [arXiv 2405.12981](https://arxiv.org/abs/2405.12981) | Varian KV "CLA" |
| P5 | [ ] | **Hymba: A Hybrid-head Architecture for Small Language Models** — [arXiv 2411.13676](https://arxiv.org/abs/2411.13676) | Contoh A+B untuk SLM |
| P6 | [ ] | **Hybrid Linear Attention Done Right (HALO/HypeNet)** — [arXiv 2601.22156](https://arxiv.org/abs/2601.22156) | Arah kebalikan N2 (Transformer → hybrid); pembeda N2 |
| P7 | [ ] | **Goldfish: Monolingual Language Models for 350 Languages** — [arXiv 2408.10441](https://arxiv.org/abs/2408.10441) | Pembanding Indonesia & protokol evaluasi FLORES |

## ⚪ Pendukung — dibaca bila perlu

| # | Status | Paper | Kenapa |
|---|---|---|---|
| S1 | [ ] | **GQA** — [arXiv 2305.13245](https://arxiv.org/abs/2305.13245) | Varian KV "GQA" |
| S2 | [ ] | **DeepSeek-V2 (MLA)** — [arXiv 2405.04434](https://arxiv.org/abs/2405.04434) | Varian KV "MLA" |
| S3 | [ ] | **Kimi Linear** — [arXiv 2510.26692](https://arxiv.org/abs/2510.26692) 🔍 cek ID | Hybrid KDA + MLA (A+B di skala besar) |
| S4 | [ ] | **Mamba-2 / Transformers are SSMs** — [arXiv 2405.21060](https://arxiv.org/abs/2405.21060) | Latar belakang SSM & dualitas dengan linear attention |
| S5 | [ ] | **Chinchilla (Training Compute-Optimal LLMs)** — [arXiv 2203.15556](https://arxiv.org/abs/2203.15556) | Dasar aturan ±20 token/parameter |
| S6 | [ ] | **IndoLEM & IndoBERT** — [arXiv 2011.00677](https://arxiv.org/abs/2011.00677) | Konteks NLP Indonesia |
| S7 | [ ] | **Cendol** — [arXiv 2404.06138](https://arxiv.org/abs/2404.06138) | Konteks LLM Indonesia (fine-tune) |

## Catatan bacaan
<!-- Tulis ringkasan per paper di sini, mis. "W2: ..." -->
