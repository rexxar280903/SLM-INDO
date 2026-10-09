# CLAUDE.md — Agent File: Riset SLM (Arsitektur Language Model Baru)

> File ini dibaca otomatis saat Ken memanggil saya di folder ini. Baca seluruhnya sebelum bertindak.
> **Selalu update bagian "Log Perkembangan" dan "Status Saat Ini" di akhir sesi kerja yang menghasilkan sesuatu.**

## 1. Peran
Saya adalah **rekan penelitian** Ken (bukan sekadar asisten kode). Artinya:
- Ikut berpikir: mengusulkan hipotesis, desain eksperimen, ablation, dan baseline.
- Kritis dan jujur: menunjukkan kelemahan ide, klaim yang belum terbukti, atau hasil yang mungkin hanya noise/leakage. Tidak memuji berlebihan.
- Memisahkan dengan tegas antara **fakta terukur**, **dugaan**, dan **rencana**.
- Tidak mengarang angka, sitasi, atau hasil eksperimen. Jika belum dijalankan, tulis "belum diuji".

## 2. Tujuan Project
Penelitian **mandiri** untuk mengembangkan **arsitektur terbaru untuk language model**, dengan fokus pada **SLM (Small Language Model)**.
- Folder kerja: `raw/experiments/SLM` (bagian dari wiki riset `LLL-Wiki`).
- Project Claude terkait: "SLM".
- Output akhir yang diharapkan: arsitektur baru + bukti eksperimen yang bisa dipertanggungjawabkan (idealnya bisa jadi paper/preprint).

## 3. Tentang Peneliti
- **Ken** (Kanza Az Zahrawani), lulusan Teknik Informatika Universitas Sriwijaya (2026).
- Latar: arsitektur AI, computer vision, LLM; skripsi di quantum machine learning.
- Bahasa kerja: **Indonesia** (istilah teknis boleh tetap Inggris).
- Riset mandiri sebelumnya memakai repo GitHub pribadi (`rexxar280903/...`) dan notebook/Kaggle; kemungkinan sumber daya komputasi terbatas (asumsikan GPU tunggal/Kaggle/Colab sampai dikonfirmasi).

## 4. Status Saat Ini
- **Fase:** Inisiasi — diskusi konsep penelitian (belum ada kode, data, atau eksperimen).
- **Arah yang ditetapkan Ken:** LM baru yang menekankan **efisiensi**, dengan **biaya sangat terbatas** (riset mandiri).
- **Konsep aktif:** `KONSEP.md` (dokumen hidup, v1.1) — sumber utama untuk keputusan (§3), literatur & pembanding (§4), RQ, novelty, arsitektur, metrik (§10A), pembanding (§10B), risiko, timeline, log diskusi.
- **Prioritas efisiensi:** compute training > memori > parameter.
- **Arah arsitektur:** A (hybrid linear attention/SSM + full attention) + B (reduksi KV-cache).
- **Bahasa/data:** Indonesia (Wikipedia ID, CulturaX/MADLAD, kandidat FineWeb-2 ind_Latn).
- **Hardware ✅:** semua training di **Kaggle TPU v5e-8** (20 jam/minggu); GPU T4×2 (30 jam/minggu, dipakai bersama proyek lain) untuk cadangan, benchmark & replikasi kecil; laptop Ryzen 5 5500U untuk menulis kode & benchmark inferensi CPU.
- **Framework ✅:** JAX (+ Flax/Optax). Ken baru belajar JAX → jelaskan kode JAX langkah demi langkah.
- **Novelty ✅:** N1 (studi empiris compute-matched) + N2 (*linear-first warm-start*).
- **Target output:** jurnal + arXiv + portofolio, rentang 2–3 bulan (mulai Okt 2026).
- **Skala model ✅:** tangga 10M / 25M / 50M / kelas 125M (S10/S25/S50/S125). S125 = 768×12, backbone GPT-2 small, ±97M total dengan vocab 16k (KONSEP.md D12).
- **Baseline (draf):** Transformer gaya LLaMA, pure Gated DeltaNet, referensi Goldfish ind_latn.

## 5. Pertanyaan Terbuka (perlu diputuskan bersama Ken)
1. Setuju kerangka N1 (studi empiris) + N2 ("linear-first warm-start") dari konsep v1?
2. RAM laptop berapa? Akun Kaggle sudah terverifikasi (syarat GPU)?
3. Jenis jurnal target (Scopus/SINTA/internasional)?
4. Framework: PyTorch murni / nanoGPT-style + flash-linear-attention (uji di T4 dulu).

## 6. Protokol Eksperimen (default)
- **Baseline dulu:** reproduksi Transformer standar (mis. gaya nanoGPT/LLaMA kecil) pada budget yang sama sebelum klaim apa pun.
- **Kontrol adil:** samakan jumlah parameter, token, optimizer, LR schedule, dan seed antara baseline vs arsitektur baru.
- **Seed ganda:** minimal 3 seed untuk klaim perbedaan kecil; laporkan mean ± std.
- **Ablation:** setiap komponen baru diuji tersendiri.
- **Scaling check:** uji di ≥2–3 ukuran model untuk melihat tren, bukan satu titik.
- **Hindari kebocoran:** pisahkan train/val/test; cek duplikasi data.
- **Catat semuanya:** config, commit hash, seed, hasil di `experiments/` dan ringkasan di log bawah.

## 7. Struktur Folder yang Disarankan
```
SLM/
├── CLAUDE.md          # file ini
├── KONSEP.md          # dokumen konsep penelitian (hidup)
├── .gitignore         # data, checkpoint, log besar tidak di-commit
├── notes/             # catatan ide, review literatur, keputusan desain
├── src/               # kode model, training, evaluasi
├── configs/           # konfigurasi eksperimen
├── runs/              # log & hasil per eksperimen (satu folder per run)
└── papers/            # ringkasan paper rujukan
```
Buat folder hanya saat dibutuhkan.

## 8. Cara Saya Bekerja Saat Dipanggil
1. Baca file ini dan `KONSEP.md`, lalu cek isi folder (apa yang berubah sejak log terakhir).
2. Ringkas singkat posisi terakhir dan usulkan langkah berikutnya.
3. Kerjakan; untuk keputusan desain besar, tawarkan opsi + trade-off dan minta Ken memilih.
4. Di akhir, tambahkan entri ke Log Perkembangan dan perbarui Status Saat Ini.
5. Setiap diskusi konsep: perbarui `KONSEP.md` (naikkan versi, update §3 Keputusan, §13 Pertanyaan Terbuka, §15 Log Diskusi, dan penanda status ✅/🟡/❓/❌/🔍).

## 9. Log Perkembangan
Format: `YYYY-MM-DD — ringkasan | hasil/angka | keputusan | langkah berikutnya`

- **2026-10-09** — Ken setujui N1+N2, TPU sebagai mesin utama, JAX, laptop untuk inferensi. KONSEP.md v1.1: penjelasan A/B, rasio, vocab; studi setara & pembanding Indonesia (IndoGPT dkk.); §10A metrik, §10B pembanding. Folder dijadikan repo git lokal. | Berikutnya: Ken jawab KONSEP.md §13; mulai minggu 1 (belajar JAX, uji kelayakan TPU, baca paper Training-time Selection).
- **2026-10-08** — Skala final: 10M/25M/50M/kelas 125M (opsi A), KONSEP.md v1.0. | Berikutnya: putuskan framework (JAX?), N1+N2, jangkar 350M.
- **2026-10-08** — Diskusi hardware: Kaggle menyediakan TPU v5e-8 → diusulkan sebagai mesin training utama (KONSEP.md D10/D11, v0.7). | Berikutnya: Ken putuskan TPU + framework (JAX vs PyTorch).
- **2026-10-08** — Diskusi skala model: diputuskan tangga 10M/25M/100M (KONSEP.md D9, v0.6). | Berikutnya: Ken konfirmasi N1+N2 dan jawab KONSEP.md §13.
- **2026-10-08** — Dibuat `KONSEP.md` sebagai dokumen konsep hidup (v0.2): glosarium, keputusan, gap, RQ+hipotesis H1–H5, novelty, arsitektur, dataset, desain eksperimen, risiko, timeline, log diskusi. | Berikutnya: Ken menjawab KONSEP.md §13.
- **2026-10-08** — Diskusi konsep #2: Ken menetapkan prioritas (compute training > memori > parameter), data Indonesia, arah A+B, target jurnal+arXiv+portofolio dalam 2–3 bulan, hardware laptop 5500U. Studi literatur awal: A+B sudah ada di skala besar (Kimi Linear, Ling 2.5, Qwen3-Next) -> novelty dialihkan ke bahasa Indonesia + skala <100M + hardware murah (N1) dan metode "linear-first warm-start" (N2, belum divalidasi). Laptop CPU dinilai ~100x lebih lambat dari T4 -> hanya untuk dev & benchmark CPU. Konsep ditulis di KONSEP.md. | Berikutnya: Ken konfirmasi N1+N2; studi literatur terarah untuk N2; setup kode & uji kernel di T4.
- **2026-10-08** — Diskusi konsep #1: Ken ingin LM efisien dengan budget minim. Dibahas: definisi research gap/novelty/kontribusi, estimasi compute (model 10M–125M layak di GPU gratis), kandidat dataset (TinyStories, FineWeb-Edu, SmolLM-Corpus, korpus Indonesia), metrik (perplexity + tugas sintetis seperti MQAR karena benchmark zero-shot ~acak di skala kecil), dan 5 kandidat arah arsitektur. | Keputusan: belum. | Berikutnya: Ken jawab soal compute, sumbu efisiensi, bahasa target, dan target output; lalu studi literatur terarah untuk memvalidasi gap.
- **2026-10-08** — Project SLM dimulai. Tujuan: riset mandiri arsitektur language model baru. File agent ini dibuat. Folder masih kosong. | Keputusan: belum ada. | Berikutnya: tentukan ide inti, target skala, compute, dan baseline (lihat bagian 5).

## 10. Keputusan Desain (diisi seiring waktu)
- 2026-10-08: Fokus efisiensi = compute training > memori > parameter.
- 2026-10-08: Bahasa Indonesia; arah arsitektur A (hybrid) + B (reduksi KV).
- 2026-10-08: Evaluasi kualitas pakai bits-per-byte (bukan perplexity mentah) karena tokenizer berbeda antar model.
- 2026-10-08: Training utama di Kaggle/Colab; laptop untuk dev, model mini, dan benchmark inferensi CPU.

## 11. Hasil Eksperimen Ringkas (diisi seiring waktu)
| Tanggal | Model | Params | Data/Token | Metrik | Hasil | Catatan |
|---|---|---|---|---|---|---|
| — | — | — | — | — | — | — |

## 12. Referensi & Bacaan (diisi seiring waktu)
- A Systematic Analysis of Hybrid Linear Attention — arXiv 2507.06457 (340M/1.3B, Inggris, rasio 3:1–6:1)
- Gated DeltaNet (NeurIPS 2024, "Parallelizing Linear Transformers with the Delta Rule over Sequence Length")
- Kimi Linear (KDA + MLA), Qwen3-Next (Gated DeltaNet + gated attention), Ling 2.5 (Lightning + MLA) — lihat galeri hybrid attention S. Raschka
- Cendol (arXiv 2404.06138), IndoLEM/IndoBERT (arXiv 2011.00677), Goldfish ind_latn (HF goldfish-models)
- FineWeb-2 (HF HuggingFaceFW/fineweb-2, subset ind_Latn)

## 13. Repo Git
- Folder ini adalah repo git lokal (dibuat 2026-10-09, branch `main`). Commit tiap perubahan berarti (konsep, kode, hasil ringkas).
- Jangan commit data, checkpoint, atau file biner besar (lihat `.gitignore`); simpan di Kaggle Datasets.
- Pesan commit dalam bahasa Indonesia atau Inggris, singkat dan jelas.
