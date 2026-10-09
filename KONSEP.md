# KONSEP PENELITIAN — SLM Efisien untuk Bahasa Indonesia

> **Dokumen hidup.** Ini satu-satunya sumber konsep penelitian. Claude memperbaruinya setiap selesai diskusi.
> Versi: **v1.6** · Terakhir diperbarui: **2026-10-09** · Fase: **Penyusunan konsep**

**Penanda status** (dipakai di seluruh dokumen):
- ✅ **Disepakati**: sudah diputuskan Ken
- 🟡 **Diusulkan**: usulan, belum disetujui
- ❓ **Terbuka**: perlu dijawab/diputuskan
- ❌ **Ditolak**: dibuang (alasan dicatat)
- 🔍 **Perlu verifikasi**: klaim yang belum dicek lewat literatur/eksperimen

**Aturan pembaruan:** perubahan keputusan dicatat di §3 dan §15. Isi lama yang diganti tidak dihapus diam-diam: pindahkan ke §15 dengan alasannya.

---

## 0. Ringkasan Singkat
Membangun **small language model (SLM) bahasa Indonesia** yang efisien, dengan menggabungkan **hybrid linear attention** dan **reduksi KV-cache**, dilatih dari nol dengan **compute gratis** (utama: TPU v5e-8 Kaggle, kode JAX). Prioritasnya adalah efisiensi compute training, lalu memori, lalu parameter. Kontribusi yang dibidik: (1) studi empiris compute-matched pertama untuk setting ini, dan (2) metode usulan *linear-first warm-start* untuk menghemat compute training. Target: preprint arXiv, submit jurnal, dan rilis kode/model sebagai portofolio dalam 2–3 bulan.

---

## 1. Glosarium Istilah Penelitian

| Istilah                    | Arti                                                                                                     | Contoh di penelitian ini                                                     |
| -------------------------- | -------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| **Research gap**           | Hal yang belum dijawab penelitian sebelumnya                                                             | Studi hybrid attention belum dilakukan untuk bahasa Indonesia di skala ≤ kelas 125M |
| **Research question (RQ)** | Pertanyaan spesifik yang mengisi gap dan bisa dijawab dengan eksperimen                                  | "Rasio hybrid mana yang terbaik per GPU-jam?"                                |
| **Hipotesis**              | Jawaban dugaan atas RQ yang bisa terbukti salah                                                          | "Rasio 3:1 menyamai Transformer dengan KV-cache 4× lebih kecil"              |
| **Novelty**                | Hal baru yang ditawarkan: mekanisme, kombinasi, setting, atau temuan                                     | Studi pertama di bahasa Indonesia + metode *linear-first warm-start*         |
| **Kontribusi**             | Daftar konkret yang diserahkan ke komunitas                                                              | Model, tokenizer, kode, hasil benchmark                                      |
| **Baseline**               | Pembanding standar dengan budget yang sama                                                               | Transformer gaya LLaMA                                                       |
| **Ablation**               | Mencopot satu komponen untuk melihat pengaruhnya                                                         | Hybrid tanpa reduksi KV vs dengan                                            |
| **Compute-matched**        | Perbandingan dengan jumlah compute (FLOPs/GPU-jam) yang sama                                             | Semua model dilatih 2 GPU-jam                                                |
| **Linear attention / SSM** | Mekanisme pengganti attention dengan biaya linear terhadap panjang konteks, tanpa KV-cache yang membesar | Gated DeltaNet, Mamba-2, HGRN2                                               |
| **Hybrid**                 | Model yang mencampur layer linear dan layer full attention                                               | Rasio 3:1 = 3 layer linear per 1 layer attention                             |
| **KV-cache**               | Memori yang menyimpan key/value token sebelumnya saat generate; tumbuh seiring panjang konteks           | Dikurangi dengan GQA/MLA/sharing                                             |
| **GQA / MLA / CLA**        | Grouped-Query Attention / Multi-head Latent Attention / Cross-Layer Attention (KV dibagi antar layer)    | Varian reduksi KV yang diuji                                                 |
| **Bits-per-byte (bpb)**    | Metrik kualitas LM yang adil antar tokenizer berbeda (makin kecil makin baik)                            | Metrik kualitas utama                                                        |
| **MQAR**                   | Multi-Query Associative Recall: tugas sintetis untuk menguji kemampuan "mengingat" konteks               | Uji recall murah untuk model kecil                                           |
| **Fertility**              | Rata-rata jumlah token per kata hasil tokenizer                                                          | Tokenizer Indonesia → fertility rendah → hemat compute                       |
| **Arah A (hybrid)** | Model yang sebagian layer-nya memakai *linear attention* (murah, memori tetap) dan sebagian memakai *full attention* (mahal, ingatan tajam) | 9 layer Gated DeltaNet + 3 layer attention |
| **Arah B (reduksi KV-cache)** | Teknik memperkecil memori KV-cache di layer full attention yang tersisa | GQA, MLA, CLA, sliding window |
| **Rasio hybrid (linear:full)** | Jumlah layer linear attention untuk setiap 1 layer full attention | 3:1 pada 12 layer → 9 linear + 3 full |
| **Vocab (ukuran kosakata tokenizer)** | Jumlah token unik yang dikenal tokenizer. Tiap token punya satu baris di tabel embedding | 16k = 16.000 token unik |
| **Embedding** | Tabel yang mengubah token menjadi vektor angka. Ukurannya vocab × d_model; di-*tie* = tabel yang sama dipakai lagi di output | 16.000 × 768 ≈ 12,3M parameter |
| **Non-embedding (backbone)** | Semua parameter selain embedding: layer attention/linear attention, FFN, normalisasi. Inilah "otak" model dan ukuran yang adil untuk membandingkan arsitektur | S125: ±85M non-embedding |
| **Panjang token** | Rata-rata jumlah karakter yang diwakili satu token. Makin panjang, makin sedikit token per teks dan makin hemat compute | Target ±4–5 karakter/token untuk teks Indonesia 🔍 |
| **Pretraining** | Tahap pertama melatih LM **dari bobot acak** pada teks mentah dalam jumlah besar (tugas: menebak token berikutnya). Bukan berarti memakai model yang sudah dilatih orang lain. Lawan katanya *fine-tuning* (melatih lanjut model yang sudah ada) | Kita melakukan pretraining S10–S125 dari nol; fine-tuning hanya untuk uji SmSA |
| **Pretrained model** | Model yang sudah selesai di-pretraining (milik orang lain atau milik kita sendiri) | IndoGPT, Goldfish = pretrained model orang lain yang hanya kita **evaluasi** sebagai pembanding |

---

## 2. Batasan & Sumber Daya

| Sumber daya          | Detail                                                                                                                                         | Peran                                                                                        | Status            |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- | ----------------- |
| Laptop | AMD Ryzen 5 5500U (6C/12T, Zen 2, CPU-only) | Menulis kode & **pengujian/benchmark tahap inferensi** (bukan training) | ✅ |
| GPU gratis | Kaggle T4×2 (sesi maks. 12 jam, **kuota akun Ken: 30 jam/minggu**, dipakai bersama proyek lain) & Colab gratis | Cadangan bila TPU bermasalah, benchmark GPU, replikasi kecil untuk RQ3 | ✅ |
| TPU gratis           | Kaggle **TPU v5e-8** (8 chip × 197 TFLOPs bf16 puncak, 16 GB HBM/chip; sesi maks. 9 jam; **kuota akun Ken: 20 jam/minggu**, terpisah dari GPU) | **Mesin training utama** (semua skala S10–S125; uji kelayakan minggu 1) | ✅ |
| TPU riset (opsional) | Program Google TPU Research Cloud (TRC), berbasis aplikasi | Tambahan compute bila diterima; didaftarkan setelah proposal singkat siap | 🟡 opsi |
| Budget uang          | Sangat terbatas / nol                                                                                                                          | —                                                                                            | ✅                 |
| Waktu                | 2–3 bulan (mulai Okt 2026)                                                                                                                     | —                                                                                            | ✅                 |
| RAM laptop | — | Tidak menentukan lagi (laptop tidak dipakai untuk training/preprocessing berat) | ❌ tidak relevan |

**Estimasi compute TRAINING (sekali latih dari nol)**. Rumus kasar FLOPs ≈ 6 × parameter × token (2× forward + 4× backward); T4 efektif 10–20 TFLOPs; laptop 0,1–0,2 TFLOPs.
- **Parameter**: jumlah bobot model. 25M = 25 juta parameter.
- **Token latih**: total potongan teks (hasil tokenizer) yang dilihat model selama training. Patokan ±20 token per parameter (aturan Chinchilla): 25M × 20 ≈ 500 juta token. Untuk bahasa Indonesia dengan tokenizer sendiri, 500 juta token ≈ 350 juta kata ≈ 2–2,5 GB teks mentah (🔍 tergantung fertility tokenizer).

| Model | Token latih | FLOPs training | T4 | Laptop |
|---|---|---|---|---|
| ~25M | 0,5B (500 juta) | ~7×10¹⁶ | ~1–2 jam | ~4–8 hari |
| ~100M | 2B (2 miliar) | ~1,2×10¹⁸ | ~16–32 jam | tidak realistis |

**Estimasi INFERENSI (memakai model jadi)**. ≈ 2 × parameter FLOPs per token yang dihasilkan, sangat ringan. Di CPU, batasnya bandwidth memori: bobot dibaca ulang tiap token.

| Model | Ukuran bobot (fp32) | Perkiraan kecepatan di laptop 5500U |
|---|---|---|
| ~25M | ~100 MB | puluhan–ratusan token/detik |
| ~100M | ~400 MB | puluhan token/detik |

🔍 Angka throughput training & inferensi harus dikalibrasi (minggu 3–4).

**Perbandingan kasar akselerator Kaggle** (efektif = perkiraan setelah efisiensi nyata; 🔍 harus diukur):

| Akselerator | Puncak             | Efektif (perkiraan) | S10                  | S25       | S125 (2B token) | ~350M (7B token) |
| ----------- | ------------------ | ------------------- | -------------------- | --------- | --------------- | ---------------- |
| T4 ×1       | 65 TFLOPs fp16     | 10–20 TFLOPs        | 10–20 mnt            | 1–2 jam   | 17–33 jam       | 200–400 jam      |
| T4 ×2       | 130 TFLOPs fp16    | 20–40 TFLOPs        | 5–10 mnt             | 0,5–1 jam | 8–17 jam        | 100–200 jam      |
| TPU v5e-8   | ~1.576 TFLOPs bf16 | 250–500 TFLOPs      | < 1 mnt (+kompilasi) | 2–5 mnt   | 40–80 mnt       | 8–17 jam         |

Model kecil memakai chip TPU kurang efisien, sehingga utilisasi realistis rendah (asumsi 15–30%). Meski begitu, TPU tetap ±10–25× lebih cepat dari T4×2.

---

**Pembagian hardware** (D10 ✅, D15 🟡)

| Hardware | Tugas | Alasan |
|---|---|---|
| **TPU v5e-8** | **Semua training** S10, S25, S50, S125 (+ opsional 350M) | Satu hardware, satu presisi (bf16), satu kode → perbandingan antar skala adil; S10/S25 di TPU hanya butuh menit |
| **GPU T4×2** | (1) Cadangan bila TPU antre/gagal; (2) benchmark kecepatan training & inferensi di GPU; (3) **replikasi kecil**: baseline + konfigurasi terbaik di S25 dilatih ulang di GPU untuk menguji apakah peringkat sama (RQ3) | Kuota GPU dipakai bersama proyek lain; T4 tanpa bf16 |
| **Laptop 5500U** | Menulis kode; uji & benchmark inferensi CPU (latensi, throughput, memori) | Mewakili perangkat "edge" yang realistis |

Usulan Ken (S10 & S25 di GPU, S125 di TPU) secara teknis **bisa**, tetapi tidak disarankan sebagai desain utama:
- Tren skala tercampur efek hardware: beda presisi (fp16 vs bf16), beda kernel, beda angka acak. Perubahan peringkat antar skala tidak bisa dipastikan berasal dari skala.
- Perlu dua jalur kode (PyTorch+Triton di GPU, JAX di TPU) → beban rekayasa dua kali lipat.
- Kuota GPU terbatas & dipakai proyek lain, sedangkan S10/S25 di TPU hanya butuh menit.
- Bagian yang berguna dari ide ini dipertahankan lewat **replikasi kecil di GPU** untuk RQ3.

---

## 3. Keputusan yang Sudah Disepakati

| # | Tanggal | Keputusan | Status |
|---|---|---|---|
| D1 | 2026-10-08 | Fokus: LM baru yang menekankan efisiensi, riset mandiri berbiaya rendah | ✅ |
| D2 | 2026-10-08 | Urutan prioritas efisiensi: **compute training > memori > parameter** | ✅ |
| D3 | 2026-10-08 | Bahasa: **Indonesia** (Wikipedia ID, CulturaX/MADLAD; kandidat FineWeb-2 `ind_Latn`) | ✅ |
| D4 | 2026-10-08 | Arah arsitektur: **A** (hybrid linear + full attention) + **B** (reduksi KV-cache) | ✅ |
| D5 | 2026-10-08 | Output: jurnal + arXiv + portofolio, rentang 2–3 bulan | ✅ |
| D6 | 2026-10-08 | Training utama di GPU gratis; laptop untuk dev & benchmark CPU | ❌ digantikan D10 & D14 |
| D7 | 2026-10-08 | Kualitas diukur dengan **bits-per-byte** (bukan perplexity mentah) | 🟡 |
| D8 | 2026-10-08 | Kerangka novelty **N1 + N2** (lihat §7) | ✅ (disetujui 2026-10-09) |
| D9 | 2026-10-08 | Tangga skala model **10M / 25M / 100M** (opsi 10/25/50 tidak dipilih) | ✅ diperbarui oleh D12 |
| D10 | 2026-10-09 | **TPU v5e-8 Kaggle sebagai mesin training utama** untuk semua skala; T4×2 sebagai cadangan, benchmark GPU & replikasi kecil | ✅ |
| D11 | 2026-10-08 | Budget compute dinyatakan dalam **FLOPs** (netral hardware); seluruh run kualitas dalam satu studi dijalankan di hardware yang sama | 🟡 |
| D12 | 2026-10-08 | Tangga skala final **10M / 25M / 50M / kelas 125M**. Skala terbesar = **opsi A**: 768 × 12 (backbone GPT-2 small, ±85M non-embedding, ±97M total dengan vocab 16k). Titik 50M standar | ✅ |
| D13 | 2026-10-09 | Framework: **JAX** (+ Flax/Optax); Ken belajar sambil jalan | ✅ |
| D14 | 2026-10-09 | Laptop dipakai untuk menulis kode & **pengujian/benchmark inferensi**, bukan training | ✅ |
| D15 | 2026-10-09 | Pembagian hardware: semua training di TPU; GPU untuk cadangan + benchmark + replikasi kecil S25 (RQ3) | ✅ (disetujui 2026-10-09) |
| D16 | 2026-10-09 | Program TPU Research Cloud (TRC) sebagai opsi tambahan compute | 🟡 opsi |
| D17 | 2026-10-09 | Metrik evaluasi & daftar pembanding sesuai §10A–§10B | 🟡 usulan |
| D18 | 2026-10-09 | Rencana perbandingan hasil: G1 internal (klaim utama), G2 eksternal Indonesia (rujukan), G3 literatur — §10C | 🟡 usulan |
| D19 | 2026-10-09 | Target publikasi: **jurnal Scopus & SINTA** (venue lain diperbolehkan); preprint arXiv lebih dulu bila kebijakan jurnal mengizinkan | ✅ |
| D20 | 2026-10-09 | Vocab: **16k sebagai default** untuk semua skala + ablation vocab kecil (8k/16k/32k) di S25 untuk RQ5 | 🟡 usulan |
| D21 | 2026-10-09 | Jangkar ±350M: **opsional (stretch goal)**, hanya B0 + OURS, dijalankan bila S125 selesai ≤ minggu 8 dan sisa kuota ≥35 TPU-jam | 🟡 usulan |
| D22 | 2026-10-09 | Metrik dibagi **inti** (wajib) dan **pelengkap** (dikerjakan bila waktu cukup), lihat §10A | 🟡 usulan |

---

## 4. Latar Belakang & Temuan Literatur

**4.1 Apa itu arah A dan B**
- **A, hybrid linear attention + full attention.** *Full attention* (Transformer biasa) membandingkan setiap token dengan semua token sebelumnya. Ingatannya tajam, tapi biayanya naik kuadratik terhadap panjang teks, dan memori KV-cache-nya terus membesar. *Linear attention* (mis. Gated DeltaNet, Mamba) merangkum masa lalu ke dalam "buku catatan" berukuran tetap: murah dan memorinya konstan, tapi ingatannya kurang tajam. Model **hybrid** memakai keduanya: sebagian besar layer linear (murah), sedikit layer full attention (ingatan tajam).
- **B, reduksi KV-cache.** Layer full attention yang tersisa tetap butuh KV-cache. Arah B memperkecilnya: **GQA** (beberapa head berbagi key/value), **MLA** (key/value dikompres ke vektor laten kecil), **CLA** (beberapa layer berbagi key/value yang sama), **sliding window** (hanya menyimpan N token terakhir).
- **Contoh (S125, 12 layer, konteks 8k):** Transformer biasa ±300 MB KV-cache → hybrid 3:1 (3 layer attention) ±75 MB → hybrid 3:1 + GQA ±25 MB.

**4.2 Kombinasi A+B sudah dipakai di model besar** → menggabungkan A+B **bukan** novelty.
- Qwen3-Next: Gated DeltaNet + gated attention, rasio 3:1
- Kimi Linear: Kimi Delta Attention + MLA, ±3:1
- Ling 2.5: Lightning Attention + MLA, 7:1
- Hymba (NVIDIA, 1,5B): head attention & SSM paralel + CLA + sliding window → KV-cache 11,67× lebih kecil dari Llama-3.2-3B

**4.3 Studi sistematis/arsitektur yang setara (rujukan utama)**

| Studi | Fokus | Skala & data | Temuan penting | Relevansi untuk kita |
|---|---|---|---|---|
| A Systematic Analysis of Hybrid Linear Attention (2507.06457) | 6 jenis linear attention × rasio hybrid | 340M & 1,3B; FineWeb-Edu (Inggris) | Rasio 3:1–6:1 dengan HGRN-2/Gated DeltaNet; linear terbaik ≠ hybrid terbaik | Rujukan rasio & mixer; kita uji di ≤125M, bahasa Indonesia |
| Hybrid Architectures for LMs: Systematic Analysis & Design Insights (2510.04800, Meta) | Hybrid antar-layer vs dalam-layer (attention + Mamba) | 100M–3B (fokus ±1B, 60–73B token) | Rasio attention:Mamba 1:1 untuk kualitas, ±1:5 untuk efisiensi; jangan taruh attention di layer awal; hybrid dalam-layer punya frontier terbaik | Rujukan penempatan layer; titik 100M hanya untuk scaling |
| Based (2402.18668) | Linear attention + sliding window; trade-off recall vs ukuran state | 360M & 1,3B; Pile 10B token | Recall dibatasi ukuran state; kombinasi linear + window pulihkan sebagian besar recall | Kerangka teori recall-memori; protokol MQAR |
| Zoology (ICLR 2024) | Mendefinisikan tugas sintetis MQAR | Model kecil, data sintetis | Kesenjangan kualitas antar arsitektur sebagian besar dijelaskan oleh kemampuan recall | Protokol evaluasi recall |
| Cross-Layer Attention (2405.12981) | Berbagi KV antar layer | 1B & 3B | KV-cache 2× lebih kecil dengan penurunan kualitas kecil | Varian KV "CLA" |
| Hymba (2411.13676) | Hybrid paralel + CLA + sliding window + meta token | 1,5B | KV 11,67× lebih kecil, throughput 3,49× | Contoh A+B untuk SLM |
| HALO/HypeNet (2601.22156) | **Konversi** Transformer jadi → hybrid (attention→linear), 25% layer tetap attention | 2B–9B dari Qwen3 | Hanya ±2,3B token untuk konversi | Arah **kebalikan** N2 (pembeda N2) |
| Training-time Selection of Linear vs Softmax Attention in Layer-based Hybrid Transformers (OpenReview) | Memilih linear/softmax per layer saat training | 🔍 belum terbaca (halaman diblokir) | 🔍 | **Wajib dibaca**: berpotensi tumpang-tindih dengan N2 |

**4.4 LM bahasa Indonesia yang ada (pembanding)**

| Model | Jenis | Ukuran & backbone | Vocab | Data latih | Catatan |
|---|---|---|---|---|---|
| **IndoGPT** (IndoNLG, 2021) | Decoder dari nol | ±117M; **768 × 12** (= backbone S125) | 40k (SentencePiece BPE) | Indo4B-Plus ±23,8 GB (Indonesia + Sunda + Jawa) | Pembanding paling langsung untuk S125; PPL validasi ±90 |
| **cahya/gpt2-small-indonesian-522M** | Decoder dari nol (GPT-2 small) | ±124M; 768 × 12 | 52k (byte-level BPE) | Wikipedia ID 522 MB; *uncased* | Hati-hati kontaminasi bila dievaluasi di Wikipedia |
| **Goldfish `ind_latn`** (2024) | Decoder monolingual dari nol | ≤125M | 🔍 | ≤1 GB | Dievaluasi dengan perplexity FLORES → pakai FLORES yang sama |
| w11wo/indo-gpt2-small | Decoder (GPT-2 small) | 🔍 | 🔍 | 🔍 | Kandidat tambahan |
| Cendol | Fine-tune LLaMA-2/mT5 | 7B–13B | – | – | Bukan dari nol; konteks saja |
| IndoBERT/IndoLEM | Encoder | – | – | – | Bukan generatif; konteks saja |

🔍 Belum ditemukan LM Indonesia kecil berarsitektur hybrid/efisien yang dilatih dari nol (pencarian 2026-10-08 & 2026-10-09). Semua pembanding Indonesia di atas memakai **Transformer biasa**.

**4.5 Fakta teknis pendukung**
- Porsi FLOPs attention kausal relatif terhadap proyeksi+FFN ≈ **L/(12d)**. Contoh: d=512, L=2048 → ~33%. Pada model kecil (d kecil), attention kuadratik relatif lebih mahal, sehingga hybrid relevan untuk SLM.
- Di model kecil, embedding + LM head bisa ~separuh parameter, dan LM head memakan 2·V·d FLOPs/token. Ukuran vocab adalah tuas compute yang besar.

---

## 5. Research Gap 🟡
1. **Bahasa Indonesia × arsitektur efisien:** semua LM Indonesia kecil yang dilatih dari nol (IndoGPT, GPT-2 Indonesia, Goldfish) memakai Transformer biasa.
2. **Bahasa:** studi hybrid attention dan reduksi KV hampir seluruhnya di bahasa Inggris.
3. **Skala & budget:** studi sistematis yang ada berfokus ≥340M (2507.06457) atau ±1B (2510.04800, titik 100M hanya untuk kurva scaling), dengan puluhan miliar token di GPU kelas atas. Setting ≤ kelas 125M dengan compute gratis belum dikaji sebagai fokus utama.
4. **Hardware:** klaim efisiensi umumnya diukur di A100/H100 dengan kernel teroptimasi. Di TPU v5e, T4 (tanpa bf16 & FlashAttention-2), dan CPU laptop, peringkatnya bisa berbeda.

---

## 6. Research Questions & Hipotesis 🟡

| RQ  | Pertanyaan                                                                                                                 | Hipotesis                                                                                                          |
| --- | -------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| RQ1 | Pada budget compute (FLOPs) tetap, rasio linear:full attention mana yang memberi bpb terbaik untuk SLM Indonesia 10M – kelas 125M? | **H1:** rasio 3:1–6:1 mencapai bpb dalam ±2% dari Transformer, dengan KV-cache ≥4× lebih kecil                     |
| RQ2 | Teknik reduksi KV mana di layer full attention yang paling hemat memori tanpa merusak recall?                              | **H2:** GQA/CLA hampir gratis secara kualitas; MLA paling hemat memori tetapi menambah compute training            |
| RQ3 | Apakah peringkat efisiensi di literatur tetap berlaku di TPU v5e, GPU T4, dan CPU laptop?                                                | **H3:** di konteks pendek, keunggulan wall-clock linear attention mengecil atau hilang di T4/CPU; peringkat kualitas tetap sama antar hardware                   |
| RQ4 | Apakah *linear-first warm-start* menghemat compute training tanpa menurunkan kualitas akhir?                               | **H4:** menghemat ≥10–20% compute (FLOPs/TPU-jam) untuk mencapai bpb target dibanding hybrid dari nol                              |
| RQ5 | Seberapa besar pengaruh tokenizer Indonesia & ukuran vocab terhadap compute per byte?                                      | **H5:** tokenizer Indonesia dengan vocab 8k–16k lebih hemat compute per byte dibanding tokenizer multilingual umum |

---

**Penjelasan H1.** "Rasio 3:1–6:1" = jumlah layer linear attention untuk setiap 1 layer full attention. Di S125 (12 layer): 3:1 → 9 linear + 3 full; 6:1 → ±10 linear + 2 full. KV-cache hanya ada di layer full attention, jadi 3:1 → KV-cache **4× lebih kecil** (3 dari 12 layer), 6:1 → ±6× lebih kecil. "bpb dalam ±2%" = kualitas prediksi teks hampir setara Transformer penuh (mis. Transformer 1,00 bpb, hybrid ≤1,02 bpb).

**Penjelasan H5.** Yang dibandingkan adalah **ukuran vocab** (jumlah token unik): rencana kita 8k–16k (default 16k) vs tokenizer multilingual umum seperti GPT-2 (±50k), IndoGPT (40k), Llama-2 (32k), Qwen (±151k), Gemma (±256k). Vocab kecil → embedding & lapisan output lebih kecil → compute per token lebih rendah; tokenizer khusus Indonesia → teks dipecah jadi lebih sedikit token. Yang diukur: compute per **byte** teks, bukan per token.

---

## 7. Novelty & Kontribusi ✅ (D8)

**N1: studi empiris (fondasi, risiko rendah)**
Studi compute-matched pertama tentang hybrid + reduksi KV untuk SLM Indonesia ≤ kelas 125M, diukur di hardware gratis (TPU v5e, T4) dan CPU laptop, dibandingkan langsung dengan LM Indonesia Transformer yang ada (IndoGPT dkk.).

**N2: metode usulan *linear-first warm-start* (risiko sedang)** 🔍
Fase 1: semua layer memakai linear attention (murah). Fase 2: sebagian layer dialihkan ke full attention, memakai ulang proyeksi Q/K/V/O karena bentuknya identik. Tujuannya menghemat compute training (prioritas #1).
Status literatur: metode konversi yang ada (HALO/HypeNet 2601.22156, LoLCATs, STILL, KL-guided layer selection) berjalan **ke arah sebaliknya**, yaitu Transformer jadi → hybrid. 🔍 Paper "Training-time Selection of Linear vs Softmax Attention in Layer-based Hybrid Transformers" (OpenReview) wajib dibaca karena berpotensi tumpang-tindih.

**Rencana kontribusi:**
1. Analisis compute-matched hybrid × varian KV untuk bahasa Indonesia
2. Metode N2 beserta evaluasinya (termasuk jika hasilnya negatif)
3. Benchmark efisiensi di TPU v5e, T4 & CPU laptop
4. Rilis terbuka: tokenizer Indonesia, checkpoint model, kode training/evaluasi

Jika N2 gagal, N1 tetap layak terbit.

---

## 8. Rancangan Arsitektur 🟡

**Blok dasar:** gaya LLaMA (pre-RMSNorm, SwiGLU, RoPE pada layer attention), embedding input/output di-*tie*.

**Mixer linear:** Gated DeltaNet (utama); cadangan HGRN2/GLA (HGRN2 kuat di hybrid 340M menurut 4.2).

**Layer full attention:** disisipkan merata (mis. tiap layer ke-4 untuk 3:1). Posisi layer = ablation.

**Varian KV di layer attention:** MHA (acuan), GQA, MLA, CLA (berbagi KV antar layer attention), opsional sliding window.

**Konfigurasi ukuran ✅ (D12: tangga 10M / 25M / 50M / kelas 125M)**

| Nama | d_model × layer | Vocab (tied) | Non-embedding | Total | Porsi embedding | Konteks | Token latih | Waktu TPU/run (🔍) | Peran |
|---|---|---|---|---|---|---|---|---|---|
| S10 | 256 × 8 | 16k | ~6M | ~10,5M | ~40% | 1024 | ~0,2B | < 1 mnt + kompilasi | Saring banyak ide dengan murah |
| S25 | 384 × 10 | 16k | ~18M | ~24M | ~26% | 1024–2048 | ~0,5B | ~2–5 mnt | Grid utama + seed + N2 |
| S50 | 512 × 14 | 16k | ~44M | ~52M | ~16% | 2048 | ~1B | ~10–20 mnt | Titik tengah tren skala |
| S125 | 768 × 12 | 16k | ~85M | ~97M | ~13% | 2048 | ~2B | ~40–80 mnt | Kelas 125M: backbone identik GPT-2 small |

S125 dilaporkan sebagai "kelas 125M (backbone 85M)"; total parameter lebih kecil dari GPT-2 small (124M) hanya karena vocab 16k vs 50k.
Debug di laptop memakai versi mini (mis. 192 × 6, ±5M) dengan token sedikit.

**Opsi skala terbesar: 100M vs 125M** ✅ opsi A dipilih (Diskusi #10–11)

| Opsi                | d_model × layer | Vocab | Non-embedding | Total | Token latih | Waktu TPU/run |
| ------------------- | --------------- | ----- | ------------- | ----- | ----------- | ------------- |
| A (usulan)          | 768 × 12        | 16k   | ~85M          | ~97M  | ~2B         | ~40–80 mnt    |
| A'                  | 768 × 12        | 32k   | ~85M          | ~110M | ~2,2B       | ~50–90 mnt    |
| A'' (= GPT-2 small) | 768 × 12        | 50k   | ~85M          | ~123M | ~2,5B       | ~1–1,5 jam    |
| B                   | 768 × 16        | 16k   | ~113M         | ~126M | ~2,5B       | ~1–2 jam      |

- **Keterangan kolom:** *Vocab* = jumlah token unik tokenizer. *Non-embedding* = parameter "otak" model di luar tabel embedding; ini yang menentukan kemampuan arsitektur. *Total* = non-embedding + embedding (vocab × 768). Karena itu, keempat opsi punya non-embedding sama (±85M) kecuali B (16 layer → ±113M), dan total berbeda hanya karena vocab.
- Model kelas 125M adalah ukuran rujukan umum: GPT-2 small, GPT-Neo/OPT-125M, Mamba-130M, MobileLLM-125M, dan **Goldfish `ind_latn` (≤125M, dievaluasi dengan perplexity FLORES)**.
- Opsi A memakai **backbone yang sama dengan GPT-2 small**. Perbedaan total parameter hanya berasal dari vocab yang lebih kecil, yang memang sejalan dengan prioritas efisiensi.
- Opsi B mencapai 125M total, tetapi backbone-nya lebih besar dari model rujukan sehingga kurang sebanding.
- Di TPU, selisih biaya antar opsi kecil (±5–8 TPU-jam untuk seluruh run S125).
- Dengan skala terbesar ±4–5× dari 25M, titik **±50M dijadikan standar** → tangga 10 / 25 / 50 / 125-kelas (kelipatan ±2–2,5×).

**Evaluasi tangga 10M / 25M / 100M** (analisis awal sebelum TPU & D12; sebagian besar kekurangan skala terbesar teratasi oleh TPU dan titik 50M)

Kelebihan:
- Rentang 10× (tiga titik: ×2,5 lalu ×4) → tren skala lebih meyakinkan dibanding rentang 5×.
- 100M lebih dekat ke literatur (±3,4× di bawah 340M, bukan ±7×) dan setara kelas GPT-2 small → mudah dibandingkan model lain.
- Di 100M porsi embedding tinggal ±13%, sehingga perbedaan antar mixer lebih terlihat.
- d_model 768 → state linear attention lebih besar → kapasitas recall & manfaat hybrid di konteks panjang lebih mungkin muncul.
- Penghematan KV-cache bermakna secara absolut. Contoh di konteks 8k (fp16): MHA ±300 MB → hybrid 3:1 ±75 MB → hybrid 3:1 + GQA ±25 MB.
- Model 100M cukup layak untuk demo portofolio dan benchmark inferensi CPU.
- 10M & 25M tetap murah untuk eksplorasi luas dan banyak seed.

Kekurangan:
- Satu run 100M (17–33 jam) melebihi batas sesi Kaggle 12 jam → wajib checkpoint & resume lintas 2–3 sesi.
- Hanya 3–4 run di 100M, biasanya 1 seed → tidak ada error bar di skala yang paling penting.
- Lompatan 25M → 100M (4×) tanpa titik tengah: kalau peringkat berubah, kita tidak tahu di mana berubahnya.
- 100M memakan sebagian besar kuota: dengan ±30 jam/minggu, 100M saja ≈ 2,5–4 minggu kuota (1× T4) → tekanan jadwal.
- Data 2B token ≈ 8–10 GB teks; tokenisasi di laptop butuh waktu dan ruang disk.
- Training lebih panjang di fp16 (T4 tanpa bf16) → risiko instabilitas lebih besar.
- Masih di bawah 340M → kritik "skala kecil" berkurang, tapi tidak hilang.

Mitigasi:
1. **Baseline 100M dijalankan lebih awal** (minggu 4–5). Baseline Transformer tidak bergantung hasil grid, sehingga kuota tersebar.
2. **Kaggle T4×2 + DDP** untuk 100M → waktu per run ±10–17 jam.
3. **Checkpoint tiap 30–60 menit** (bobot, optimizer, posisi data, RNG); uji fitur resume di S10 sebelum dipakai di S125.
4. **Konfigurasi 100M dipilih belakangan**: baseline + 2–3 terbaik yang konsisten di 10M dan 25M.
5. **Tanpa seed ganda di 100M**: variansi seed di 25M dipakai sebagai estimasi noise, plus bootstrap confidence interval di data evaluasi. Tambah seed ke-2 untuk perbandingan inti jika budget sisa.
6. **Token 100M fleksibel**: turunkan ke 1–1,5B token jika kuota menipis. Perbandingan tetap adil karena compute-matched.
7. **Data pra-tokenisasi** ke file biner uint16 (vocab 16k < 65.536): 2B token ≈ 4 GB; diunggah sebagai dataset privat Kaggle.
8. **Stabilitas fp16**: loss scaling, gradient clipping, state & normalisasi linear attention di fp32; kriteria hentikan dini bila loss divergen.
9. **Opsional titik tengah**: 2 run di ±50M (konfigurasi terbaik + baseline, ±12 GPU-jam) bila peringkat 25M vs 100M berbeda.
10. **Framing**: low-resource, edge, hardware-realistic; laporkan parameter non-embedding dan rasio memori.

**Referensi pilihan skala** (dipertimbangkan 2026-10-08). Estimasi 1× T4, ±20 token/parameter:

| Skala | Token latih | Waktu di T4 | Status |
|---|---|---|---|
| ~10M | 0,2B | ~10–20 menit | ✅ dipakai |
| ~25M | 0,5B | ~1–2 jam | ✅ dipakai |
| ~50M | 1B | ~4–8 jam | ✅ dipakai (standar, D12) |
| ~100M | 2B | ~17–33 jam | ✅ dipakai sebagai S125 (kelas 125M, opsi A) |
| ~125M | 2,5B | ~26–52 jam | tidak dipakai |
| ~350M | 7B | ~200–400 jam | ❌ menghabiskan seluruh budget |

Catatan umum: klaim arsitektur butuh ≥2–3 skala; 20 token/parameter hanya patokan; laporkan parameter dengan & tanpa embedding; desain iso-compute adalah alternatif yang tidak dipilih.

❌ **Opsi tangga 10M / 25M / 50M**: tidak dipilih (2026-10-08). Lebih murah (±150 GPU-jam, semua run muat 1 sesi) tetapi rentangnya hanya 5× dan skala terbesarnya ±7× di bawah literatur.

---

## 9. Dataset & Tokenizer

| Peran | Dataset | Status |
|---|---|---|
| **Data latih utama** (*pretraining* = proses melatih model **dari nol**, bukan memakai model jadi) | FineWeb-2 `ind_Latn` atau CulturaX `id` (subset 0,5–2B token) | 🟡 pilih setelah cek kualitas |
| Evaluasi held-out | Wikipedia ID (dipisah dari data latih, cek duplikasi) | 🟡 |
| Recall sintetis | MQAR (dibangkitkan sendiri) | 🟡 |
| Downstream ringan | Satu tugas IndoNLU (mis. SmSA sentimen) via fine-tune singkat | 🟡 |
| Referensi eksternal | Goldfish `ind_latn` (≤125M; bandingkan via bpb) | 🟡 |
| Evaluasi pembanding | FLORES-200 `ind_Latn` (dipakai Goldfish → perbandingan langsung) | 🟡 |

**Tokenizer:** SentencePiece (BPE/Unigram + byte-fallback), dilatih sendiri dari nol di sesi CPU Kaggle/TPU-VM (bukan laptop, sesuai D14) dari ~1 GB teks Indonesia. Vocab: **16k default** (D20); ablation 8k / 16k / 32k hanya di S25 (±9 run × 3–5 mnt TPU). 🟡

---

## 10. Desain Eksperimen 🟡

**Prinsip:** baseline dulu; compute-matched; optimizer, LR schedule, dan data identik; ≥3 seed untuk klaim utama; satu perubahan per ablation.

**Grid utama (S10 lalu S25):**
- Rasio: Transformer penuh, 1:1, 3:1, 6:1, pure linear
- KV (untuk model yang punya layer attention): MHA, GQA, MLA, CLA
- Total ±17 konfigurasi × 1 seed; 5 terbaik diulang 3 seed

**Skala S50 & S125:** baseline Transformer (dijalankan awal) + 2–3 konfigurasi terbaik yang konsisten di S10 & S25; 3 seed bila memakai TPU.

**Eksperimen N2:** titik peralihan linear→attention di {10%, 25%, 50%} total langkah vs hybrid dari nol.

**Metrik & pembanding:** lihat §10A dan §10B.

**Alokasi compute (asumsi ±240 GPU-jam T4 selama 8 minggu)** 🔍

| Bagian | Run | GPU-jam (1× T4) | Dengan T4×2 |
|---|---|---|---|
| Uji kernel & kalibrasi | – | ~15–20 | ~15–20 |
| Grid S10 (17 konfigurasi + 5×2 seed) | 27 | ~7 | ~7 |
| Grid S25 (17 konfigurasi + 5×2 seed) | 27 | ~40 | ~40 |
| N2 di S25 | ~10 | ~15 | ~15 |
| S125 (baseline + 2–3 terbaik) | 3–4 | ~70–130 | ~35–65 |
| Cadangan / rerun | – | ~25 | ~25 |
| **Total** | | **~170–240** | **~135–175** |

Ketat jika hanya 1× T4. T4×2 + DDP untuk S125 sangat dianjurkan.

**Anggaran berbasis kuota TPU (20 jam/minggu)** 🟡. Estimasi waktu sudah termasuk kompilasi & evaluasi (🔍 dikalibrasi minggu 1):

| Bagian | Run | TPU-jam |
|---|---|---|
| Uji kelayakan & kalibrasi | – | ~3–5 |
| Grid S10 (17 konfigurasi + 5×2 seed) | 27 | ~1–2 |
| Grid S25 (17 konfigurasi + 5×2 seed) | 27 | ~3–5 |
| N2 di S25 | ~10 | ~1–2 |
| S125: baseline + 2–3 terbaik × 3 seed | 9–12 | ~9–16 |
| S50 (standar): baseline + 2–3 terbaik × 3 seed | 9–12 | ~3–5 |
| Jangkar ±350M: baseline + terbaik (opsional) | 2 | ~16–34 |
| Cadangan / rerun | – | ~10 |
| **Total** | | **~46–81 TPU-jam ≈ 3–4 minggu kuota** |

Tersedia ±160–200 TPU-jam selama 8–10 minggu, jadi margin besar. Kuota GPU tidak lagi menjadi penentu dan cukup untuk cadangan & benchmark.

**Jika TPU v5e-8 dipakai (D10)** 🟡 alokasi di atas menyusut drastis (S125 ±1 jam/run), sehingga budget bisa dipakai untuk:
- 3 seed di S125 (menghapus kekurangan "1 seed");
- titik 50M menjadi standar (D12);
- **opsional jangkar ±350M** (±8–17 jam TPU, butuh 1–2 sesi) yang bertumpang-tindih dengan skala terkecil studi 2507.06457;
- RQ3 diperluas: peringkat efisiensi di **TPU vs GPU vs CPU**.

---

## 10A. Metrik Evaluasi 🟡 (D17)

| Kategori | Metrik | Cara ukur | Data / setting |
|---|---|---|---|
| **Kualitas LM (utama)** | **Bits-per-byte (bpb)** | Total NLL (bit) ÷ total byte UTF-8 teks; adil antar tokenizer | Held-out web (FineWeb-2/CulturaX), Wikipedia ID, **FLORES-200 `ind_Latn` devtest** |
| Kualitas LM (sekunder) | Perplexity per token | Hanya antar model dengan tokenizer sama | Sama seperti di atas |
| **Recall (probe arsitektur)** | Akurasi **MQAR** | Model kecil dilatih langsung pada tugas sintetis; variasikan jumlah pasangan key-value & panjang urutan | Protokol Zoology/Based |
| Recall (model jadi) | Akurasi *passkey/needle* | Sisipkan kode/fakta dalam teks Indonesia panjang, minta model mengulanginya | Konteks 512 / 1k / 2k / 4k (/8k) |
| **Downstream** | Akurasi & macro-F1 | Fine-tune singkat (anggaran sama untuk semua model) | IndoNLU **SmSA** (sentimen); opsional 1 tugas lain |
| **Efisiensi training** | Kurva bpb vs FLOPs; **compute untuk mencapai bpb target**; throughput (token/detik/chip); MFU | FLOPs analitik + pengukuran waktu nyata | Semua run TPU |
| **Efisiensi memori** | Ukuran cache per token (KV + state rekuren); memori puncak training & inferensi | Analitik + terukur | Konteks 512 / 2k / 8k / 32k |
| **Kecepatan inferensi** | Latensi *prefill*; latensi decode (ms/token); throughput (token/detik) | Median dari ≥5 ulangan | **CPU laptop 5500U** (utama, edge), T4, TPU; batch 1 & batch besar |
| **Ukuran model** | Parameter total & non-embedding; ukuran file bobot | Hitung langsung | bf16/fp32 (opsional int8) |
| **Statistik** | Mean ± std (3 seed); bootstrap 95% CI; uji bootstrap berpasangan untuk selisih bpb | – | Semua perbandingan utama |

**Prioritas metrik (D22)**
- **Inti (wajib, masuk paper):** bpb FLORES-200 & bpb web held-out; akurasi MQAR; compute untuk mencapai bpb target (kurva bpb vs FLOPs); KV-cache + memori puncak; latensi decode CPU laptop; SmSA macro-F1 (aplikasi nyata, disukai reviewer jurnal Indonesia).
- **Pelengkap (bila waktu cukup):** bpb Wikipedia ID; passkey 2k–8k; throughput T4 & TPU; MFU; ukuran file int8.
- **Wajib dilaporkan juga:** cek kontaminasi data uji (deduplikasi n-gram antara data latih dan FLORES/Wikipedia/SmSA), serta 3 seed + CI untuk semua angka inti.

Catatan: benchmark zero-shot penalaran (HellaSwag dkk.) tidak dipakai sebagai bukti utama karena hampir acak di skala ≤125M.

## 10B. Pembanding 🟡 (D17)

**Pembanding internal** (dilatih sendiri, data/token/optimizer identik → compute-matched):

| Kode | Model | Fungsi |
|---|---|---|
| B0 | Transformer gaya LLaMA (MHA) | Baseline utama |
| B1 | Transformer + GQA | Baseline reduksi KV yang kuat |
| B2 | Pure Gated DeltaNet (semua layer linear) | Batas bawah memori |
| B3 | Hybrid resep literatur: 3:1 Gated DeltaNet + full attention (gaya Qwen3-Next) | Resep terkini sebagai pembanding |
| B4 (opsional) | Hybrid linear + sliding window (gaya Based) | Varian reduksi KV ekstrem |

**Pembanding eksternal bahasa Indonesia** (model jadi, dievaluasi di data uji & perangkat yang sama; **tidak** compute-matched → dilaporkan sebagai titik rujukan beserta jumlah data latihnya):

| Model | Alasan |
|---|---|
| IndoGPT (±117M) | Backbone identik dengan S125 (768 × 12) |
| cahya/gpt2-small-indonesian-522M (±124M) | GPT-2 small Indonesia; dievaluasi di FLORES/web (bukan Wikipedia) karena data latihnya Wikipedia |
| Goldfish `ind_latn` (≤125M) | LM monolingual kecil modern; dievaluasi di FLORES yang sama |

**Pembanding temuan literatur** (bukan angka langsung, karena bahasa/skala berbeda): rasio optimal 2507.06457 & 2510.04800, trade-off recall Based/Zoology, penghematan KV CLA/Hymba.

---

## 10C. Rencana Perbandingan di Bagian Hasil 🟡 (D18)

Ya, bagian hasil membandingkan model kita dengan model lain dalam **tiga kelompok**. Klaim utama paper hanya didasarkan pada kelompok 1, karena hanya kelompok itu yang adil (compute-matched).

| Kelompok | Siapa dibandingkan | Adil (compute-matched)? | Dipakai untuk |
|---|---|---|---|
| **G1 · Internal** | Model kita vs baseline B0–B3 yang kita latih sendiri, di setiap skala S10–S125 | ✅ Ya: data, token, optimizer, hardware identik | **Klaim utama** (RQ1–RQ5) |
| **G2 · Eksternal Indonesia** | S125 terbaik + B0-S125 vs IndoGPT, GPT-2 Indonesia (cahya), Goldfish `ind_latn` | ❌ Tidak: data, tokenizer & budget berbeda | Titik rujukan: "seberapa dekat model efisien kita dengan LM Indonesia yang ada" |
| **G3 · Literatur** | Tren kita vs temuan 2507.06457, 2510.04800, Based, CLA | ❌ Beda bahasa & skala | Diskusi: apakah temuan literatur berlaku di Indonesia ≤125M |

### Data model yang dibandingkan

| Kode | Model | Kelompok | Arsitektur | Param total | Non-embedding | Vocab | Konteks maks | Data latih | KV-cache / 1.000 token (fp16)* |
|---|---|---|---|---|---|---|---|---|---|
| **OURS** | Hybrid terbaik kita (target: Gated DeltaNet 3:1 + GQA) | – | Hybrid A+B | ~97M | ~85M | 16k | 2048 (uji s.d. 8k) | ~2B token (±8–10 GB) | **~3,1 MB** + state tetap ~0,9 MB 🎯 |
| B0 | Transformer gaya LLaMA (MHA) | G1 | Full attention | ~97M | ~85M | 16k | 2048 | sama dengan OURS | ~36,9 MB |
| B1 | Transformer + GQA (4 KV head) | G1 | Full attention + B | ~95M | ~83M | 16k | 2048 | sama | ~12,3 MB |
| B2 | Pure Gated DeltaNet | G1 | Linear penuh | ~97M | ~85M | 16k | 2048 | sama | 0 MB + state tetap ~1,2 MB |
| B3 | Hybrid 3:1 (gaya Qwen3-Next, MHA) | G1 | Hybrid A | ~97M | ~85M | 16k | 2048 | sama | ~9,2 MB + state tetap ~0,9 MB |
| E1 | **IndoGPT** (IndoNLG, 2021) | G2 | GPT-2 (768×12) | ~117M | ~85M | 40k | 1024 | Indo4B-Plus ±23,8 GB | ~36,9 MB |
| E2 | **cahya/gpt2-small-indonesian-522M** | G2 | GPT-2 small | ~124M | ~85M | 52k | 1024 🔍 | Wikipedia ID 522 MB | ~36,9 MB |
| E3 | **Goldfish `ind_latn` 1000mb** | G2 | GPT-2-style 🔍 | ~125M | 🔍 | 🔍 | 🔍 | ≤1 GB | ~36,9 MB 🔍 |

\* KV-cache dihitung analitik: 2 (K,V) × layer attention × 768 × 2 byte × 1.000 token. Konfigurasi 768×12: MHA = 12 layer attention; hybrid 3:1 = 3 layer; GQA 4/12 head = ⅓. "State tetap" = memori rekuren Gated DeltaNet yang **tidak** tumbuh dengan panjang teks. Baris OURS adalah **target/hipotesis**, bukan hasil. 🔍 = perlu dicek di model card/paper.

### Metrik pembanding per kelompok

| Metrik | G1 Internal | G2 Eksternal | Satuan | Lebih baik |
|---|---|---|---|---|
| bpb FLORES-200 `ind_Latn` devtest | ✅ | ✅ (metrik utama G2) | bit/byte | ↓ |
| bpb held-out web (FineWeb-2/CulturaX) | ✅ | ✅ | bit/byte | ↓ |
| bpb Wikipedia ID | ✅ | ⚠️ kecuali E2 (dilatih di Wikipedia) | bit/byte | ↓ |
| Akurasi MQAR | ✅ | – (tugas sintetis, dilatih sendiri) | % | ↑ |
| Akurasi passkey/needle 1k–8k | ✅ | ✅ (s.d. 1k untuk E1–E3) | % | ↑ |
| SmSA macro-F1 (fine-tune singkat, anggaran sama) | ✅ | ✅ | % | ↑ |
| Compute untuk mencapai bpb target | ✅ | – | FLOPs / TPU-jam | ↓ |
| KV-cache per 1.000 token & memori puncak inferensi | ✅ | ✅ | MB | ↓ |
| Latensi decode CPU laptop 5500U (batch 1) | ✅ | ✅ | ms/token | ↓ |
| Throughput inferensi CPU & T4 | ✅ | ✅ | token/detik | ↑ |
| Parameter total & non-embedding | ✅ | ✅ | juta | ↓ (pada kualitas setara) |

### Grafik data pembanding (dari spesifikasi, bukan hasil eksperimen)

Grafik di bawah memakai Mermaid. Tampil otomatis di GitHub; di VS Code pasang ekstensi *Markdown Preview Mermaid Support*.

**KV-cache per 1.000 token (MB, fp16, analitik)**. OURS = target.

```mermaid
xychart-beta
    title "KV-cache per 1.000 token (MB)"
    x-axis ["B0 MHA", "E1 IndoGPT", "E2 GPT2-Indo", "B1 GQA", "B3 Hybrid 3:1", "OURS (target)", "B2 Pure GDN"]
    y-axis "MB" 0 --> 40
    bar [36.9, 36.9, 36.9, 12.3, 9.2, 3.1, 0]
```

**Parameter total (juta)**. Selisih model kita vs pembanding hampir seluruhnya berasal dari vocab yang lebih kecil; non-embedding semuanya ±85M.

```mermaid
xychart-beta
    title "Parameter total (juta)"
    x-axis ["OURS / B0 (vocab 16k)", "E1 IndoGPT (40k)", "E2 GPT2-Indo (52k)", "E3 Goldfish"]
    y-axis "Juta parameter" 0 --> 140
    bar [97, 117, 124, 125]
```

**Ukuran data latih (GB teks)**. Menunjukkan mengapa G2 bukan perbandingan compute-matched.

```mermaid
xychart-beta
    title "Data latih (GB teks)"
    x-axis ["E1 IndoGPT", "OURS (rencana)", "E3 Goldfish", "E2 GPT2-Indo"]
    y-axis "GB" 0 --> 25
    bar [23.8, 9, 1, 0.52]
```

### Tabel & grafik hasil yang akan diisi (template)

**Tabel R1: Kualitas (S125 + G2)**

| Model | bpb FLORES ↓ | bpb web ↓ | bpb Wiki ↓ | SmSA F1 ↑ | Passkey 1k ↑ |
|---|---|---|---|---|---|
| OURS | – | – | – | – | – |
| B0 | – | – | – | – | – |
| B1 | – | – | – | – | – |
| B2 | – | – | – | – | – |
| B3 | – | – | – | – | – |
| E1 IndoGPT | – | – | – | – | – |
| E2 GPT2-Indo | – | – | – | – | – |
| E3 Goldfish | – | – | – | – | – |

**Tabel R2: Efisiensi (S125 + G2)**

| Model | KV/1k token (MB) ↓ | Memori puncak inferensi 8k (MB) ↓ | Latensi CPU (ms/token) ↓ | Throughput T4 (tok/s) ↑ | TPU-jam ke bpb target ↓ |
|---|---|---|---|---|---|
| OURS | – | – | – | – | – |
| B0 | – | – | – | – | – |
| … | – | – | – | – | – |

**Grafik hasil yang direncanakan** (dibuat dari data eksperimen nanti, disimpan di `figures/`):
1. **bpb vs FLOPs**, per arsitektur, di S10/S25/S50/S125 → efisiensi compute training (RQ1, RQ4)
2. **Frontier Pareto: bpb vs KV-cache** → trade-off kualitas–memori (RQ1, RQ2)
3. **Latensi CPU vs panjang konteks** (512 → 8k) → efisiensi inferensi edge (RQ3)
4. **Akurasi MQAR vs jumlah pasangan key-value** → kemampuan recall
5. **bpb vs skala model** (10M → 125M), per arsitektur → tren skala
6. **Peringkat arsitektur di TPU vs T4 vs CPU** → RQ3

---

## 11. Risiko & Mitigasi

| Risiko | Dampak | Mitigasi |
|---|---|---|
| T4 tidak mendukung bf16 → fp16 tidak stabil untuk linear attention | Training gagal/divergen | Loss scaling, gating/normalisasi, uji minggu 1 |
| Kernel Triton (flash-linear-attention) tidak jalan/lambat di T4 | Perbandingan wall-clock tidak adil | Uji minggu 1; fallback PyTorch chunked; laporkan sebagai temuan RQ3 |
| Kuota GPU gratis berubah/berkurang | Jadwal mundur | Kurangi skala M; prioritaskan grid S |
| Kernel CUDA/Triton (flash-linear-attention, Mamba) tidak jalan di TPU | Linear attention harus ditulis ulang | Implementasi chunkwise di JAX murni (berbasis matmul, cocok untuk TPU); verifikasi vs implementasi rekuren naif di CPU |
| Kurva belajar JAX/XLA (shape statis, kompilasi ulang, debug lebih sulit) | Minggu awal molor | Mulai dari kode JAX minimal (gaya nanoGPT); uji S10 end-to-end di minggu 1; fallback ke T4×2 + PyTorch |
| Antrean/ketersediaan TPU Kaggle & sesi 9 jam | Run tertunda/terputus | Checkpoint rutin; jadwalkan run panjang di luar jam sibuk; T4×2 sebagai cadangan |
| Mencampur hasil TPU (bf16) & GPU (fp16) | Perbandingan tidak adil | Satu studi kualitas = satu hardware; GPU dipakai untuk benchmark efisiensi atau seluruh studi diulang di GPU bila fallback |
| N2 ternyata sudah ada di literatur | Novelty metode hilang | Validasi minggu 1–2; ganti dengan varian lain atau fokus N1 |
| Paper "Training-time Selection of Linear vs Softmax Attention" ternyata sama dengan N2 | Novelty N2 melemah | Baca di minggu 1; bila mirip, posisikan N2 sebagai varian (mis. fokus compute training & bahasa Indonesia) atau beralih ke kontribusi N1 |
| Kurva belajar JAX untuk Ken | Minggu 1–2 lebih lambat | Mulai dari contoh minimal; Claude menyiapkan kerangka kode & penjelasan per bagian |
| Model kecil terlalu lemah untuk benchmark downstream | Hasil tidak bermakna | Andalkan bpb + MQAR; downstream hanya pelengkap |
| Laptop throttling saat benchmark CPU | Angka latensi bias | Ulangi pengukuran, catat suhu/daya, pakai median |

---

## 12. Timeline (10–12 minggu)

| Minggu | Kegiatan | Keluaran |
|---|---|---|
| 1–2 | Belajar dasar JAX; studi literatur (validasi gap & N2, baca paper Training-time Selection); setup repo, tokenizer, data; **uji kelayakan TPU** (S10 end-to-end di JAX, ukur throughput); siapkan pendaftaran TRC | Daftar pustaka, tokenizer, data siap, keputusan hardware final |
| 3–4 | Baseline Transformer & pure linear (S10, S25), MQAR, kalibrasi throughput; **mulai baseline S125 (minggu 4–5)** | Kurva baseline |
| 5–7 | Grid utama rasio × KV (S10 → S25); eksperimen N2 | Tabel hasil utama |
| 8–9 | S50 & S125 untuk 2–3 konfigurasi terbaik; ablation; seed tambahan; benchmark CPU | Hasil final |
| 10–12 | Penulisan paper, rilis kode/model, arXiv, submit jurnal | Preprint + submission |

Catatan: dalam 3 bulan target realistis adalah preprint + submit jurnal, karena review jurnal biasanya berbulan-bulan. arXiv kemungkinan perlu *endorsement* untuk pengunggah pertama. 🔍

---

## 13. Pertanyaan Terbuka ❓
1. Setuju usulan D20 (vocab 16k + ablation 8k/16k/32k di S25)?
2. Setuju usulan D21 (350M sebagai stretch goal bersyarat)?
3. Setuju usulan D22 (metrik inti vs pelengkap)?
4. Daftar calon jurnal Scopus/SINTA: perlu diverifikasi indeks & scope-nya sebelum menulis (minggu 9–10).

---

## 14. Ide Cadangan (belum dipakai)
- C: weight sharing / recurrent depth (efisiensi parameter)
- D: adaptive compute per token (Mixture-of-Depths, early exit)
- E: parameterisasi tensor-network / quantum-inspired (terkait latar quantum ML Ken; harus dibandingkan ketat dengan low-rank biasa)

---

## 15. Log Diskusi
Format: tanggal — topik — hasil — keputusan — langkah berikutnya

- **2026-10-08 · Diskusi #1**: Ken ingin LM efisien dengan budget minim. Dibahas istilah penelitian, estimasi compute, kandidat dataset, metrik, dan 5 arah arsitektur (A–E).
- **2026-10-08 · Diskusi #2**: Ken menetapkan D2–D6. Studi literatur awal menemukan A+B sudah dipakai di model besar → novelty dialihkan ke N1 (Indonesia + <100M + hardware murah) dan N2 (*linear-first warm-start*). Laptop 5500U dinilai ~100× lebih lambat dari T4.
- **2026-10-08 · Diskusi #3**: Dokumen konsep dijadikan file hidup `KONSEP.md` (v0.2). Ditambahkan hipotesis H1–H5, rancangan ukuran model, alokasi compute, dan tabel risiko. → Berikutnya: Ken menjawab §13.
- **2026-10-08 · Diskusi #4**: Klarifikasi tabel compute: 25M = 25 juta parameter; 0,5B = 500 juta token data latih (±2–2,5 GB teks Indonesia); tabel itu untuk TRAINING. Ditambahkan estimasi inferensi: model 25M–100M jalan lancar di laptop CPU.
- **2026-10-08 · Diskusi #5**: Ken bertanya apakah wajib 25M. Jawab: tidak, ukuran adalah pilihan desain. Ditambahkan tabel pilihan skala (10M–350M) di §8; 350M tidak layak dengan budget. Usulan: tangga 3 skala (mis. 10M/25M/100M). Pertanyaan §13 no. 6 ditambahkan.
- **2026-10-08 · Diskusi #6**: Ken mempertimbangkan tangga 10M/25M/50M. Dianalisis kelebihan (murah ±150 GPU-jam, muat 1 sesi Kaggle, banyak seed) dan kekurangan (rentang sempit, jauh dari 340M di literatur, risiko konfigurasi terbaik bergeser, model lemah). Usulan: 10/25/50 sebagai inti + opsional run jangkar ±100M.
- **2026-10-08 · Diskusi #7**: Ken memutuskan tangga **10M / 25M / 100M** (D9 ✅). Dianalisis kelebihan (rentang 10×, lebih dekat literatur, KV-cache bermakna, demo layak), kekurangan (run 100M > sesi 12 jam, 1 seed, tanpa titik tengah, makan kuota, fp16), dan 10 mitigasi (baseline 100M dijalankan awal, T4×2+DDP, checkpoint, dll.). §8, §10, §12 diperbarui.
- **2026-10-08 · Diskusi #8**: Ken menunjukkan Kaggle menyediakan TPU v5e-8 selain T4×2. Dianalisis: TPU ±10–25× lebih cepat dari T4×2 (bf16 native → masalah fp16 hilang), tapi kernel Triton/CUDA tidak jalan sehingga linear attention perlu ditulis di JAX. Usulan D10 (TPU utama jika lolos uji minggu 1) & D11 (budget dalam FLOPs). Dampak: S125 tiga seed, titik 50M murah, opsional jangkar 350M.
- **2026-10-08 · Diskusi #9**: Ken membagikan kuota Kaggle: **TPU 20 jam/minggu** (utuh), GPU 30 jam/minggu (tersisa ±7 jam minggu ini). Dibuat anggaran TPU: seluruh rencana (termasuk S125 3 seed & opsional jangkar 350M) ±45–80 TPU-jam ≈ 3–4 minggu kuota. D10 makin layak, masih menunggu uji kelayakan & keputusan framework.
- **2026-10-08 · Diskusi #10**: Ken bertanya 100M atau 125M. Usulan D12: 'kelas 125M' = backbone GPT-2 small (768×12, ±85M non-embedding); total tergantung vocab (±97M di 16k). Alasan: ukuran rujukan umum termasuk Goldfish ind_latn (≤125M, eval FLORES). Biaya tambahan di TPU kecil. Titik 50M diusulkan jadi standar. FLORES-200 ind_Latn ditambahkan ke evaluasi.
- **2026-10-08 · Diskusi #11**: Ken memilih **opsi A** (768×12, backbone GPT-2 small, ±97M total) dan **50M sebagai standar** → D12 ✅, tangga final 10M / 25M / 50M / kelas 125M (S10/S25/S50/S125). Tabel konfigurasi, alokasi TPU, gap, RQ, dan timeline disesuaikan.
- **2026-10-09 · Diskusi #12**: Ken menyetujui N1+N2 (D8 ✅), fokus TPU (D10 ✅), belajar JAX sambil jalan (D13 ✅), laptop untuk pengujian inferensi (D14 ✅), TRC sebagai opsi (D16). Pertanyaan S10/S25 di GPU + S125 di TPU: bisa tapi tidak disarankan (tren skala tercampur efek hardware, dua jalur kode) → usulan D15. Ditambahkan penjelasan arah A/B, rasio, vocab & non-embedding; studi setara (2510.04800, Based, Zoology, CLA, Hymba, HALO) & pembanding Indonesia (IndoGPT 117M backbone sama dengan S125, GPT-2 Indonesia, Goldfish); §10A metrik & §10B pembanding. Folder dijadikan repo git lokal.
- **2026-10-09 · Diskusi #13**: Dibuat folder `papers/` (wajib/penting/pendukung) dengan `DAFTAR-BACA.md` (19 paper, prioritas & fokus baca) dan skrip `unduh_paper.ps1`.
- **2026-10-09 · Diskusi #14**: Ditambahkan 9 paper model/benchmark Indonesia (IndoNLG, Goldfish, IndoNLU, IndoLEM, Cendol, Sailor, SEA-LION, IndoMMLU, NusaCrowd) ke `papers/DAFTAR-BACA.md` (folder `papers/indonesia/`). PDF W2, W3, I1 sudah diunduh; sisanya lewat `unduh_paper.ps1`.
- **2026-10-09 · Diskusi #15**: Klarifikasi istilah: "pretraining" di §9 = proses melatih model kita dari nol, bukan memakai model jadi. Ditambahkan ke glosarium; tokenizer dilatih di CPU Kaggle/TPU-VM, bukan laptop.
- **2026-10-09 · Diskusi #16**: Ditambahkan §10C rencana perbandingan hasil: 3 kelompok (G1 internal compute-matched, G2 IndoGPT/GPT-2 Indo/Goldfish, G3 literatur), tabel data 8 model, metrik per kelompok, 3 grafik Mermaid (KV-cache, parameter, data latih), template tabel hasil R1–R2 & daftar 6 grafik hasil.
- **2026-10-09 · Diskusi #17**: Ken menyetujui D15 (semua training di TPU) dan target jurnal Scopus & SINTA (D19). Rekomendasi: vocab 16k + ablation di S25 (D20), 350M sebagai stretch goal bersyarat (D21), metrik inti vs pelengkap + cek kontaminasi (D22).

---

## 16. Referensi
Daftar baca berprioritas: `papers/DAFTAR-BACA.md`.

- A Systematic Analysis of Hybrid Linear Attention: https://arxiv.org/abs/2507.06457
- Gated DeltaNet / Parallelizing Linear Transformers with the Delta Rule (NeurIPS 2024)
- LLM Architecture Gallery: Hybrid Attention (S. Raschka): https://sebastianraschka.com/llm-architecture-gallery/hybrid-attention/
- Cendol: https://arxiv.org/abs/2404.06138
- IndoLEM & IndoBERT: https://arxiv.org/abs/2011.00677
- Goldfish: Monolingual Language Models for 350 Languages: https://arxiv.org/abs/2408.10441 · model: https://huggingface.co/goldfish-models
- FineWeb-2: https://huggingface.co/datasets/HuggingFaceFW/fineweb-2
- Free Cloud GPUs in 2026 (kuota Kaggle/Colab): https://gpuperhour.com/blog/free-cloud-gpus-and-credits
- Hybrid Architectures for Language Models: Systematic Analysis and Design Insights: https://arxiv.org/abs/2510.04800
- Based: Simple linear attention language models balance the recall-throughput tradeoff: https://arxiv.org/abs/2402.18668
- Zoology (MQAR): https://hazyresearch.stanford.edu/blog/2023-12-11-zoology2-based
- Reducing Transformer Key-Value Cache Size with Cross-Layer Attention: https://arxiv.org/abs/2405.12981
- Hymba: A Hybrid-head Architecture for Small Language Models: https://arxiv.org/abs/2411.13676
- Hybrid Linear Attention Done Right (HALO/HypeNet): https://arxiv.org/abs/2601.22156
- Training-time Selection of Linear Vs. Softmax Attention in Layer-based Hybrid Transformers: https://openreview.net/forum?id=4cP69sGjiG
- IndoNLG / IndoGPT: https://arxiv.org/abs/2104.08200
- cahya/gpt2-small-indonesian-522M: https://huggingface.co/cahya/gpt2-small-indonesian-522M
- w11wo/indo-gpt2-small: https://huggingface.co/w11wo/indo-gpt2-small
- Google Cloud TPU v5e: https://docs.cloud.google.com/tpu/docs/v5e
- IndoNLU: https://arxiv.org/abs/2009.05387
- Sailor: https://arxiv.org/abs/2404.03608
- SEA-LION: https://arxiv.org/abs/2504.05747
- IndoMMLU: https://arxiv.org/abs/2310.04928
- NusaCrowd: https://arxiv.org/abs/2212.09648
