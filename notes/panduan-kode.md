# Panduan membaca kode SLM-INDO (JAX untuk pemula)

Dokumen ini menjelaskan kode di `src/slm/` langkah demi langkah. Urutan baca yang disarankan: konsep JAX → `config.py` → `layers.py` → `gdn.py` → `model.py` → `train.py`.

## 1. Lima konsep JAX yang dipakai

| Konsep | Arti | Contoh di kode |
|---|---|---|
| **Pytree** | Struktur bersarang dari dict, list, atau tuple yang berisi array. Parameter model kita adalah dict biasa, bukan objek kelas. | `params = {"embed": …, "layers": [ {...}, … ], "norm_f": …}` di `model.py` |
| **Fungsi murni** | Output hanya bergantung pada input, tanpa state tersembunyi. Syarat ini yang membuat `jit` dan `grad` bisa bekerja. | `rms_norm(x, weight)`, `swiglu_ffn(p, x)` |
| **`jax.jit`** | Mengompilasi fungsi Python menjadi program XLA. Panggilan pertama lambat karena ada kompilasi; panggilan berikutnya cepat. Jika bentuk input berubah, fungsi dikompilasi ulang. | `make_train_step` di `train.py` |
| **`jax.grad` / `value_and_grad`** | Menurunkan fungsi loss terhadap parameter secara otomatis, sehingga tidak perlu menulis backward pass sendiri. | `grad_fn = jax.value_and_grad(loss_fn, has_aux=True)` |
| **`jax.lax.scan`** | Loop yang dikompilasi sekali sebagai satu operasi. Dipakai untuk mengulang hal yang sama: token, chunk, atau micro-batch. | state GDN antar-chunk (`gdn.py`), akumulasi gradien (`train.py`) |

Tiga konsep tambahan:
- **Kunci acak (`jax.random.PRNGKey`)**: JAX tidak punya RNG global. Setiap pengambilan acak memakai kunci eksplisit, dan kunci dipecah dengan `jax.random.split`. Akibatnya seed menentukan bobot awal secara penuh.
- **`jax.checkpoint` (remat)**: aktivasi di dalam satu blok tidak disimpan untuk backward, melainkan dihitung ulang. Memori turun drastis, sedangkan compute naik sekitar ⅓. Opsi ini aktif secara default (`remat=True`).
- **Sharding**: `Mesh(devices, ("data",))` menamai 8 chip TPU sebagai sumbu `data`. Batch diberi `NamedSharding(mesh, P(None, "data"))`, artinya batch dibagi ke 8 chip, sedangkan parameter direplikasi. XLA menambahkan *all-reduce* gradien secara otomatis. Inilah data parallel tanpa kode komunikasi manual.

## 2. Peta file

| File | Isi | Dibaca bersama |
|---|---|---|
| `config.py` | Skala S10–S125 (+mini/tiny untuk tes), parser nama arsitektur, pola hybrid, `TrainConfig`, jadwal LR | KONSEP §8, §10D |
| `layers.py` | RMSNorm, RoPE, SwiGLU, attention (MHA/GQA/CLA/sliding window/MLA), konvolusi kausal | paper Pers. 2, 4 |
| `gdn.py` | Gated DeltaNet: bentuk rekuren (acuan) dan chunkwise (training) | paper Pers. 3 |
| `model.py` | Inisialisasi parameter, forward hidden, cross-entropy berpotongan | — |
| `flops.py` | Jumlah parameter, FLOPs per token, memori KV/state/conv | paper Pers. 5–7 |
| `data.py` | Shard uint16, urutan batch deterministik, tabel byte per token | KONSEP D33 |
| `train.py` | Rencana budget, langkah training, evaluasi bpb, checkpoint, N2, MQAR, CLI | dashboard bagian 3 |
| `n2.py` | Konversi GDN → attention beserta momen AdamW | paper §3.5 |
| `mqar.py` | Generator tugas MQAR | paper §4.3 |
| `evaluate.py` | bpb akhir pada val/test/FLORES/Wikipedia | paper Pers. 8 |
| `dataprep.py` | Filter spam judi, hash, split, indeks n-gram dekontaminasi | KONSEP §9 |

## 3. Alur satu langkah training (`train.py`)

1. `plan_run` menghitung budget FLOPs. Rumusnya: FLOPs/token B0 × token referensi skala. Dari budget ini, jumlah langkah tiap arsitektur ditentukan sehingga model yang lebih murah per token mendapat lebih banyak langkah (*compute-matched*).
2. `lr[step]` diambil dari jadwal yang dihitung terhadap progres FLOPs: warmup 3% (minimal 50 langkah), lalu cosine turun ke 10%.
3. `train_data.batch(step)` mengembalikan array `(B, T+1)` yang hanya bergantung pada `(seed, step)`. Array ini dibentuk ulang menjadi `(akumulasi, micro, T+1)`.
4. Di dalam `train_step` yang di-jit:
   - `lax.scan` menjalankan setiap micro-batch: hitung loss dan gradien, lalu jumlahkan.
   - Gradien dirata-rata, kemudian `clip_by_global_norm(1.0)` → `scale_by_adam` → `add_decayed_weights(0.1)`.
   - Parameter diperbarui dengan `p − lr·u`.
5. Log ditulis setiap N langkah ke `metrics.jsonl`. Evaluasi bpb dijalankan pada 10%, 20%, … budget dan dicatat di `eval.jsonl`. Checkpoint disimpan setiap 30 menit.

## 4. Gated DeltaNet chunkwise dalam satu paragraf

Bentuk rekuren memperbarui memori `S` (64×64 per head) token demi token. Ini benar tetapi berurutan, sehingga lambat di TPU. Bentuk chunkwise memotong urutan menjadi chunk 64 token. Di dalam chunk, interaksi antartoken dihitung dengan perkalian matriks: `KKᵀ`, solve segitiga `(I+A)⁻¹`, `QKᵀ`. Di antara chunk, hanya `S` yang diteruskan, sehingga cukup 2048/64 = 32 langkah `scan`. Derivasinya ada di docstring `gdn.py`. `tests/test_gdn.py` membuktikan bahwa output dan gradien kedua bentuk sama, untuk berbagai panjang dan ukuran chunk.

## 5. Menambah varian arsitektur

1. Tambahkan nama varian di `config.KV_VARIANTS` atau `MIXES`, lalu atur parameternya di `build_model_config`.
2. Tambahkan parameter di `model.init_attention`/`init_gdn` dan forward-nya di `layers.py`/`gdn.py`.
3. Perbarui `flops.mixer_params`, `mixer_flops_per_token`, dan `inference_memory`. Tes `test_param_count_matches_analytic` akan gagal bila rumus tidak cocok dengan bentuk tensor.
4. Jalankan `pytest` dan `python scripts/arch_report.py`.

## 6. Hal yang sengaja belum dibuat

- Decode token demi token dengan KV-cache/state (untuk benchmark latensi CPU). Ini dikerjakan setelah ada checkpoint, di fase evaluasi.
- Evaluasi model eksternal (IndoGPT, GPT-2 Indonesia, Goldfish) dan fine-tune SmSA. Keduanya butuh PyTorch/transformers dan juga dikerjakan di fase evaluasi.
