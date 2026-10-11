# SLM-INDO

Riset mandiri arsitektur **small language model (SLM) efisien untuk bahasa Indonesia**. Pendekatannya menggabungkan hybrid Gated DeltaNet + full attention, reduksi KV-cache, dan metode usulan *linear-first warm-start* (N2). Implementasinya ditulis dalam **JAX murni + Optax** untuk Kaggle TPU v5e-8.

- Konsep, keputusan, dan rencana eksperimen: [`KONSEP.md`](KONSEP.md)
- Draf paper: [`manuscript/paper-slm-indonesia.html`](manuscript/paper-slm-indonesia.html)
- Peta alur & arsitektur interaktif: [`visual/peta-alur-arsitektur.html`](visual/peta-alur-arsitektur.html)
- Panduan membaca kode JAX: [`notes/panduan-kode.md`](notes/panduan-kode.md)

> **Status:** kode, data pipeline, dan tes sudah siap. **Belum ada hasil eksperimen.** Langkah berikutnya adalah uji kelayakan TPU, lalu pilot B0-S10.

## Struktur

```
src/slm/            kode inti (config, layers, gdn, model, flops, data, train, evaluate, n2, mqar, checkpoint, dataprep)
scripts/data/       00 set evaluasi → 01 sampel FineWeb-2 → 02 audit → 03 tokenizer → 04 tokenisasi
scripts/            arch_report.py (parameter/FLOPs/cache), check_flops.py (validasi XLA), tpu_check.py
tests/              unit test (pytest): chunkwise = rekuren, kausalitas, jumlah parameter, N2, resume, …
notes/              audit data, laporan arsitektur (arch_report.json), panduan kode
```

## Menjalankan di laptop (tes saja)

```bash
python -m venv .venv && .venv/Scripts/pip install "jax[cpu]" optax numpy pyyaml pytest sentencepiece pyarrow huggingface_hub
.venv/Scripts/python -m pytest -q
PYTHONPATH=src .venv/Scripts/python -m slm.train --arch b3 --scale mini --synthetic --max-steps 30
```

## Kaggle · Notebook A: data & tokenizer (CPU, Internet ON)

```bash
git clone https://github.com/rexxar280903/SLM-INDO.git && cd SLM-INDO
pip install -q sentencepiece pyarrow huggingface_hub
python scripts/data/00_eval_sets.py --out /kaggle/working/eval
python scripts/data/02_audit.py --row-groups 3 --out /kaggle/working/audit.md
python scripts/data/01_sample_fineweb2.py --eval-dir /kaggle/working/eval --out /kaggle/working/clean --target-gb 14 --workers 4
python scripts/data/03_train_tokenizer.py --clean /kaggle/working/clean --out /kaggle/working/tok16k --vocab 16384
python scripts/data/04_tokenize.py --clean /kaggle/working/clean --eval-dir /kaggle/working/eval \
    --tok /kaggle/working/tok16k --out /kaggle/working/tokens16k --workers 4
```

Setelah selesai, klik *Save Version*, lalu jadikan output notebook sebagai dua Kaggle Dataset privat:
- `slm-indo-tokens16k`: berisi `tokens16k/` dan `eval/`
- `slm-indo-clean`: berisi `clean/`, dipakai untuk ablation vocab 8k/32k

Catatan: jika pelatihan tokenizer kehabisan RAM, turunkan `--sample-gb` ke 0,5. Tokenizer ablation dibuat dengan `--vocab 8192` atau `--vocab 32768` dari `clean/` yang sama.

## Kaggle · Notebook B: training (TPU v5e-8)

```bash
git clone https://github.com/rexxar280903/SLM-INDO.git && cd SLM-INDO
pip install -q optax
python -c "import jax; print(jax.devices())"                       # harus 8 perangkat TPU
python -m pytest -q tests/test_gdn.py                              # matematika GDN di TPU
python scripts/tpu_check.py --out /kaggle/working/tpu_check.json   # Keputusan B: kelayakan TPU
PYTHONPATH=src python -m slm.train --arch b0 --scale s10 --seed 0 \
    --data-dir /kaggle/input/slm-indo-tokens16k/tokens16k --out-dir /kaggle/working/runs
PYTHONPATH=src python -m slm.evaluate --run-dir /kaggle/working/runs/b0-s10-seed0-flops \
    --data-dir /kaggle/input/slm-indo-tokens16k/tokens16k --eval-dir /kaggle/input/slm-indo-tokens16k/eval \
    --sets val flores_dev
```

Jika repo bersifat privat, unggah folder ini sebagai Kaggle Dataset lalu jalankan dari `/kaggle/input/...` sebagai pengganti `git clone`.

Opsi penting `slm.train`:
- `--arch`: `b0`–`b4` atau grid `full|h1|h3|h6`-`mha|gqa|mla|cla`, serta `linear`
- `--budget-mode tokens`: kontrol token-matched
- `--lr-mult 0.5|2`: sweep LR
- `--n2-tau 0.25 --n2-target h3-mha`: N2
- `--task mqar`: probe recall
- `--set attn_impl=xla gdn_precision=default remat=false`: override implementasi

## Lisensi data

FineWeb-2 dirilis di bawah ODC-By 1.0, dengan revisi yang dipatok di skrip. Data mentah dan token tidak disimpan di git (lihat `.gitignore`).
