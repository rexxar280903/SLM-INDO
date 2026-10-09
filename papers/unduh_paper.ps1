# Unduh PDF daftar bacaan. Jalankan dari folder papers:
#   powershell -ExecutionPolicy Bypass -File .\unduh_paper.ps1
$papers = @(
  @{d="wajib";     f="W1_training-time-selection-linear-vs-softmax.pdf"; u="https://openreview.net/pdf?id=4cP69sGjiG"},
  @{d="wajib";     f="W2_2507.06457_systematic-hybrid-linear-attention.pdf"; u="https://arxiv.org/pdf/2507.06457"},
  @{d="wajib";     f="W3_2412.06464_gated-deltanet.pdf"; u="https://arxiv.org/pdf/2412.06464"},
  @{d="wajib";     f="W4_2510.04800_hybrid-architectures-meta.pdf"; u="https://arxiv.org/pdf/2510.04800"},
  @{d="wajib";     f="W5_2104.08200_indonlg-indogpt.pdf"; u="https://arxiv.org/pdf/2104.08200"},
  @{d="penting";   f="P1_2406.06484_deltanet-parallel.pdf"; u="https://arxiv.org/pdf/2406.06484"},
  @{d="penting";   f="P2_2312.04927_zoology-mqar.pdf"; u="https://arxiv.org/pdf/2312.04927"},
  @{d="penting";   f="P3_2402.18668_based.pdf"; u="https://arxiv.org/pdf/2402.18668"},
  @{d="penting";   f="P4_2405.12981_cross-layer-attention.pdf"; u="https://arxiv.org/pdf/2405.12981"},
  @{d="penting";   f="P5_2411.13676_hymba.pdf"; u="https://arxiv.org/pdf/2411.13676"},
  @{d="penting";   f="P6_2601.22156_halo-hypenet.pdf"; u="https://arxiv.org/pdf/2601.22156"},
  @{d="penting";   f="P7_2408.10441_goldfish.pdf"; u="https://arxiv.org/pdf/2408.10441"},
  @{d="pendukung"; f="S1_2305.13245_gqa.pdf"; u="https://arxiv.org/pdf/2305.13245"},
  @{d="pendukung"; f="S2_2405.04434_deepseek-v2-mla.pdf"; u="https://arxiv.org/pdf/2405.04434"},
  @{d="pendukung"; f="S3_2510.26692_kimi-linear.pdf"; u="https://arxiv.org/pdf/2510.26692"},
  @{d="pendukung"; f="S4_2405.21060_mamba2.pdf"; u="https://arxiv.org/pdf/2405.21060"},
  @{d="pendukung"; f="S5_2203.15556_chinchilla.pdf"; u="https://arxiv.org/pdf/2203.15556"},
  @{d="pendukung"; f="S6_2011.00677_indolem-indobert.pdf"; u="https://arxiv.org/pdf/2011.00677"},
  @{d="pendukung"; f="S7_2404.06138_cendol.pdf"; u="https://arxiv.org/pdf/2404.06138"}
)
foreach ($p in $papers) {
  $out = Join-Path $PSScriptRoot (Join-Path $p.d $p.f)
  if (Test-Path $out) { Write-Host "Sudah ada: $($p.f)"; continue }
  try {
    Invoke-WebRequest -Uri $p.u -OutFile $out -UserAgent "Mozilla/5.0"
    Write-Host "OK   : $($p.f)"
  } catch { Write-Host "GAGAL: $($p.f) -> buka manual: $($p.u)" }
  Start-Sleep -Seconds 3
}
