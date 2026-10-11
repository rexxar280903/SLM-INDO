"""Tes memeriksa matematika: matmul fp32 dipaksa presisi penuh (penting bila tes dijalankan di TPU)."""
import jax

jax.config.update("jax_default_matmul_precision", "float32")
