from std.sys import has_nvidia_gpu_accelerator, is_defined

# ===----------------------------------------------------------------------=== #
# Vendor GPU dispatch
# ===----------------------------------------------------------------------=== #
#
# Single source of truth for whether the NVIDIA-only fast paths (hand-rolled
# cuBLAS/cuBLASLt FFI in matmul.mojo/attention.mojo, inline PTX ex2.approx in
# fused_classifier.mojo) are compiled in. Every GPU dispatch that currently
# assumes "GPU == NVIDIA" should branch on HAS_CUBLAS, nested inside the
# existing `is_gpu[target]()` check, so CPU dispatch and (when HAS_CUBLAS is
# True) the NVIDIA GPU dispatch are completely unaffected.
#
# Defaults to auto-detection (has_nvidia_gpu_accelerator()), so existing
# NVIDIA builds need zero flag changes. Overridable for portability
# testing/CI on NVIDIA hardware, or to force the vendor-neutral fallback if
# the auto-detected accelerator is wrong for some reason:
#   mojo build -D LLMM_FORCE_PORTABLE_GPU=1 ...
comptime _FORCE_PORTABLE_GPU = is_defined["LLMM_FORCE_PORTABLE_GPU"]()
comptime HAS_CUBLAS = has_nvidia_gpu_accelerator() and not _FORCE_PORTABLE_GPU
