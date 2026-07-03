from std.sys import (
    has_nvidia_gpu_accelerator,
    has_apple_gpu_accelerator,
    is_defined,
)

# ===----------------------------------------------------------------------=== #
# Vendor GPU dispatch — flag glossary
# ===----------------------------------------------------------------------=== #
#
# HAS_CUBLAS — True on NVIDIA: enables the cuBLAS/cuBLASLt FFI fast path
#   (matmul.mojo / attention.mojo) and inline PTX ex2.approx
#   (fused_classifier.mojo). On Apple GPU this is always False; those
#   CUDA-specific code paths are excluded entirely at compile time.
#
#   Auto-detected via has_nvidia_gpu_accelerator(). Override to force the
#   vendor-neutral fallback on NVIDIA hardware (e.g. portability CI):
#     mojo build -D LLMM_FORCE_PORTABLE_GPU=1 ...
#   This flag has no effect on the Metal path (HAS_METAL is unaffected).
#
comptime _FORCE_PORTABLE_GPU = is_defined["LLMM_FORCE_PORTABLE_GPU"]()
comptime HAS_CUBLAS = has_nvidia_gpu_accelerator() and not _FORCE_PORTABLE_GPU

# HAS_METAL — True on Apple Silicon: enables the Metal device path, which
#   uses linalg.matmul for GEMM and the standalone bias_gelu_fwd epilogue
#   (because linalg's fused elementwise_lambda_fn is broken on Metal —
#   see gelu.mojo). On NVIDIA this is always False; those Metal-specific
#   branches are excluded at compile time.
#
#   Auto-detected via has_apple_gpu_accelerator(). Override to force the
#   CPU fallback on Apple Silicon (e.g. for debugging or CI without GPU):
#     mojo build -D LLMM_DISABLE_METAL=1 ...
#   This flag has no effect on the NVIDIA path (HAS_CUBLAS is unaffected).
#
comptime _DISABLE_METAL = is_defined["LLMM_DISABLE_METAL"]()
comptime HAS_METAL = has_apple_gpu_accelerator() and not _DISABLE_METAL
