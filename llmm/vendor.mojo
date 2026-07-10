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

# USE_TF32 — True by default: fp32 cuBLAS(Lt) GEMMs (matmul.mojo's
#   _matmul_cublaslt and attention.mojo's _attn_gemm_batched cuBLAS tail) route
#   through TF32 tensor cores (ComputeType.COMPUTE_32F_FAST_TF32) instead of
#   plain FP32 CUDA cores (ComputeType.COMPUTE_32F). This mirrors llm.c's fp32
#   arm (train_gpt2_fp32.cu:1614-1618), which auto-enables
#   CUBLAS_COMPUTE_32F_FAST_TF32 on any compute-capability-8.0+ GPU — i.e.
#   llm.c's "fp32" is already TF32-vs-TF32 by definition on Ampere+/Blackwell.
#   bf16/fp16 builds are unaffected either way (input dtype alone already
#   selects tensor cores for those).
#
#   Disable to fall back to true IEEE fp32 math (no tensor cores) for
#   numerical debugging, e.g. to isolate TF32 rounding from a suspected
#   correctness bug:
#     mojo build -D LLMM_NO_TF32=1 ...
#
comptime _DISABLE_TF32 = is_defined["LLMM_NO_TF32"]()
comptime USE_TF32 = not _DISABLE_TF32
