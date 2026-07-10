// probe_fp4.cu — minimal cuBLASLt NVFP4 (CUDA_R_4F_E2M1, block-scaled vs16)
// dispatch probe for GB10 (sm_121). See docs/ai/fp4_modular_support_research.md
// §5 for the question this answers: does cuBLASLt's sm_120 NVFP4 block-scaled
// GEMM cubin dispatch on this sm_121 device?
//
// Deliberately links against the *pixi-pinned* cudart/cublasLt
// (12.9.2.10 — the version audited in the research doc), not the system
// CUDA 13.0 toolkit, via -L/-Wl,-rpath in the Makefile. Headers come from
// the system CUDA 13.0 install (no headers ship in the pixi env), which is
// ABI-compatible for the enums/structs used here (CUDA_R_4F_E2M1,
// CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3, block-scale descriptor
// attributes were all already present verbatim in the pixi .so, confirmed
// separately via `nm`/`strings` in the research doc).
//
// Scale-factor swizzle ("128x4 tile, 32x4x4 internal" layout) implemented
// per cuBLAS docs §3.1.4.3.2, cross-checked against PyTorch's reference
// `to_blocked()` (torch/testing/_internal/common_quantized.py).

#include <cublasLt.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

// ---------------------------------------------------------------------------
// Status helpers
// ---------------------------------------------------------------------------

static const char* cublas_status_str(cublasStatus_t s) {
    switch (s) {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
        default: return "UNKNOWN_CUBLAS_STATUS";
    }
}

#define CUDA_CHECK(expr)                                                          \
    do {                                                                          \
        cudaError_t _e = (expr);                                                  \
        if (_e != cudaSuccess) {                                                  \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,         \
                    cudaGetErrorString(_e));                                      \
            std::exit(1);                                                        \
        }                                                                         \
    } while (0)

// Returns the status rather than aborting, so callers can report+branch on it
// (the whole point of this probe is to observe NOT_SUPPORTED/ARCH_MISMATCH).
#define CUBLAS_TRY(expr, out_status)                                              \
    do {                                                                          \
        (out_status) = (expr);                                                    \
    } while (0)

// ---------------------------------------------------------------------------
// e2m1 (NVFP4 data) encode/decode — OCP MX E2M1 magnitude table
// ---------------------------------------------------------------------------

static const float E2M1_MAGS[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

static uint8_t encode_e2m1(float x) {
    bool sign = x < 0.0f;
    float ax = fabsf(x);
    int best = 0;
    float best_d = fabsf(ax - E2M1_MAGS[0]);
    for (int i = 1; i < 8; i++) {
        float d = fabsf(ax - E2M1_MAGS[i]);
        if (d < best_d) { best_d = d; best = i; }
    }
    return (uint8_t)((sign ? 8 : 0) | best);
}

static float decode_e2m1(uint8_t code) {
    bool sign = (code >> 3) & 1;
    float mag = E2M1_MAGS[code & 7];
    return sign ? -mag : mag;
}

// Pack two e2m1 nibbles into one byte. PyTorch's pack_uint4 convention
// (common_quantized.py): even index -> low nibble, odd index -> high nibble,
// contiguous along the packed (K) dimension.
static uint8_t pack_e2m1x2(uint8_t lo, uint8_t hi) {
    return (uint8_t)((hi << 4) | (lo & 0xF));
}

// ---------------------------------------------------------------------------
// e4m3 (block scale) encode — standard FP8 E4M3 (sign always 0 here; the
// "UE4M3" cuBLASLt scale dtype is byte-identical to CUDA_R_8F_E4M3, per
// library_types.h: CUDA_R_8F_UE4M3 == CUDA_R_8F_E4M3).
// ---------------------------------------------------------------------------

static uint8_t encode_e4m3(float x) {
    // Positive scale values only; saturate to E4M3 max (448), flush tiny
    // values to the smallest subnormal instead of zero (scales must be >0
    // to avoid div-by-zero on dequant, though we don't dequant on-device).
    if (x <= 0.0f) return 0x00;
    if (x != x) return 0x7F;  // NaN
    float ax = x;
    if (ax > 448.0f) ax = 448.0f;
    int exp;
    float mant = frexpf(ax, &exp);  // ax = mant * 2^exp, mant in [0.5, 1)
    // Convert to E4M3: value = 1.mmm * 2^(e-7), bias 7, e in [1,15) normal;
    // subnormal when unbiased exp < -6.
    int e_unbiased = exp - 1;  // ax = (mant*2) * 2^(exp-1), mant*2 in [1,2)
    float mant2 = mant * 2.0f;  // in [1,2)
    if (e_unbiased < -6) {
        // subnormal: value = m * 2^-9, m in [0,7]
        float scaled = ax * 512.0f;  // 2^9
        int m = (int)roundf(scaled);
        if (m > 7) m = 7;
        return (uint8_t)(m & 0x7);
    }
    int e_biased = e_unbiased + 7;
    int m3 = (int)roundf((mant2 - 1.0f) * 8.0f);  // round mantissa to 3 bits
    if (m3 == 8) { m3 = 0; e_biased += 1; }
    if (e_biased >= 15) { e_biased = 14; m3 = 7; }  // saturate to max normal (448)
    return (uint8_t)(((e_biased & 0xF) << 3) | (m3 & 0x7));
}

static float decode_e4m3(uint8_t code) {
    int e = (code >> 3) & 0xF;
    int m = code & 0x7;
    if (e == 0) {
        return (float)m * powf(2.0f, -9.0f);
    }
    float mant = 1.0f + (float)m / 8.0f;
    return mant * powf(2.0f, (float)(e - 7));
}

// ---------------------------------------------------------------------------
// Scale swizzle: cuBLAS block-scaling factor layout (docs §3.1.4.3.2):
// 128-row x 4-col tiles, each internally laid out as a 32x4x4 arrangement.
// Cross-checked against PyTorch's to_blocked()/from_blocked() reference.
// `rows` x `cols` is the *logical* unswizzled scale shape (rows = M or N,
// cols = K/16). Output buffer must be `32*ceil(rows,128) * 16*ceil(cols,4)`
// bytes.
// ---------------------------------------------------------------------------

static int ceil_div(int a, int b) { return (a + b - 1) / b; }

static void swizzle_scales(const std::vector<uint8_t>& unswizzled, int rows,
                            int cols, std::vector<uint8_t>& out) {
    int n_row_tiles = ceil_div(rows, 128);
    int n_col_tiles = ceil_div(cols, 4);
    out.assign((size_t)32 * n_row_tiles * 16 * n_col_tiles, 0);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            int scale_tile_h = i / 128;
            int scale_tile_w = j / 4;
            int tile_offset = 512 * (scale_tile_h * n_col_tiles + scale_tile_w);
            int outer = i % 128;
            int inner = j % 4;
            int offset = tile_offset + (outer % 32) * 16 + (outer / 32) * 4 + inner;
            out[offset] = unswizzled[(size_t)i * cols + j];
        }
    }
}

// ---------------------------------------------------------------------------
// Main probe
// ---------------------------------------------------------------------------

int main() {
    const int M = 512, N = 512, K = 512;
    const int BLOCK = 16;  // NVFP4 block size (vs16)
    printf("=== FP4 (NVFP4/CUDA_R_4F_E2M1) cuBLASLt dispatch probe ===\n");
    printf("M=%d N=%d K=%d, block=%d\n", M, N, K, BLOCK);

    int dev = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    printf("Device: %s, sm_%d%d\n", prop.name, prop.major, prop.minor);

    int rt_version = 0, drv_version = 0;
    cudaRuntimeGetVersion(&rt_version);
    cudaDriverGetVersion(&drv_version);
    printf("cudart runtime version (linked): %d, driver version: %d\n", rt_version,
           drv_version);

    // ---- Build fp32 reference A[M,K], B[K,N] (row-major host arrays) ----
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-3.0f, 3.0f);
    std::vector<float> A_ref(M * K), B_ref(K * N), D_ref(M * N, 0.0f);
    for (auto& v : A_ref) v = dist(rng);
    for (auto& v : B_ref) v = dist(rng);

    // fp32 reference GEMM: D = A(MxK) * B(KxN), row-major.
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            double acc = 0.0;
            for (int k = 0; k < K; k++) acc += (double)A_ref[m * K + k] * (double)B_ref[k * N + n];
            D_ref[m * N + n] = (float)acc;
        }

    // ---- Quantize A (row-major MxK, blocks of 16 along K) ----
    // Physical layout for cuBLASLt: A stored as K-major (row = K, col = M),
    // op(A) = TRANS -> logical MxK. So packed bytes are laid out column by
    // column of the KxM physical matrix, i.e. for each m, K contiguous
    // e2m1 values (2 per byte) — same order as our row-major A_ref[m*K+k].
    std::vector<uint8_t> A_q(M * K / 2);
    std::vector<uint8_t> A_scale_unsw(M * (K / BLOCK));
    for (int m = 0; m < M; m++) {
        for (int kb = 0; kb < K / BLOCK; kb++) {
            float amax = 0.0f;
            for (int kk = 0; kk < BLOCK; kk++)
                amax = fmaxf(amax, fabsf(A_ref[m * K + kb * BLOCK + kk]));
            float scale = amax / 6.0f;
            if (scale <= 0.0f) scale = 1.0f;
            uint8_t sc_code = encode_e4m3(scale);
            float sc_val = decode_e4m3(sc_code);  // the value the GEMM will actually use
            A_scale_unsw[m * (K / BLOCK) + kb] = sc_code;
            for (int kk = 0; kk < BLOCK; kk += 2) {
                int k0 = kb * BLOCK + kk, k1 = k0 + 1;
                uint8_t c0 = encode_e2m1(A_ref[m * K + k0] / (sc_val > 0 ? sc_val : 1.0f));
                uint8_t c1 = encode_e2m1(A_ref[m * K + k1] / (sc_val > 0 ? sc_val : 1.0f));
                A_q[(m * K + k0) / 2] = pack_e2m1x2(c0, c1);
            }
        }
    }
    std::vector<uint8_t> A_scale_sw;
    swizzle_scales(A_scale_unsw, M, K / BLOCK, A_scale_sw);

    // ---- Quantize B (col-major KxN physically, op(B) = NOTRANS -> logical
    // KxN; scale tensor logical shape is N x (K/16) per NVIDIA convention
    // (scale indexed by output-column row / K-block), packed along K same
    // as B_ref[k*N+n] contiguous-in-K per fixed n would require a transpose
    // — instead we quantize directly against B's physical K-major layout:
    // for each n, K contiguous values at B_ref[k*N+n]. ----
    std::vector<uint8_t> B_q(K * N / 2);
    std::vector<uint8_t> B_scale_unsw(N * (K / BLOCK));
    for (int n = 0; n < N; n++) {
        for (int kb = 0; kb < K / BLOCK; kb++) {
            float amax = 0.0f;
            for (int kk = 0; kk < BLOCK; kk++)
                amax = fmaxf(amax, fabsf(B_ref[(kb * BLOCK + kk) * N + n]));
            float scale = amax / 6.0f;
            if (scale <= 0.0f) scale = 1.0f;
            uint8_t sc_code = encode_e4m3(scale);
            float sc_val = decode_e4m3(sc_code);
            B_scale_unsw[n * (K / BLOCK) + kb] = sc_code;
            for (int kk = 0; kk < BLOCK; kk += 2) {
                int k0 = kb * BLOCK + kk, k1 = k0 + 1;
                uint8_t c0 = encode_e2m1(B_ref[k0 * N + n] / (sc_val > 0 ? sc_val : 1.0f));
                uint8_t c1 = encode_e2m1(B_ref[k1 * N + n] / (sc_val > 0 ? sc_val : 1.0f));
                // Physical B is K-major (ld=K): byte index for (k,n) pair
                // packing k0/k1 at column n, ld=K -> element offset n*K+k.
                B_q[(n * K + k0) / 2] = pack_e2m1x2(c0, c1);
            }
        }
    }
    std::vector<uint8_t> B_scale_sw;
    swizzle_scales(B_scale_unsw, N, K / BLOCK, B_scale_sw);

    // ---- Diagnostic: pure-software dequant GEMM using our own quant
    // tables (no cuBLASLt, no swizzle) — isolates "is my e2m1/e4m3 quant
    // itself sane" from "did I get the cuBLASLt layout/swizzle right". ----
    {
        std::vector<float> D_sw(M * N, 0.0f);
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                double acc = 0.0;
                for (int kb = 0; kb < K / BLOCK; kb++) {
                    float a_sc = decode_e4m3(A_scale_unsw[m * (K / BLOCK) + kb]);
                    float b_sc = decode_e4m3(B_scale_unsw[n * (K / BLOCK) + kb]);
                    for (int kk = 0; kk < BLOCK; kk++) {
                        int k = kb * BLOCK + kk;
                        uint8_t ab = A_q[(m * K + k) / 2];
                        uint8_t a_nib = (k % 2 == 0) ? (ab & 0xF) : (ab >> 4);
                        uint8_t bb = B_q[(n * K + k) / 2];
                        uint8_t b_nib = (k % 2 == 0) ? (bb & 0xF) : (bb >> 4);
                        float av = decode_e2m1(a_nib) * a_sc;
                        float bv = decode_e2m1(b_nib) * b_sc;
                        acc += (double)av * (double)bv;
                    }
                }
                D_sw[m * N + n] = (float)acc;
            }
        }
        double sq_err = 0.0, sq_ref = 0.0;
        for (int i = 0; i < M * N; i++) {
            double e = (double)D_sw[i] - (double)D_ref[i];
            sq_err += e * e;
            sq_ref += (double)D_ref[i] * (double)D_ref[i];
        }
        printf("\n[diagnostic] pure-software dequant GEMM (no cuBLASLt, no swizzle) "
               "vs fp32 ref: rel_l2 = %.4f\n",
               sqrt(sq_err / (sq_ref > 0 ? sq_ref : 1.0)));
    }

    // ---- Device buffers ----
    void *dA, *dB, *dA_scale, *dB_scale, *dD;
    CUDA_CHECK(cudaMalloc(&dA, A_q.size()));
    CUDA_CHECK(cudaMalloc(&dB, B_q.size()));
    CUDA_CHECK(cudaMalloc(&dA_scale, A_scale_sw.size()));
    CUDA_CHECK(cudaMalloc(&dB_scale, B_scale_sw.size()));
    CUDA_CHECK(cudaMalloc(&dD, (size_t)M * N * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMemcpy(dA, A_q.data(), A_q.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, B_q.data(), B_q.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dA_scale, A_scale_sw.data(), A_scale_sw.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB_scale, B_scale_sw.data(), B_scale_sw.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dD, 0, (size_t)M * N * sizeof(__nv_bfloat16)));

    // ---- cuBLASLt setup ----
    cublasLtHandle_t lt;
    cublasStatus_t st;
    CUBLAS_TRY(cublasLtCreate(&lt), st);
    printf("\ncublasLtCreate: %s\n", cublas_status_str(st));
    if (st != CUBLAS_STATUS_SUCCESS) return 1;

    cublasLtMatmulDesc_t desc;
    CUBLAS_TRY(cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F), st);
    printf("cublasLtMatmulDescCreate: %s\n", cublas_status_str(st));
    if (st != CUBLAS_STATUS_SUCCESS) return 1;

    cublasOperation_t opT = CUBLAS_OP_T, opN = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT));
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));

    cublasLtMatmulMatrixScale_t scale_mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
    CUBLAS_TRY(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE,
                                               &scale_mode, sizeof(scale_mode)), st);
    printf("set A_SCALE_MODE=VEC16_UE4M3: %s\n", cublas_status_str(st));
    CUBLAS_TRY(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE,
                                               &scale_mode, sizeof(scale_mode)), st);
    printf("set B_SCALE_MODE=VEC16_UE4M3: %s\n", cublas_status_str(st));
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &dA_scale, sizeof(dA_scale));
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &dB_scale, sizeof(dB_scale));

    // A physical: K-major (rows=K, cols=M, ld=K), op=T -> logical MxK.
    cublasLtMatrixLayout_t Adesc, Bdesc, Ddesc;
    CUBLAS_TRY(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_4F_E2M1, K, M, K), st);
    printf("cublasLtMatrixLayoutCreate(A, CUDA_R_4F_E2M1): %s\n", cublas_status_str(st));
    if (st != CUBLAS_STATUS_SUCCESS) return 1;
    // B physical: K-major (rows=K, cols=N, ld=K), op=N -> logical KxN.
    CUBLAS_TRY(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_4F_E2M1, K, N, K), st);
    printf("cublasLtMatrixLayoutCreate(B, CUDA_R_4F_E2M1): %s\n", cublas_status_str(st));
    if (st != CUBLAS_STATUS_SUCCESS) return 1;
    CUBLAS_TRY(cublasLtMatrixLayoutCreate(&Ddesc, CUDA_R_16BF, M, N, M), st);
    printf("cublasLtMatrixLayoutCreate(D, CUDA_R_16BF): %s\n", cublas_status_str(st));
    if (st != CUBLAS_STATUS_SUCCESS) return 1;

    cublasLtMatmulPreference_t pref;
    cublasLtMatmulPreferenceCreate(&pref);
    size_t ws_bytes = 32ull * 1024 * 1024;
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                          &ws_bytes, sizeof(ws_bytes));

    cublasLtMatmulHeuristicResult_t heur[4];
    int returned = 0;
    CUBLAS_TRY(cublasLtMatmulAlgoGetHeuristic(lt, desc, Adesc, Bdesc, Ddesc, Ddesc, pref, 4,
                                               heur, &returned), st);
    printf("\ncublasLtMatmulAlgoGetHeuristic: %s, returned=%d\n", cublas_status_str(st), returned);
    for (int i = 0; i < returned; i++) {
        printf("  algo[%d]: workspaceSize=%zu, wavesCount=%f\n", i,
               heur[i].workspaceSize, heur[i].wavesCount);
    }

    bool dispatched_and_ran = false;
    if (st == CUBLAS_STATUS_SUCCESS && returned > 0) {
        void* ws;
        CUDA_CHECK(cudaMalloc(&ws, heur[0].workspaceSize > 0 ? heur[0].workspaceSize : 1));
        float alpha = 1.0f, beta = 0.0f;
        cudaEvent_t ev0, ev1;
        cudaEventCreate(&ev0);
        cudaEventCreate(&ev1);
        cudaEventRecord(ev0);
        CUBLAS_TRY(cublasLtMatmul(lt, desc, &alpha, dA, Adesc, dB, Bdesc, &beta, dD, Ddesc, dD,
                                   Ddesc, &heur[0].algo, ws, heur[0].workspaceSize, 0),
                   st);
        cudaEventRecord(ev1);
        cudaError_t sync_err = cudaEventSynchronize(ev1);
        printf("\ncublasLtMatmul: %s\n", cublas_status_str(st));
        printf("cudaEventSynchronize after matmul: %s\n", cudaGetErrorString(sync_err));
        cudaError_t last = cudaGetLastError();
        printf("cudaGetLastError after matmul: %s\n", cudaGetErrorString(last));

        if (st == CUBLAS_STATUS_SUCCESS && sync_err == cudaSuccess && last == cudaSuccess) {
            dispatched_and_ran = true;
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, ev0, ev1);
            printf("FP4 GEMM wall time: %.4f ms\n", ms);

            // ---- Correctness check vs fp32 reference ----
            std::vector<__nv_bfloat16> D_host(M * N);
            CUDA_CHECK(cudaMemcpy(D_host.data(), dD, (size_t)M * N * sizeof(__nv_bfloat16),
                                   cudaMemcpyDeviceToHost));
            // D is written column-major (rows=M, cols=N, ld=M): D_host[m+n*M].
            // D_ref is row-major (D_ref[m*N+n]) — must un-transpose to compare.
            double sq_err = 0.0, sq_ref = 0.0;
            float max_abs_err = 0.0f;
            for (int m = 0; m < M; m++) {
                for (int n = 0; n < N; n++) {
                    float got = __bfloat162float(D_host[m + n * M]);
                    float want = D_ref[m * N + n];
                    double e = (double)got - (double)want;
                    sq_err += e * e;
                    sq_ref += (double)want * (double)want;
                    max_abs_err = fmaxf(max_abs_err, fabsf(got - want));
                }
            }
            double rel_l2 = sqrt(sq_err / (sq_ref > 0 ? sq_ref : 1.0));
            printf("\nNumeric check: relative L2 error = %.4f, max abs err = %.4f\n", rel_l2,
                   max_abs_err);
            printf("(NVFP4 e2m1 data has ~6-12%% per-element quantization noise; "
                   "rel_l2 < 0.20 is considered PASS for this probe)\n");
            printf("VERDICT numeric: %s\n", rel_l2 < 0.20 ? "PASS" : "FAIL (layout/packing likely wrong, not a dispatch failure)");

            // ---- Timing comparison: equivalent bf16 GEMM ----
            // Same physical layout convention as A_q/B_q: A_bf flat-copies
            // cleanly (A_l is K-major op=T, ld=K, and A_ref's row-major
            // M*K+k flatten already matches that physical layout). B_bf
            // must be explicitly transposed into K-major (row=k,col=n)
            // order to match Bdesc — a naive flat copy of row-major
            // B_ref[k*N+n] would silently transpose B whenever N==K.
            std::vector<__nv_bfloat16> A_bf(M * K), B_bf(K * N);
            for (int i = 0; i < M * K; i++) A_bf[i] = __float2bfloat16(A_ref[i]);
            for (int k = 0; k < K; k++)
                for (int n = 0; n < N; n++)
                    B_bf[k + n * K] = __float2bfloat16(B_ref[k * N + n]);
            void *dAbf, *dBbf, *dDbf;
            CUDA_CHECK(cudaMalloc(&dAbf, M * K * sizeof(__nv_bfloat16)));
            CUDA_CHECK(cudaMalloc(&dBbf, K * N * sizeof(__nv_bfloat16)));
            CUDA_CHECK(cudaMalloc(&dDbf, (size_t)M * N * sizeof(__nv_bfloat16)));
            CUDA_CHECK(cudaMemcpy(dAbf, A_bf.data(), M * K * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(dBbf, B_bf.data(), K * N * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

            cublasLtMatmulDesc_t desc_bf;
            cublasLtMatmulDescCreate(&desc_bf, CUBLAS_COMPUTE_32F, CUDA_R_32F);
            cublasLtMatmulDescSetAttribute(desc_bf, CUBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT));
            cublasLtMatmulDescSetAttribute(desc_bf, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));
            cublasLtMatrixLayout_t Abf_l, Bbf_l, Dbf_l;
            cublasLtMatrixLayoutCreate(&Abf_l, CUDA_R_16BF, K, M, K);
            cublasLtMatrixLayoutCreate(&Bbf_l, CUDA_R_16BF, K, N, K);
            cublasLtMatrixLayoutCreate(&Dbf_l, CUDA_R_16BF, M, N, M);
            cublasLtMatmulPreference_t pref_bf;
            cublasLtMatmulPreferenceCreate(&pref_bf);
            cublasLtMatmulPreferenceSetAttribute(pref_bf, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                  &ws_bytes, sizeof(ws_bytes));
            cublasLtMatmulHeuristicResult_t heur_bf[1];
            int ret_bf = 0;
            cublasStatus_t st_bf = cublasLtMatmulAlgoGetHeuristic(lt, desc_bf, Abf_l, Bbf_l, Dbf_l,
                                                                    Dbf_l, pref_bf, 1, heur_bf, &ret_bf);
            if (st_bf == CUBLAS_STATUS_SUCCESS && ret_bf > 0) {
                void* ws_bf;
                CUDA_CHECK(cudaMalloc(&ws_bf, heur_bf[0].workspaceSize > 0 ? heur_bf[0].workspaceSize : 1));
                cudaEvent_t bf0, bf1;
                cudaEventCreate(&bf0);
                cudaEventCreate(&bf1);
                cudaEventRecord(bf0);
                cublasLtMatmul(lt, desc_bf, &alpha, dAbf, Abf_l, dBbf, Bbf_l, &beta, dDbf, Dbf_l,
                                dDbf, Dbf_l, &heur_bf[0].algo, ws_bf, heur_bf[0].workspaceSize, 0);
                cudaEventRecord(bf1);
                cudaEventSynchronize(bf1);
                float bf_ms = 0.0f;
                cudaEventElapsedTime(&bf_ms, bf0, bf1);
                printf("\nEquivalent bf16 GEMM wall time: %.4f ms (FP4/BF16 time ratio = %.3f)\n",
                       bf_ms, ms / bf_ms);
                // Sanity-check the *layout convention itself* (TN, K-major
                // operands, col-major D) independent of FP4 packing: does
                // this exact op/layout scheme reproduce D_ref for plain
                // bf16 operands?
                std::vector<__nv_bfloat16> Dbf_host(M * N);
                CUDA_CHECK(cudaMemcpy(Dbf_host.data(), dDbf, (size_t)M * N * sizeof(__nv_bfloat16),
                                       cudaMemcpyDeviceToHost));
                double sq_e2 = 0.0, sq_r2 = 0.0;
                for (int m = 0; m < M; m++) {
                    for (int n = 0; n < N; n++) {
                        double e = (double)__bfloat162float(Dbf_host[m + n * M]) - (double)D_ref[m * N + n];
                        sq_e2 += e * e;
                        sq_r2 += (double)D_ref[m * N + n] * (double)D_ref[m * N + n];
                    }
                }
                printf("[diagnostic] bf16-GEMM (same TN layout convention) vs fp32 ref: "
                       "rel_l2 = %.4f (should be small, ~bf16 rounding only)\n",
                       sqrt(sq_e2 / (sq_r2 > 0 ? sq_r2 : 1.0)));
                cudaFree(ws_bf);
            } else {
                printf("bf16 comparison GEMM heuristic failed: %s, returned=%d\n",
                       cublas_status_str(st_bf), ret_bf);
            }
            cudaFree(dAbf); cudaFree(dBbf); cudaFree(dDbf);
        }
        cudaFree(ws);
    }

    printf("\n=== PROBE RESULT ===\n");
    printf("dispatched_and_ran = %s\n", dispatched_and_ran ? "true" : "false");
    return dispatched_and_ran ? 0 : 2;
}
