# ===----------------------------------------------------------------------=== #
# Shared GPU context helper for GPU-launching test modules (tests/test_hadamard
# .mojo, tests/test_nvfp4_quant.mojo, tests/test_rng_sr.mojo, and any sibling
# added later).
#
# Split into its own module (not `test_`-prefixed, so the Makefile's
# `tests/test_*.mojo` glob never runs it directly and `TestSuite.discover_tests
# [__functions_in_module()]()` in each real test file never reflects over it --
# see tests/_lowp_test_common.mojo's module comment for why that reflection
# split matters) so every GPU test file shares one canonical helper instead of
# one copy per file.
# ===----------------------------------------------------------------------=== #
#
# All GPU tests below share ONE process-wide `DeviceContext` rather than each
# constructing its own. Constructing a *second* fresh `DeviceContext()` in the
# same process -- after a prior context has launched a kernel and done a
# device->host readback -- deadlocks on the GB10 (Mojo 26.5.0.dev2026071006):
# the second `DeviceContext()` constructor never returns (all threads park in
# futex_wait, 0% CPU, GPU idle). The training loop only ever uses a single
# persistent context, so this only ever bit the multi-`DeviceContext()` test
# pattern here (and in the sibling FP4-campaign tests). Holding one context in
# a process global -- never torn down between tests -- sidesteps it, mirroring
# the `persistent_device_buffer` global idiom in `llmm/memory.mojo`.

from std.ffi import _get_global_or_null, external_call
from std.gpu.host import DeviceContext
from std.memory import alloc


def shared_gpu_ctx() raises -> DeviceContext:
    var name = String("LLMM_TEST_SHARED_GPU_CTX")
    if gp := _get_global_or_null(name):
        return gp.value().bitcast[DeviceContext]()[]
    var ctx = DeviceContext()
    var hp = alloc[DeviceContext](1)
    hp.unsafe_write(ctx^)
    external_call["KGEN_CompilerRT_InsertGlobal", NoneType](
        StringSlice(name), hp.bitcast[NoneType]()
    )
    return hp[]
