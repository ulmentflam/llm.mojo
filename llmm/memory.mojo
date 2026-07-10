from std.ffi import _get_global_or_null, external_call
from std.gpu.host import DeviceContext
from std.memory import UnsafePointer, alloc


# ===----------------------------------------------------------------------=== #
# Memory pointer aliases
# ===----------------------------------------------------------------------=== #


# Owned heap / HostBuffer — safe in struct fields (GPU-target builds reject AnyOrigin).
comptime MutMemPtr[dtype: DType] = UnsafePointer[
    Scalar[dtype], MutUntrackedOrigin
]
comptime ImmutMemPtr[dtype: DType] = UnsafePointer[
    Scalar[dtype], ImmutUntrackedOrigin
]


# llmm kernel / tensor API — interoperates with InputTensor.unsafe_ptr().
comptime MutKernelPtr[dtype: DType] = UnsafePointer[Scalar[dtype], MutAnyOrigin]
comptime ImmutKernelPtr[dtype: DType] = UnsafePointer[
    Scalar[dtype], ImmutAnyOrigin
]


@always_inline
def rebind_mut_mem[
    dtype: DType
](ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],) -> MutMemPtr[dtype]:
    return rebind[MutMemPtr[dtype]](ptr)


@always_inline
def rebind_immut_mem[
    dtype: DType
](ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],) -> ImmutMemPtr[dtype]:
    return rebind[ImmutMemPtr[dtype]](ptr)


@always_inline
def as_mut_kernel[
    dtype: DType
](ptr: MutMemPtr[dtype],) -> MutKernelPtr[dtype]:
    """Explicit MemPtr → KernelPtr bridge for llmm dispatch from owned memory.
    """
    return rebind[MutKernelPtr[dtype]](ptr.as_unsafe_any_origin())


@always_inline
def as_immut_kernel[
    dtype: DType
](ptr: ImmutMemPtr[dtype],) -> ImmutKernelPtr[dtype]:
    return rebind[ImmutKernelPtr[dtype]](ptr.as_unsafe_any_origin())


@always_inline
def as_immut_kernel_from_mut[
    dtype: DType
](ptr: MutMemPtr[dtype],) -> ImmutKernelPtr[dtype]:
    return rebind[ImmutKernelPtr[dtype]](ptr.as_unsafe_any_origin())


# ===----------------------------------------------------------------------=== #
# Persistent device-buffer process globals
# ===----------------------------------------------------------------------=== #


@always_inline
def persistent_device_buffer[
    dtype: DType
](
    ctx: DeviceContext, name_suffix: String, count: Int, *, zero: Bool = False
) raises -> MutKernelPtr[dtype]:
    """Allocate-once, heap-held device buffer via a device-keyed process
    global (`KGEN_CompilerRT_InsertGlobal`) — mirrors llm.c's single global
    scratch-buffer idiom, and factors out what used to be ~18 lines of
    identical `_get_global_or_null` / `alloc` / `init_pointee_move` /
    `external_call` / `rebind` boilerplate repeated at each call site
    (`_cublaslt_workspace`, `_dbias_scratch`, `_dbias_counters`,
    `_ln_dparam_scratch`, `_ln_bwd_dparam_scratch` as of the 2026-07-10 DRY
    pass — see `docs/ai/dry_consolidation_audit_2026-07-10.md` finding F4).

    The first call for a given `(name_suffix, ctx.id())` pair allocates
    `count` elements (memset to zero iff `zero=True`); every subsequent call
    with the same name/device — regardless of what `count`/`zero` it passes —
    just looks up and returns the already-cached pointer without
    reallocating or re-zeroing. Callers are responsible for choosing a
    `count` upper bound valid for every call site sharing a `name_suffix`,
    exactly as before this helper existed (each call site already documents
    its own bound derivation).
    """
    comptime BufType = type_of(ctx.enqueue_create_buffer[dtype](1))
    var name = String("LLMM_") + name_suffix + String("_") + String(ctx.id())
    if gp := _get_global_or_null(name):
        var p = gp.value().bitcast[BufType]()
        return rebind[MutKernelPtr[dtype]](p[].unsafe_ptr())
    var buf = ctx.enqueue_create_buffer[dtype](count)
    if zero:
        ctx.enqueue_memset(buf, Scalar[dtype](0))
    var hp = alloc[BufType](1)
    hp.init_pointee_move(buf^)
    external_call["KGEN_CompilerRT_InsertGlobal", NoneType](
        StringSlice(name), hp.bitcast[NoneType]()
    )
    return rebind[MutKernelPtr[dtype]](hp[].unsafe_ptr())
