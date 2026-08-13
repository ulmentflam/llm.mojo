from std.ffi import _get_global_or_null, external_call
from max.gpu.host import DeviceContext
from std.memory import UnsafePointer, alloc


# ===----------------------------------------------------------------------=== #
# Raw heap allocation
# ===----------------------------------------------------------------------=== #


@always_inline
def heap_alloc[T: AnyType](count: Int) -> Pointer[T, MutUntrackedOrigin]:
    """Raw heap allocation of `count` `T`s, freed with `.unsafe_free()`.

    The single call site of the deprecated count-based `alloc` in this repo.
    See docs/ai/mojo_1_0_migration.md for why it cannot be migrated to the
    `Layout`-based replacement yet.
    """
    return alloc[T](count)


# ===----------------------------------------------------------------------=== #
# Memory pointer aliases
# ===----------------------------------------------------------------------=== #


# Owned heap / HostBuffer — safe in struct fields (GPU-target builds reject AnyOrigin).
comptime MutMemPtr[dtype: DType] = Pointer[Scalar[dtype], MutUntrackedOrigin]
comptime ImmutMemPtr[dtype: DType] = Pointer[Scalar[dtype], ImmUntrackedOrigin]


# llmm kernel / tensor API — interoperates with InputTensor.unsafe_ptr().
comptime MutKernelPtr[dtype: DType] = Pointer[Scalar[dtype], MutAnyOrigin]
comptime ImmutKernelPtr[dtype: DType] = Pointer[Scalar[dtype], ImmutAnyOrigin]


@always_inline
def rebind_mut_mem[
    dtype: DType
](ptr: Pointer[Scalar[dtype], MutAnyOrigin],) -> MutMemPtr[dtype]:
    return rebind[MutMemPtr[dtype]](ptr)


@always_inline
def rebind_immut_mem[
    dtype: DType
](ptr: Pointer[Scalar[dtype], ImmutAnyOrigin],) -> ImmutMemPtr[dtype]:
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
    """Allocate-once, heap-held device buffer keyed by (name_suffix, ctx.id())
    via a process global. The first call for a given key allocates `count`
    elements (zeroed iff `zero=True`); every later call with the same
    name/device returns the cached pointer and ignores its `zero`. `count`
    on a later call may be <= the originally-allocated size; a larger
    `count` raises (see below) rather than silently handing back a
    too-small buffer. The caller must still pass a `count` that
    upper-bounds every call site sharing a name_suffix -- whichever call
    runs first is what actually sizes the allocation.
    """
    comptime BufType = type_of(ctx.enqueue_create_buffer[dtype](1))
    var name = String("LLMM_") + name_suffix + String("_") + String(ctx.id())
    var cached = _get_global_or_null(name)
    if cached:
        var p = cached.value().unsafe_bitcast[BufType]()
        var cached_count = len(p[])
        if count > cached_count:
            raise Error(
                "persistent_device_buffer: cached '"
                + name_suffix
                + "' buffer holds "
                + String(cached_count)
                + " elements but this call requested "
                + String(count)
                + " -- every call site sharing a name_suffix must pass a"
                " count that upper-bounds all of them (the first call for"
                " a given name_suffix fixes the allocation size)"
            )
        return rebind[MutKernelPtr[dtype]](p[].unsafe_ptr())
    var buf = ctx.enqueue_create_buffer[dtype](count)
    if zero:
        ctx.enqueue_memset(buf, Scalar[dtype](0))
    var hp = heap_alloc[BufType](1)
    hp.unsafe_write(buf^)
    external_call["KGEN_CompilerRT_InsertGlobal", NoneType](
        StringSlice(name), hp.unsafe_bitcast[NoneType]()
    )
    return rebind[MutKernelPtr[dtype]](hp[].unsafe_ptr())
