from std.memory import UnsafePointer


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
