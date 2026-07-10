from std.memory import Span

from llmm.memory import ImmutKernelPtr, MutMemPtr


def get_dtype_size(dtype: DType) -> Int:
    if dtype == DType.float32:
        return 4
    elif dtype == DType.bfloat16 or dtype == DType.float16:
        return 2
    elif dtype == DType.int32:
        return 4
    elif dtype == DType.int16:
        return 2
    elif dtype == DType.int8 or dtype == DType.uint8:
        return 1
    elif (
        dtype == DType.float8_e4m3fn
        or dtype == DType.float8_e5m2
        or dtype == DType.float8_e4m3fnuz
        or dtype == DType.float8_e5m2fnuz
    ):
        # 1 byte. Defensive: no host buffer is fp8-typed on the normal path
        # (fp8 is a transient GEMM-operand dtype only — see
        # docs/ai/fp8_training_design.md §1.1/§4), but the original silent
        # `else: return 4` fallback is exactly the landmine that caused the
        # bf16 host-buffer element-size mismatch this function's other
        # branches were added to fix; fp8 gets its own explicit case rather
        # than falling through to the wrong 4-byte default.
        return 1
    else:
        # Sub-byte dtypes (e.g. fp4/e2m1, 2 values/byte) have no whole-byte
        # element size and are not expected to reach a byte-count call site;
        # rather than silently returning a wrong size (the landmine this
        # function guards against), fail loudly so a future fp4 checkpoint/
        # dump path is forced to special-case its packed layout explicitly
        # (docs/ai/fp8_training_design.md §3, extension seam 6) instead of
        # inheriting a wrong default.
        debug_assert(
            False,
            (
                "get_dtype_size: unrecognized dtype (sub-byte/fp4 packed dtypes"
                " need dedicated packed handling, not a per-element byte size)"
            ),
        )
        return 4


# Generic utility function to read count elements from a FileHandle directly into an UnsafePointer.
def read_and_copy[
    dtype: DType,
](mut file: FileHandle, dest: MutMemPtr[dtype], count: Int,) raises:
    var bytes_to_read = count * get_dtype_size(dtype)
    var bytes_read = file.read_bytes(bytes_to_read)
    if len(bytes_read) < bytes_to_read:
        raise Error("Failed to read enough bytes from file")

    var src_ptr = bytes_read.unsafe_ptr().bitcast[Scalar[dtype]]()
    for i in range(count):
        dest.store(i, src_ptr.load(i))


# Generic utility to write count elements from a buffer to a FileHandle as raw
# little-endian bytes. The write-side counterpart to read_and_copy.
@always_inline
def write_buffer[
    dtype: DType,
](mut file: FileHandle, src: MutMemPtr[dtype], count: Int) raises:
    var nbytes = count * get_dtype_size(dtype)
    file.write_all(Span(ptr=src.bitcast[Byte](), length=nbytes))
