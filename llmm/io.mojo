from std.memory import Span

from llmm.memory import ImmutKernelPtr, MutMemPtr


def get_dtype_size(dtype: DType) raises -> Int:
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
        # 1 byte. fp8 is a transient GEMM-operand dtype and should not reach
        # a host byte-count call site, but give it an explicit case so it
        # never falls through to the wrong 4-byte default below.
        return 1
    else:
        # Sub-byte/packed dtypes (fp4/e2m1) have no whole-byte element size
        # and must be special-cased by any future packed path. A
        # `debug_assert` here is a release no-op — it would silently return
        # the wrong 4-byte size in a release build, exactly the failure mode
        # this branch exists to close — so raise instead, unconditionally.
        raise Error(
            "get_dtype_size: unrecognized dtype (sub-byte/fp4 packed dtypes"
            " need dedicated packed handling, not a per-element byte size)"
        )


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
