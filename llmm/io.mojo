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
    else:
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
