import compiler
from tensor import InputTensor
from std.sys import simd_width_of
from std.memory import UnsafePointer
from std.math import fma, sqrt, ceildiv
from std.gpu.host.info import is_cpu, is_gpu
from std.runtime.asyncrt import DeviceContextPtr
from std.gpu import block_dim, block_idx, thread_idx
from std.algorithm import vectorize, sync_parallelize
from tensor.managed_tensor_slice import (
    _MutableInputTensor as MutableInputTensor,
)


# ===----------------------------------------------------------------------=== #
# Constants and Comptime Variables
# ===----------------------------------------------------------------------=== #

comptime CHUNK_SIZE = 4096
comptime UNROLL = 4


# ===----------------------------------------------------------------------=== #
# Softmax
# ===----------------------------------------------------------------------=== #
