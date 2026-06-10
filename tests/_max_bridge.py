"""MAX-graph bridge for tests of `@compiler.register`'d Mojo kernels.

This module is the generic harness only: `run_custom_op` builds a graph
from a typed list of positional arguments (`MutableBuf`, `ReadTensor`,
`ScalarArg`), compiles, executes, and returns post-step numpy views of
every mutable buffer.

Compiled models are cached on (kernel name, device, arg signature) —
scalars are runtime 0-d tensor inputs rather than baked constants, so a
multi-step trajectory (varying `t`, same shapes/dtypes) compiles exactly
once. This is what keeps the equivalence suite fast; don't switch scalars
back to `ops.constant` without re-measuring.

Per-kernel wrappers (e.g., `tests.kernels.adamw.step`) live under
`tests/kernels/` and just package their kernel's parameters into the right
arg list. Add new kernels there; do not add per-kernel boilerplate here.

## How MAX 26.3 delivers runtime args to a registered op

After reading `max.graph.ops.custom` + the production op
`mo.scatter_set_constant` (MOGGKernelAPI.mojo:1300-1318 / kernels.py:3737-3766):

  *  Runtime tensors of any rank go in `values=`.
  *  Runtime scalars are 0-d TensorValues the kernel receives as a Scalar
     or 0-d InputTensor — either `ops.constant(...)` or, as here, a 0-d
     graph input (the kernel can't tell the difference, but only the
     latter is reusable across calls with different values).
  *  `parameters=` is compile-time only (`bool | int | str | DType`); not
     used here.
  *  Mutable buffers (kernel-side `MutableInputTensor[...]`) get a
     `BufferType` graph input and the call goes through `ops.inplace_custom`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Union

import numpy as np

if TYPE_CHECKING:
    from max.driver import Device
    from max.engine import InferenceSession, Model


MOJO_KERNELS_DIR = Path(__file__).resolve().parents[1] / "llmm"


class KernelSignatureMismatch(RuntimeError):
    """Raised when MAX rejects the registered op's signature.

    Distinct from a generic verifier error so the pytest skip reason
    stays actionable ("update the kernel signature" rather than a raw
    MLIR traceback).
    """


# ---------------------------------------------------------------------------
# Argument descriptors. Build one per kernel parameter, in the kernel's
# declared order, and hand the list to `run_custom_op`.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MutableBuf:
    """In-place buffer; maps to a kernel `MutableInputTensor[...]`.

    The harness returns this buffer's post-execution numpy view to the caller
    (mutables are reported in their `args`-list order).
    """

    array: np.ndarray
    dtype_name: str  # "float32" | "bfloat16" | "float16"


@dataclass(frozen=True)
class ReadTensor:
    """Read-only tensor input; maps to a kernel `InputTensor[...]`."""

    array: np.ndarray
    dtype_name: str


@dataclass(frozen=True)
class ScalarArg:
    """Runtime 0-d scalar input; maps to a kernel `Scalar[...]` or
    unsigned-int parameter. Fed at execution time, so a step counter or
    hyperparameter sweep reuses one compiled model."""

    value: Union[int, float]
    dtype_name: str  # "float32" | "bfloat16" | "float16" | "uint32" | "int32" | "int64"


KernelArg = Union[MutableBuf, ReadTensor, ScalarArg]


def max_dtype(name: str):
    from max.dtype import DType

    return {
        "float32": DType.float32,
        "bfloat16": DType.bfloat16,
        "float16": DType.float16,
        "uint32": DType.uint32,
        "int32": DType.int32,
        "int64": DType.int64,
    }[name]


def pick_device():
    # Default to CPU. The kernel's GPU branch compiles in Mojo but Apple
    # Metal's metallib stage currently rejects MAX's lowered KGEN
    # ("could not elaborate the generated KGEN"). Modular's Apple-GPU
    # custom ops use `foreach` rather than direct compile_function;
    # migrating the optimizer to that pattern is a separate task.
    # Set MAX_USE_ACCELERATOR=1 to opt in for experimentation.
    import os
    from max.driver import CPU, Accelerator, accelerator_count

    if os.environ.get("MAX_USE_ACCELERATOR") and accelerator_count() > 0:
        return Accelerator()
    return CPU()


# Compiled-model cache. Keyed on everything that shapes the graph:
# kernel name, device, and each arg's (kind, dtype, shape). Scalar *values*
# are deliberately absent — they're runtime inputs, so 65 optimizer steps
# hit one entry. Sessions are cached alongside models because a model is
# only valid while its session lives.
_MODEL_CACHE: dict[tuple, "Model"] = {}
_SESSION_CACHE: dict[tuple, "InferenceSession"] = {}
_DEFAULT_DEVICE: "Device | None" = None


def _signature(kernel_name: str, args: list[KernelArg], device) -> tuple:
    parts: list[tuple] = []
    for a in args:
        if isinstance(a, MutableBuf):
            parts.append(("mut", a.dtype_name, a.array.shape))
        elif isinstance(a, ReadTensor):
            parts.append(("read", a.dtype_name, a.array.shape))
        else:
            parts.append(("scalar", a.dtype_name))
    return (kernel_name, type(device).__name__, str(device), tuple(parts))


def _compile_model(kernel_name: str, args: list[KernelArg], device):
    from max.driver import CPU
    from max.engine import InferenceSession
    from max.graph import BufferType, DeviceRef, Graph, TensorType, ops

    dev_ref = DeviceRef.from_device(device)
    cpu_ref = DeviceRef.from_device(CPU())

    # Every arg — tensor or scalar — is a graph input, in kernel order.
    # Scalars are 0-d tensors pinned to CPU (mirroring where the old
    # `ops.constant` scalars lived).
    input_types: list = []
    for a in args:
        t = max_dtype(a.dtype_name)
        if isinstance(a, MutableBuf):
            input_types.append(BufferType(t, shape=list(a.array.shape), device=dev_ref))
        elif isinstance(a, ReadTensor):
            input_types.append(TensorType(t, shape=list(a.array.shape), device=dev_ref))
        else:
            input_types.append(TensorType(t, shape=[], device=cpu_ref))

    def forward(*graph_inputs):
        ops.inplace_custom(name=kernel_name, device=dev_ref, values=list(graph_inputs))
        return ()

    # Graph construction with `custom_extensions=[...]` triggers kernel
    # compilation, so wrap both Graph() and session.load() in the same
    # try/except — either can raise on a signature/kernel mismatch.
    try:
        graph = Graph(
            kernel_name,
            forward=forward,
            input_types=input_types,
            custom_extensions=[MOJO_KERNELS_DIR],
        )
        session = InferenceSession(devices=[device])
        model = session.load(graph)
    except Exception as e:
        raise KernelSignatureMismatch(
            f"MAX could not compile/load the {kernel_name!r} graph. Most "
            "likely the kernel has a Mojo compile error or its registered "
            "signature does not match the args this bridge is sending. "
            "See the docstring at the top of tests/_max_bridge.py for how "
            f"arguments map to the kernel.\nUnderlying error: {e}"
        ) from e
    return session, model


def run_custom_op(
    *,
    kernel_name: str,
    args: list[KernelArg],
    device: "Device | None" = None,
) -> list[np.ndarray]:
    """Compile (cached), load, and execute a registered custom op.

    `args` must be ordered to match the kernel's declared positional
    parameters. Tensors and scalars alike become graph inputs, fed at
    execution time. Returns the post-execution numpy view of each
    `MutableBuf`, in the order they appear in `args`.
    """
    if device is None:
        global _DEFAULT_DEVICE
        if _DEFAULT_DEVICE is None:
            _DEFAULT_DEVICE = pick_device()
        device = _DEFAULT_DEVICE

    if not any(isinstance(a, MutableBuf) for a in args):
        # Non-inplace kernels would need `ops.custom` + output specs.
        # Add when the first such kernel lands.
        raise NotImplementedError(
            f"{kernel_name!r}: run_custom_op currently requires at least one "
            "MutableBuf (uses ops.inplace_custom). Extend the bridge when "
            "you add a kernel that returns outputs by value."
        )

    key = _signature(kernel_name, args, device)
    model = _MODEL_CACHE.get(key)
    if model is None:
        session, model = _compile_model(kernel_name, args, device)
        _SESSION_CACHE[key] = session
        _MODEL_CACHE[key] = model

    # Upload graph inputs and remember each MutableBuf's buffer so we can
    # read it back post-execution.
    exec_args: list = []
    mutable_buffers: dict[int, object] = {}
    for slot, a in enumerate(args):
        if isinstance(a, ScalarArg):
            buf = _scalar_buffer(a.value, a.dtype_name)
        else:
            buf = _to_device_buffer(a.array, a.dtype_name, device)
            if isinstance(a, MutableBuf):
                mutable_buffers[slot] = buf
        exec_args.append(buf)

    model.execute(*exec_args)

    return [
        _from_device_buffer(mutable_buffers[i], a.dtype_name)
        for i, a in enumerate(args)
        if isinstance(a, MutableBuf)
    ]


def _scalar_buffer(value: Union[int, float], dtype_name: str):
    """0-d CPU buffer for a runtime scalar input (see `ScalarArg`)."""
    from max.driver import Buffer

    if dtype_name == "bfloat16":
        import torch

        # Truncate fp32→bf16 (drop low mantissa bits) to match what MAX's
        # `ops.constant` produces. torch's round-to-nearest would turn
        # beta2=0.999 into bf16 1.0, which zeroes the second moment and
        # makes the kernel's bias correction divide by zero.
        bits = np.float32(value).view(np.uint32) >> np.uint32(16)
        t = torch.tensor(int(bits), dtype=torch.uint16).view(torch.bfloat16)
        return Buffer.from_dlpack(t)
    np_dtype = {
        "float32": np.float32,
        "float16": np.float16,
        "uint32": np.uint32,
        "int32": np.int32,
        "int64": np.int64,
    }[dtype_name]
    return Buffer.from_numpy(np.array(value, dtype=np_dtype))


def _to_device_buffer(arr: np.ndarray, dtype_name: str, device):
    """Wrap a numpy array as a max.driver.Buffer on `device`.

    For bf16, `arr` is uint16-shaped (numpy can't represent bf16 natively).
    Reinterpret through a torch bf16 tensor so the Buffer dtype tag is bf16,
    not ui16 — MAX checks this against the graph's BufferType.
    """
    from max.driver import Buffer

    if dtype_name == "bfloat16":
        import torch

        bf16 = torch.from_numpy(arr.copy()).view(torch.bfloat16).contiguous()
        return Buffer.from_dlpack(bf16).to(device)
    return Buffer.from_numpy(arr).to(device)


def _from_device_buffer(buf, dtype_name: str) -> np.ndarray:
    """Mirror of `_to_device_buffer`: returns a numpy array, with bf16
    re-encoded as uint16 so callers can diff fixtures consistently."""
    if dtype_name == "bfloat16":
        import torch

        # Pull back via dlpack and reinterpret as uint16 for numpy storage.
        return torch.from_dlpack(buf).view(torch.uint16).cpu().numpy()
    return buf.to_numpy()
