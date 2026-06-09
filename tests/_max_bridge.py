"""MAX-graph bridge for tests of `@compiler.register`'d Mojo kernels.

This module is the generic harness only: `run_custom_op` builds a graph
from a typed list of positional arguments (`MutableBuf`, `ReadTensor`,
`ConstScalar`), compiles, executes, and returns post-step numpy views of
every mutable buffer.

Per-kernel wrappers (e.g., `tests.kernels.adamw.step`) live under
`tests/kernels/` and just package their kernel's parameters into the right
arg list. Add new kernels there; do not add per-kernel boilerplate here.

## How MAX 26.3 delivers runtime args to a registered op

After reading `max.graph.ops.custom` + the production op
`mo.scatter_set_constant` (MOGGKernelAPI.mojo:1300-1318 / kernels.py:3737-3766):

  *  Runtime tensors of any rank go in `values=`.
  *  Runtime scalars are wrapped with `ops.constant(py_val, dtype, device)`,
     producing a 0-d TensorValue the kernel receives as a Scalar or 0-d
     InputTensor.
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
class ConstScalar:
    """0-d scalar baked via `ops.constant`; maps to a kernel `Scalar[...]`
    or unsigned-int parameter. Folded into the graph at compile time, so
    every distinct value triggers a fresh compile."""

    value: Union[int, float]
    dtype_name: str  # "float32" | "bfloat16" | "float16" | "uint32"


KernelArg = Union[MutableBuf, ReadTensor, ConstScalar]


def max_dtype(name: str):
    from max.dtype import DType

    return {
        "float32": DType.float32,
        "bfloat16": DType.bfloat16,
        "float16": DType.float16,
        "uint32": DType.uint32,
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


def run_custom_op(
    *,
    kernel_name: str,
    args: list[KernelArg],
    device: "Device | None" = None,
) -> list[np.ndarray]:
    """Compile, load, and execute a registered custom op.

    `args` must be ordered to match the kernel's declared positional
    parameters. Mutable buffers and read-only tensors become graph inputs;
    scalars are inlined via `ops.constant`. Returns the post-execution
    numpy view of each `MutableBuf`, in the order they appear in `args`.
    """
    from max.driver import CPU
    from max.engine import InferenceSession
    from max.graph import BufferType, DeviceRef, Graph, TensorType, ops

    if device is None:
        device = pick_device()
    dev_ref = DeviceRef.from_device(device)
    cpu_ref = DeviceRef.from_device(CPU())

    # Partition: which args become graph inputs (in order) vs compile-time
    # constants. `graph_slots` records each graph input's slot in the final
    # positional `values` list so we can reassemble it inside `forward`.
    input_types: list = []
    graph_slots: list[int] = []
    for i, a in enumerate(args):
        if isinstance(a, (MutableBuf, ReadTensor)):
            shape = list(a.array.shape)
            t = max_dtype(a.dtype_name)
            input_types.append(
                BufferType(t, shape=shape, device=dev_ref)
                if isinstance(a, MutableBuf)
                else TensorType(t, shape=shape, device=dev_ref)
            )
            graph_slots.append(i)

    if not any(isinstance(a, MutableBuf) for a in args):
        # Non-inplace kernels would need `ops.custom` + output specs.
        # Add when the first such kernel lands.
        raise NotImplementedError(
            f"{kernel_name!r}: run_custom_op currently requires at least one "
            "MutableBuf (uses ops.inplace_custom). Extend the bridge when "
            "you add a kernel that returns outputs by value."
        )

    def forward(*graph_inputs):
        values: list = [None] * len(args)
        for slot, gv in zip(graph_slots, graph_inputs):
            values[slot] = gv
        for i, a in enumerate(args):
            if isinstance(a, ConstScalar):
                values[i] = ops.constant(
                    a.value, max_dtype(a.dtype_name), device=cpu_ref
                )
        ops.inplace_custom(name=kernel_name, device=dev_ref, values=values)
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

    # Upload graph inputs and remember each MutableBuf's buffer so we can
    # read it back post-execution.
    exec_args: list = []
    mutable_buffers: dict[int, object] = {}
    for slot in graph_slots:
        a = args[slot]
        # `graph_slots` only ever indexes MutableBuf/ReadTensor entries
        # (ConstScalar args are baked into the graph, not uploaded). Assert
        # the invariant to narrow the union for the type checker.
        assert isinstance(a, (MutableBuf, ReadTensor))
        buf = _to_device_buffer(a.array, a.dtype_name, device)
        exec_args.append(buf)
        if isinstance(a, MutableBuf):
            mutable_buffers[slot] = buf

    model.execute(*exec_args)

    return [
        _from_device_buffer(mutable_buffers[i], a.dtype_name)
        for i, a in enumerate(args)
        if isinstance(a, MutableBuf)
    ]


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
