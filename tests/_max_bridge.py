"""MAX-graph bridge for tests of `@compiler.register`'d Mojo kernels.

This module is the generic harness only: `run_custom_op` builds a graph
from a typed list of positional arguments (`MutableBuf`, `ReadTensor`,
`ScalarArg`), compiles, executes, and returns post-step numpy views of
every mutable buffer.

Compiled models are cached on (kernel name, device, dtypes/ranks,
parameters) — tensor dims are symbolic graph dims and scalars are runtime
0-d tensor inputs rather than baked constants, so neither a new shape nor
a new scalar value recompiles: a multi-step trajectory (varying `t`) and a
multi-case shape sweep each compile exactly once per kernel/dtype. This is
what keeps the equivalence suite fast; don't switch scalars back to
`ops.constant` or dims back to concrete shapes without re-measuring
(symbolic-dim support verified against this MAX version by
~/Workspace/scripts/probe_symbolic_dims.py).

Per-kernel wrappers (e.g., `tests.kernels.adamw.step`) live under
`tests/kernels/` and just package their kernel's parameters into the right
arg list. Add new kernels there; do not add per-kernel boilerplate here.

## How MAX 26.5 / Mojo 1.0.0b3 delivers runtime args to a registered op

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

import hashlib
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Union

import numpy as np

if TYPE_CHECKING:
    from max.driver import Device
    from max.engine import InferenceSession, Model


# Source dir by default; `_ensure_packaged` reassigns this to a prebuilt
# .mojopkg before any compile so MAX never repackages the sources into its
# shared (and race-prone) temp mojo_pkg cache. Compiling custom_extensions
# from the SOURCE dir makes MAX repackage it into one shared temp .mojopkg
# (/var/folders/.../.modular_*/mojo_pkg/, content-hashed name) on every
# Graph build, rewritten non-atomically and read back immediately. That
# file is the root of the nondeterministic "Failed to compile the model"
# flakes: a single process can read its own half-written package, and
# concurrent pytest runs tear each other's down (observed corrupt "invalid
# magic bytes" leftovers that poison later runs until deleted). A prebuilt
# package is loaded directly — the temp dir is never created (verified).
MOJO_KERNELS_DIR = Path(__file__).resolve().parents[1] / "llmm"
_SOURCE_KERNELS_DIR = MOJO_KERNELS_DIR

# Persistent compiled-model cache (tests/.mef_cache/<fingerprint>/). A MAX
# model compile costs 4-10s per (kernel, dtype, parameters) and the suite
# triggers ~30 of them: that was ~90% of a cold suite's wall clock. A
# compiled model exported as MEF reloads in milliseconds with outputs
# verified bit-identical and symbolic dims intact
# (~/Workspace/scripts/probe_mef_cache.py), so each compile is paid once
# per kernel-source change, not once per pytest run. The fingerprint dir
# name hashes the llmm/*.mojo sources, the MAX version, the mojo binary,
# and _MEF_SCHEMA: editing a kernel lands in a fresh dir, and stale dirs
# are pruned. Set LLMM_DISABLE_MEF_CACHE=1 to force full recompiles.
# `mojo package` output is NOT bit-stable across identical sources
# (verified), hence hashing sources rather than the package.
_MEF_CACHE_ROOT = Path(__file__).resolve().parent / ".mef_cache"
_MEF_SCHEMA = 1  # bump when _compile_model's graph construction changes
_MEF_CACHE_DIR: "Path | None | Literal[False]" = False  # False = unresolved


def _mef_cache_dir() -> "Path | None":
    """Fingerprint-named cache dir for this kernel-source + toolchain state,
    or None when disabled / sources unlocatable. Resolved once per process;
    stale sibling fingerprints are pruned on first resolution."""
    global _MEF_CACHE_DIR
    if _MEF_CACHE_DIR is not False:
        return _MEF_CACHE_DIR
    _MEF_CACHE_DIR = None
    if os.environ.get("LLMM_DISABLE_MEF_CACHE"):
        return None
    sources = sorted(_SOURCE_KERNELS_DIR.rglob("*.mojo"))
    if not sources:
        return None
    h = hashlib.sha256()
    h.update(f"schema={_MEF_SCHEMA}".encode())
    try:
        from max import _core

        h.update(f"max={_core.__version__}".encode())
    except Exception:
        h.update(b"max=unknown")
    mojo = shutil.which("mojo")
    if mojo:
        st = Path(mojo).resolve().stat()
        h.update(f"mojo={st.st_size}:{st.st_mtime_ns}".encode())
    for f in sources:
        h.update(f.relative_to(_SOURCE_KERNELS_DIR).as_posix().encode())
        h.update(f.read_bytes())
    cache = _MEF_CACHE_ROOT / h.hexdigest()[:16]
    cache.mkdir(parents=True, exist_ok=True)
    for sibling in _MEF_CACHE_ROOT.iterdir():
        if sibling != cache:
            shutil.rmtree(sibling, ignore_errors=True)
    _MEF_CACHE_DIR = cache
    return cache


def _ensure_packaged(echo_warnings: bool = False) -> Path:
    """Build llmm.mojopkg once (per kernel-source state) and reuse it.

    Lazy: a fully MEF-cached run never compiles a graph, so it never pays
    for packaging either. The package lands in the fingerprint cache dir
    (reused across runs); with the cache disabled it falls back to a
    per-process temp dir. Written via temp file + os.replace so concurrent
    pytest processes can't read a half-written package.

    With echo_warnings (the `make build-mojo` path), mojo's compile
    warnings are forwarded to stderr instead of swallowed.
    """
    global MOJO_KERNELS_DIR
    if MOJO_KERNELS_DIR.suffix == ".mojopkg" and MOJO_KERNELS_DIR.exists():
        return MOJO_KERNELS_DIR
    cache = _mef_cache_dir()
    if cache is None:
        target = Path(tempfile.mkdtemp(prefix="llmm_pkg_")) / "llmm.mojopkg"
    else:
        target = cache / "llmm.mojopkg"
        if target.exists():
            MOJO_KERNELS_DIR = target
            return target
    # The package embeds its module name from the build-time FILENAME
    # (a `llmm.tmp123.mojopkg` build leaves kernels under `llmm.tmp123.*`,
    # which MAX's generated code then can't resolve), so the temp build
    # must be named exactly llmm.mojopkg; uniqueness comes from a
    # per-process scratch dir beside the target (same filesystem, so the
    # final os.replace stays atomic).
    scratch = target.parent / f".pkg_build{os.getpid()}"
    scratch.mkdir(parents=True, exist_ok=True)
    tmp = scratch / target.name
    try:
        proc = subprocess.run(
            ["mojo", "package", str(_SOURCE_KERNELS_DIR), "-o", str(tmp)],
            check=True,
            capture_output=True,
            text=True,
        )
        os.replace(tmp, target)
    except FileNotFoundError as e:
        raise RuntimeError(
            "`mojo` not on PATH; run the suite via `pixi run pytest` or "
            "activate the pixi env (see CLAUDE notes)."
        ) from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"mojo package failed:\n{e.stderr}") from e
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    if echo_warnings and proc.stderr:
        import sys

        noise = ("Crashpad",)
        for line in proc.stderr.splitlines():
            if not any(n in line for n in noise):
                print(line, file=sys.stderr)
    MOJO_KERNELS_DIR = target
    return target


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
# kernel name, device, and each arg's (kind, dtype, rank). Shapes and
# scalar *values* are deliberately absent — dims are symbolic and scalars
# are runtime inputs, so 65 optimizer steps and a multi-shape case sweep
# each hit one entry. Sessions are cached alongside models because a model
# is only valid while its session lives.
_MODEL_CACHE: dict[tuple, "Model"] = {}
_SESSION_CACHE: dict[tuple, "InferenceSession"] = {}
_DEFAULT_DEVICE: "Device | None" = None


def _signature(
    kernel_name: str, args: list[KernelArg], device, parameters: dict | None
) -> tuple:
    parts: list[tuple] = []
    for a in args:
        if isinstance(a, MutableBuf):
            parts.append(("mut", a.dtype_name, a.array.ndim))
        elif isinstance(a, ReadTensor):
            parts.append(("read", a.dtype_name, a.array.ndim))
        else:
            parts.append(("scalar", a.dtype_name))
    params = tuple(sorted((parameters or {}).items()))
    return (kernel_name, type(device).__name__, str(device), tuple(parts), params)


def _symbolic_shape(slot: int, arr: np.ndarray) -> list[str]:
    """All-distinct named dims for graph input `slot`. The kernels learn
    their true sizes at runtime (scalar args / .size()), so the graph needs
    no cross-input dim constraints — and shape-free input types are what
    let one compiled model serve every test-case shape (see _MODEL_CACHE)."""
    return [f"arg{slot}_dim{d}" for d in range(arr.ndim)]


def _compile_model(
    kernel_name: str, args: list[KernelArg], device, parameters: dict | None
):
    from max.driver import CPU
    from max.engine import InferenceSession
    from max.graph import BufferType, DeviceRef, Graph, TensorType, ops

    dev_ref = DeviceRef.from_device(device)
    cpu_ref = DeviceRef.from_device(CPU())

    # Every arg — tensor or scalar — is a graph input, in kernel order.
    # Tensor dims are symbolic (see _symbolic_shape) so the model binds
    # shapes at execute time. Scalars are 0-d tensors pinned to CPU
    # (mirroring where the old `ops.constant` scalars lived).
    input_types: list = []
    for slot, a in enumerate(args):
        t = max_dtype(a.dtype_name)
        if isinstance(a, MutableBuf):
            input_types.append(
                BufferType(t, shape=_symbolic_shape(slot, a.array), device=dev_ref)
            )
        elif isinstance(a, ReadTensor):
            input_types.append(
                TensorType(t, shape=_symbolic_shape(slot, a.array), device=dev_ref)
            )
        else:
            input_types.append(TensorType(t, shape=[], device=cpu_ref))

    def forward(*graph_inputs):
        ops.inplace_custom(
            name=kernel_name,
            device=dev_ref,
            values=list(graph_inputs),
            parameters=parameters or None,
        )
        return ()

    # Graph construction with `custom_extensions=[...]` triggers kernel
    # compilation, so wrap both Graph() and session.load() in the same
    # try/except — either can raise on a signature/kernel mismatch.
    try:
        graph = Graph(
            kernel_name,
            forward=forward,
            input_types=input_types,
            custom_extensions=[_ensure_packaged()],
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


def _load_cached_mef(
    kernel_name: str, key: tuple, device
) -> "tuple[InferenceSession | None, Model | None, Path | None]":
    """(session, model, mef_path) for `key` from the disk cache.

    model is None on a miss (corrupt or version-stale files load-fail and
    count as misses; the subsequent compile overwrites them). mef_path is
    where a freshly compiled model should be exported, or None when the
    cache is disabled.
    """
    cache = _mef_cache_dir()
    if cache is None:
        return None, None, None
    sig = hashlib.sha256(repr(key).encode()).hexdigest()[:16]
    mef = cache / f"{kernel_name}-{sig}.mef"
    if mef.exists():
        from max.engine import InferenceSession

        try:
            session = InferenceSession(devices=[device])
            return session, session.load(mef), mef
        except Exception:
            pass
    return None, None, mef


def _export_mef(model: "Model", mef: "Path | None") -> None:
    """Persist a freshly compiled model for future runs. Best-effort (the
    suite just recompiles next run if it fails); temp file + os.replace so
    concurrent pytest processes never see a partial write."""
    if mef is None:
        return
    tmp = mef.with_name(f"{mef.name}.tmp{os.getpid()}")
    try:
        model._export_mef(str(tmp))
        os.replace(tmp, mef)
    except Exception:
        tmp.unlink(missing_ok=True)


def run_custom_op(
    *,
    kernel_name: str,
    args: list[KernelArg],
    device: "Device | None" = None,
    parameters: dict | None = None,
) -> list[np.ndarray]:
    """Compile (cached), load, and execute a registered custom op.

    `args` must be ordered to match the kernel's declared positional
    parameters. Tensors and scalars alike become graph inputs, fed at
    execution time. Returns the post-execution numpy view of each
    `MutableBuf`, in the order they appear in `args`.

    `parameters` are compile-time op parameters (bool | int | str | DType)
    matched by name to comptime parameters the kernel's `execute` declares
    beyond dtype/target (verified against this MAX version by
    ~/Workspace/scripts/probe_graph_parameters.py). Each distinct value
    compiles (and caches) its own model.
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

    key = _signature(kernel_name, args, device, parameters)
    model = _MODEL_CACHE.get(key)
    if model is None:
        session, model, mef = _load_cached_mef(kernel_name, key, device)
        if model is None or session is None:
            try:
                session, model = _compile_model(kernel_name, args, device, parameters)
            except KernelSignatureMismatch:
                # MAX occasionally fails a load nondeterministically ("Failed
                # to compile the model ... should have been caught during
                # construction") on a graph that compiles fine seconds later.
                # One retry separates that flake from a real signature
                # mismatch, which fails identically both times.
                session, model = _compile_model(kernel_name, args, device, parameters)
            _export_mef(model, mef)
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


if __name__ == "__main__":
    # `python -m tests._max_bridge`: the build half of the test chain.
    # Builds (or reuses) llmm.mojopkg in the persistent cache and prints
    # its path, so `make build-mojo` produces exactly the artifact the
    # test step consumes; rerunning with unchanged sources is a no-op.
    print(_ensure_packaged(echo_warnings=True))
