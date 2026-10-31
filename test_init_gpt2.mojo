"""From-scratch weight-init equivalence: Mojo `-e d12` random init vs the PyTorch
reference's `GPT(d12)` init, tensor by tensor.

`make verify-init [ZERO_INIT_PROJ=0|1|2]` writes the reference checkpoint with
`train_gpt2.py --model d12` (env GPT2_ZERO_INIT_PROJ) into build/ref_init/ and
builds this test with `-D LLMM_ZERO_INIT_PROJ`, so both sides run the same mode.

Why a separate gate: `make verify` loads PRETRAINED gpt2_124M.bin, so
initialization never executes there and zero-init-projections could be wrong
(or the RNG stream shifted) without it noticing.

Checks, per parameter tensor: max |mojo - ref| <= INIT_TOL; and for the mode
under test the residual projections (attn_proj_weight, proj_weight) must be
exactly zero (mode 1) / non-zero with the right std (mode 0, 2), while every
other weight tensor must be non-degenerate (so "zero everything" cannot pass).
"""

from std.sys import argv, exit
from std.math import sqrt

from max.gpu.host import DeviceContext

from train_gpt2 import GPT2, Parameters, ZERO_INIT_PROJ, _parameter_name

comptime INIT_TOL = Float32(1e-6)


def main() raises:
    var args = argv()
    var ref_path = String("build/ref_init/gpt2_d12.bin")
    if len(args) > 1:
        ref_path = String(args[1])

    var ctx = DeviceContext(api="cpu")
    var mojo = GPT2["cpu", 1, False]("d12", rank=0, zero_stage=0, ctx=ctx)
    var ref_model = GPT2["cpu", 1, False](ref_path, rank=0, zero_stage=0, ctx=ctx)

    if mojo.num_parameters != ref_model.num_parameters:
        print("result gate=init status=fail reason=num_parameters",
              mojo.num_parameters, ref_model.num_parameters)
        exit(1)

    var ok = True
    var offset = 0
    var L = mojo.config.num_layer
    var res_std = Float32(0.02) / sqrt(Float32(2 * L))
    for t in range(len(mojo.param_sizes)):
        var n = mojo.param_sizes[t]
        var maxdiff = Float32(0.0)
        var nonzero_m = 0
        var nonzero_r = 0
        var sumsq_m = Float64(0.0)
        for i in range(n):
            var a = mojo.params_memory[unsafe_offset=offset + i].cast[DType.float32]()
            var b = ref_model.params_memory[unsafe_offset=offset + i].cast[DType.float32]()
            var d = abs(a - b)
            if d > maxdiff:
                maxdiff = d
            if a != 0:
                nonzero_m += 1
            if b != 0:
                nonzero_r += 1
            sumsq_m += Float64(a) * Float64(a)
        var rms = Float32(sqrt(sumsq_m / Float64(n)))
        var is_proj = (
            t == Parameters.attn_proj_weight or t == Parameters.proj_weight
        )
        var is_weight = is_proj or t == Parameters.wpe or t == Parameters.qkv_weight or t == Parameters.fc_weight or t == Parameters.wte
        var tensor_ok = maxdiff <= INIT_TOL
        # Planted-value structure checks (a tolerance alone would pass if both
        # sides were wrongly all-zero).
        if is_proj:
            comptime if ZERO_INIT_PROJ == 1:
                if nonzero_m != 0 or nonzero_r != 0:
                    tensor_ok = False
            else:
                var want = res_std
                comptime if ZERO_INIT_PROJ == 2:
                    want = Float32(0.002)
                if nonzero_m < n // 2 or abs(rms - want) > 0.05 * want:
                    tensor_ok = False
        elif is_weight:
            if nonzero_m < n // 2 or abs(rms - Float32(0.02)) > 0.001:
                tensor_ok = False
        print(
            "init tensor=" + String(_parameter_name(t)),
            "n=" + String(n),
            "maxdiff=" + String(maxdiff),
            "rms=" + String(rms),
            "nonzero=" + String(nonzero_m),
            "ref_nonzero=" + String(nonzero_r),
            "ok=" + String(tensor_ok),
        )
        if not tensor_ok:
            ok = False
        offset += n

    print(
        "result gate=init zero_init_proj=" + String(ZERO_INIT_PROJ)
        + " status=" + String("pass" if ok else "fail")
    )
    if not ok:
        exit(1)
