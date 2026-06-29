from std.os import getenv
from std.python import Python
from std.sys import (
    argv,
    has_accelerator,
    has_apple_gpu_accelerator,
    simd_width_of,
)
from std.time import global_perf_counter_ns
from std.algorithm import sync_parallelize
from std.gpu.host.info import is_cpu, is_gpu
from std.gpu.host import DeviceContext, HostBuffer, DeviceBuffer
from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from std.memory import alloc, UnsafePointer, memcpy, memset_zero
from std.math import sqrt

from llmm.io import read_and_copy
from llmm.dataloader import DataLoader
from llmm.checkpointing import (
    CheckpointConfig,
    TrainingState,
    write_model_checkpoint,
    read_model_checkpoint,
    write_state_checkpoint,
    read_state_checkpoint,
    make_training_state,
    restore_dataloader_state,
)
from llmm.encoder import encoder_fwd, encoder_bwd, build_wte_buckets
from llmm.layernorm import (
    layernorm_fwd,
    layernorm_bwd,
    layernorm_fused_residual_fwd,
    layernorm_fused_residual_bwd,
)
from llmm.attention import attention_fwd, attention_bwd, KVCache, KVCachePtr
from llmm.matmul import matmul_fwd, matmul_bwd
from llmm.softmax import softmax_fwd, softmax_bwd
from llmm.crossentropy import crossentropy_ohe_fwd, crossentropy_ohe_bwd
from llmm.global_norm import global_norm_squared
from llmm.adamw import adamw_update, AdamWConfig
from llmm.fused_classifier import fused_classifier
from llmm.scheduler import LearningRateScheduler
from llmm.sampler import random_f32, sample_softmax
from llmm.tokenizer import Tokenizer, safe_print
from llmm.memory import (
    ImmutKernelPtr,
    ImmutMemPtr,
    MutMemPtr,
    as_immut_kernel,
    as_immut_kernel_from_mut,
    as_mut_kernel,
    rebind_mut_mem,
)
from llmm.zero import (
    ZeroContext,
    ShardedParameter,
    CpuCoordinator,
)


# ===----------------------------------------------------------------------=== #
# Constants and Comptime Variables
# ===----------------------------------------------------------------------=== #


from std.sys import get_defined_int

comptime WORLD_SIZE_DEF = get_defined_int["WORLD_SIZE", 1]()

comptime NUM_PARAMETER_TENSORS = 16
comptime NUM_ACTIVATION_TENSORS = 25
comptime GPT2_DTYPE = DType.float32
comptime GPT2_MAGIC = 20240520
comptime EPSILON = 1e-5
comptime UNROLL = 4


# ===----------------------------------------------------------------------=== #
# Parameter Tensors
# ===----------------------------------------------------------------------=== #


struct Parameters:
    comptime wte = 0
    comptime wpe = 1
    comptime ln_1_gamma = 2
    comptime ln_1_beta = 3
    comptime qkv_weight = 4
    comptime qkv_bias = 5
    comptime attn_proj_weight = 6
    comptime attn_proj_bias = 7
    comptime ln_2_gamma = 8
    comptime ln_2_beta = 9
    comptime fc_weight = 10
    comptime fc_bias = 11
    comptime proj_weight = 12
    comptime proj_bias = 13
    comptime ln_f_gamma = 14
    comptime ln_f_beta = 15


struct ParameterTensors[
    dtype: DType = DType.float32,
]:
    var params_memory: MutMemPtr[Self.dtype]

    # Encoder
    var wte: MutMemPtr[Self.dtype]  # (V, C)
    var wpe: MutMemPtr[Self.dtype]  # (max(T), C)

    # Layer Norm 1
    var ln_1_gamma: MutMemPtr[Self.dtype]  # (L, C)
    var ln_1_beta: MutMemPtr[Self.dtype]  # (L, C)

    # Attention
    var qkv_weight: MutMemPtr[Self.dtype]  # (L, 3*C, C)
    var qkv_bias: MutMemPtr[Self.dtype]  # (L, 3*C)
    var attn_proj_weight: MutMemPtr[Self.dtype]  # (L, C, C)
    var attn_proj_bias: MutMemPtr[Self.dtype]  # (L, C)

    # Layer Norm 2
    var ln_2_gamma: MutMemPtr[Self.dtype]  # (L, C)
    var ln_2_beta: MutMemPtr[Self.dtype]  # (L, C)

    # MLP
    var fc_weight: MutMemPtr[Self.dtype]  # (L, 4*C, C)
    var fc_bias: MutMemPtr[Self.dtype]  # (L, 4*C)
    var proj_weight: MutMemPtr[Self.dtype]  # (L, C, 4*C)
    var proj_bias: MutMemPtr[Self.dtype]  # (L, C)

    # Layer Norm Final
    var ln_f_gamma: MutMemPtr[Self.dtype]  # (C, )
    var ln_f_beta: MutMemPtr[Self.dtype]  # (C, )

    def __init__(out self):
        var zero = 0
        self.params_memory = MutMemPtr[Self.dtype](unsafe_from_address=zero)

        self.wte = MutMemPtr[Self.dtype](unsafe_from_address=zero)
        self.wpe = MutMemPtr[Self.dtype](unsafe_from_address=zero)
        self.ln_1_gamma = MutMemPtr[Self.dtype](unsafe_from_address=zero)
        self.ln_1_beta = MutMemPtr[Self.dtype](unsafe_from_address=zero)
        self.qkv_weight = MutMemPtr[Self.dtype](unsafe_from_address=zero)
        self.qkv_bias = MutMemPtr[Self.dtype](unsafe_from_address=zero)
        self.attn_proj_weight = MutMemPtr[Self.dtype](unsafe_from_address=zero)
        self.attn_proj_bias = MutMemPtr[Self.dtype](unsafe_from_address=zero)
        self.ln_2_gamma = MutMemPtr[Self.dtype](unsafe_from_address=zero)
        self.ln_2_beta = MutMemPtr[Self.dtype](unsafe_from_address=zero)
        self.fc_weight = MutMemPtr[Self.dtype](unsafe_from_address=zero)
        self.fc_bias = MutMemPtr[Self.dtype](unsafe_from_address=zero)
        self.proj_weight = MutMemPtr[Self.dtype](unsafe_from_address=zero)
        self.proj_bias = MutMemPtr[Self.dtype](unsafe_from_address=zero)
        self.ln_f_gamma = MutMemPtr[Self.dtype](unsafe_from_address=zero)
        self.ln_f_beta = MutMemPtr[Self.dtype](unsafe_from_address=zero)

    def point_parameters(
        mut self,
        param_sizes: List[Int],
        params_memory: MutMemPtr[Self.dtype],
    ) -> None:
        self.params_memory = params_memory

        comptime ParamPtr = MutMemPtr[Self.dtype]
        comptime ParamPtrPtr = UnsafePointer[ParamPtr, MutAnyOrigin]

        var ptrs = List[ParamPtrPtr]()
        ptrs.append(ParamPtrPtr(to=self.wte))
        ptrs.append(ParamPtrPtr(to=self.wpe))
        ptrs.append(ParamPtrPtr(to=self.ln_1_gamma))
        ptrs.append(ParamPtrPtr(to=self.ln_1_beta))
        ptrs.append(ParamPtrPtr(to=self.qkv_weight))
        ptrs.append(ParamPtrPtr(to=self.qkv_bias))
        ptrs.append(ParamPtrPtr(to=self.attn_proj_weight))
        ptrs.append(ParamPtrPtr(to=self.attn_proj_bias))
        ptrs.append(ParamPtrPtr(to=self.ln_2_gamma))
        ptrs.append(ParamPtrPtr(to=self.ln_2_beta))
        ptrs.append(ParamPtrPtr(to=self.fc_weight))
        ptrs.append(ParamPtrPtr(to=self.fc_bias))
        ptrs.append(ParamPtrPtr(to=self.proj_weight))
        ptrs.append(ParamPtrPtr(to=self.proj_bias))
        ptrs.append(ParamPtrPtr(to=self.ln_f_gamma))
        ptrs.append(ParamPtrPtr(to=self.ln_f_beta))

        var params_memory_iterator = self.params_memory

        for i in range(len(param_sizes)):
            ptrs[i][] = params_memory_iterator
            params_memory_iterator += param_sizes[i]


# ===----------------------------------------------------------------------=== #
# Activation Tensors
# ===----------------------------------------------------------------------=== #


struct Activations:
    comptime encoded = 0
    comptime ln_1 = 1
    comptime ln_1_mean = 2
    comptime ln_1_rstd = 3
    comptime qkv = 4
    comptime lse = 5
    comptime attn = 6
    comptime attn_proj = 7
    comptime residual_2 = 8
    comptime ln_2 = 9
    comptime ln_2_mean = 10
    comptime ln_2_rstd = 11
    comptime fch = 12
    comptime fch_gelu = 13
    comptime fc_proj = 14
    comptime residual_3 = 15
    comptime ln_f = 16
    comptime ln_f_mean = 17
    comptime ln_f_rstd = 18
    comptime logits = 19
    comptime losses = 20
    comptime q = 21
    comptime k = 22
    comptime v = 23
    comptime attn_merged = 24


struct ActivationTensors[
    dtype: DType = DType.float32,
]:
    # Encoder
    var encoded: MutMemPtr[Self.dtype]  # (B, T, C)

    # Layer Norm 1
    var ln_1: MutMemPtr[Self.dtype]  # (L, B, T, C)
    var ln_1_mean: MutMemPtr[Self.dtype]  # (L, B, T)
    var ln_1_rstd: MutMemPtr[Self.dtype]  # (L, B, T)

    # Attention
    var qkv: MutMemPtr[Self.dtype]  # (L, B, T, 3*C)
    var lse: MutMemPtr[Self.dtype]  # (L, B, NH, T)
    var attn: MutMemPtr[Self.dtype]  # (L, B, T, C)
    var attn_proj: MutMemPtr[Self.dtype]  # (L, B, T, C)
    var residual_2: MutMemPtr[Self.dtype]  # (L, B, T, C)

    # Layer Norm 2
    var ln_2: MutMemPtr[Self.dtype]  # (L, B, T, C)
    var ln_2_mean: MutMemPtr[Self.dtype]  # (L, B, T)
    var ln_2_rstd: MutMemPtr[Self.dtype]  # (L, B, T)

    # MLP
    var fch: MutMemPtr[Self.dtype]  # (L, B, T, 4*C)
    var fch_gelu: MutMemPtr[Self.dtype]  # (L, B, T, 4*C)
    var fc_proj: MutMemPtr[Self.dtype]  # (L, B, T, C)
    var residual_3: MutMemPtr[Self.dtype]  # (L, B, T, C)

    # Layer Norm Final
    var ln_f: MutMemPtr[Self.dtype]  # (B, T, C)
    var ln_f_mean: MutMemPtr[Self.dtype]  # (B, T)
    var ln_f_rstd: MutMemPtr[Self.dtype]  # (B, T)
    var logits: MutMemPtr[Self.dtype]  # (B, T, V)
    var losses: MutMemPtr[Self.dtype]  # (B, T)

    # Scratch / Split-attention
    var q: MutMemPtr[Self.dtype]  # (L, B, NH, T, HS)
    var k: MutMemPtr[Self.dtype]  # (L, B, NH, T, HS)
    var v: MutMemPtr[Self.dtype]  # (L, B, NH, T, HS)
    var attn_merged: MutMemPtr[Self.dtype]  # (L, B, T, C)

    def __init__(out self):
        var zero = 0
        self.encoded = MutMemPtr[Self.dtype](unsafe_from_address=zero)
        self.ln_1 = MutMemPtr[Self.dtype](unsafe_from_address=zero)
        self.ln_1_mean = MutMemPtr[Self.dtype](unsafe_from_address=zero)
        self.ln_1_rstd = MutMemPtr[Self.dtype](unsafe_from_address=zero)
        self.qkv = MutMemPtr[Self.dtype](unsafe_from_address=zero)
        self.lse = MutMemPtr[Self.dtype](unsafe_from_address=zero)
        self.attn = MutMemPtr[Self.dtype](unsafe_from_address=zero)
        self.attn_proj = MutMemPtr[Self.dtype](unsafe_from_address=zero)
        self.residual_2 = MutMemPtr[Self.dtype](unsafe_from_address=zero)
        self.ln_2 = MutMemPtr[Self.dtype](unsafe_from_address=zero)
        self.ln_2_mean = MutMemPtr[Self.dtype](unsafe_from_address=zero)
        self.ln_2_rstd = MutMemPtr[Self.dtype](unsafe_from_address=zero)
        self.fch = MutMemPtr[Self.dtype](unsafe_from_address=zero)
        self.fch_gelu = MutMemPtr[Self.dtype](unsafe_from_address=zero)
        self.fc_proj = MutMemPtr[Self.dtype](unsafe_from_address=zero)
        self.residual_3 = MutMemPtr[Self.dtype](unsafe_from_address=zero)
        self.ln_f = MutMemPtr[Self.dtype](unsafe_from_address=zero)
        self.ln_f_mean = MutMemPtr[Self.dtype](unsafe_from_address=zero)
        self.ln_f_rstd = MutMemPtr[Self.dtype](unsafe_from_address=zero)
        self.logits = MutMemPtr[Self.dtype](unsafe_from_address=zero)
        self.losses = MutMemPtr[Self.dtype](unsafe_from_address=zero)
        self.q = MutMemPtr[Self.dtype](unsafe_from_address=zero)
        self.k = MutMemPtr[Self.dtype](unsafe_from_address=zero)
        self.v = MutMemPtr[Self.dtype](unsafe_from_address=zero)
        self.attn_merged = MutMemPtr[Self.dtype](unsafe_from_address=zero)

    def point_activations(
        mut self,
        activation_sizes: List[Int],
        activations_memory: MutMemPtr[Self.dtype],
    ) -> None:
        comptime ActPtr = MutMemPtr[Self.dtype]
        comptime ActPtrPtr = UnsafePointer[ActPtr, MutAnyOrigin]

        var ptrs = List[ActPtrPtr]()
        ptrs.append(ActPtrPtr(to=self.encoded))
        ptrs.append(ActPtrPtr(to=self.ln_1))
        ptrs.append(ActPtrPtr(to=self.ln_1_mean))
        ptrs.append(ActPtrPtr(to=self.ln_1_rstd))
        ptrs.append(ActPtrPtr(to=self.qkv))
        ptrs.append(ActPtrPtr(to=self.lse))
        ptrs.append(ActPtrPtr(to=self.attn))
        ptrs.append(ActPtrPtr(to=self.attn_proj))
        ptrs.append(ActPtrPtr(to=self.residual_2))
        ptrs.append(ActPtrPtr(to=self.ln_2))
        ptrs.append(ActPtrPtr(to=self.ln_2_mean))
        ptrs.append(ActPtrPtr(to=self.ln_2_rstd))
        ptrs.append(ActPtrPtr(to=self.fch))
        ptrs.append(ActPtrPtr(to=self.fch_gelu))
        ptrs.append(ActPtrPtr(to=self.fc_proj))
        ptrs.append(ActPtrPtr(to=self.residual_3))
        ptrs.append(ActPtrPtr(to=self.ln_f))
        ptrs.append(ActPtrPtr(to=self.ln_f_mean))
        ptrs.append(ActPtrPtr(to=self.ln_f_rstd))
        ptrs.append(ActPtrPtr(to=self.logits))
        ptrs.append(ActPtrPtr(to=self.losses))
        ptrs.append(ActPtrPtr(to=self.q))
        ptrs.append(ActPtrPtr(to=self.k))
        ptrs.append(ActPtrPtr(to=self.v))
        ptrs.append(ActPtrPtr(to=self.attn_merged))

        var activations_memory_iterator = activations_memory
        for i in range(len(activation_sizes)):
            ptrs[i][] = activations_memory_iterator
            activations_memory_iterator += activation_sizes[i]


# ===----------------------------------------------------------------------=== #
# GPT-2 Model
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct GPT2Config:
    var max_seq_len: Int  # Max sequence length (e.g. 1024).
    var vocab_size: Int  # Vocab size (e.g. 50257).
    var num_layer: Int  # Number of layers (e.g. 12).
    var num_heads: Int  # Number of heads (e.g. 12).
    var channels: Int  # Number of channels (e.g. 768).
    var padded_vocab_size: Int  # Padded vocab size (%128 == 0, e.g. 50304).


# `recompute` enables activation (gradient) checkpointing: when True, the
# per-layer MLP activations fch (pre-GELU) and fch_gelu (post-GELU) are not
# persisted across the forward/backward boundary. They collapse to a single
# scratch slot that the forward fills and the backward rematerializes by
# re-running the same fused FC matmul (matmul_fwd[use_gelu=True]) from the
# still-resident ln_2. This reuses the forward's fused-GELU kernel and drops the
# two largest per-layer activation buffers (each B*T*4C), trading one FC GEMM
# per layer in backward for the memory. That activation-memory cut is what makes
# ZeRO-3 tractable (ZeRO-3 shards params/grads/optimizer but not activations).
struct GPT2[target: StaticString, WORLD_SIZE: Int = 1, recompute: Bool = False]:
    var ctx: DeviceContext
    var config: GPT2Config

    # Buffer objects managing physical memory lifetimes
    var params_buf: DeviceBuffer[GPT2_DTYPE]
    var grads_buf: DeviceBuffer[GPT2_DTYPE]
    var m_buf: DeviceBuffer[GPT2_DTYPE]
    var v_buf: DeviceBuffer[GPT2_DTYPE]
    var acts_buf: DeviceBuffer[GPT2_DTYPE]
    var grad_acts_buf: DeviceBuffer[GPT2_DTYPE]
    var inputs_buf: HostBuffer[DType.int32]
    var targets_buf: HostBuffer[DType.int32]
    var bucket_info_buf: HostBuffer[DType.int32]
    var workload_indices_buf: HostBuffer[DType.int32]
    var losses_host_buf: HostBuffer[GPT2_DTYPE]
    var logits_host_buf: HostBuffer[GPT2_DTYPE]

    # Model weights and their sizes
    var params: ParameterTensors[GPT2_DTYPE]
    var param_sizes: List[Int]
    var params_memory: MutMemPtr[GPT2_DTYPE]
    var num_parameters: Int

    # Gradients of the weights
    var grads: ParameterTensors[GPT2_DTYPE]
    var grads_memory: MutMemPtr[GPT2_DTYPE]

    # Buffers for AdamW
    var m_memory: MutMemPtr[GPT2_DTYPE]
    var v_memory: MutMemPtr[GPT2_DTYPE]

    # Activations of the model, and their sizes
    var acts: ActivationTensors[GPT2_DTYPE]
    var act_sizes: List[Int]
    var acts_memory: MutMemPtr[GPT2_DTYPE]
    var num_activations: Int

    # Gradients of the activations
    var grad_acts: ActivationTensors[GPT2_DTYPE]
    var grad_acts_memory: MutMemPtr[GPT2_DTYPE]
    var num_grads: Int

    # Runstate Configurations
    var batch_size: Int  # The batch size of the current forward pass (Our B).
    var seq_len: Int  # The sequence length of the current forward pass (Our T).
    var inputs: MutMemPtr[
        DType.int32
    ]  # The input tokens for the current forward pass.
    var targets: MutMemPtr[
        DType.int32
    ]  # The target tokens for the current forward pass.
    var bucket_info: MutMemPtr[DType.int32]
    var workload_indices: MutMemPtr[DType.int32]
    var num_wte_buckets: Int
    var wte_bucket_capacity: Int
    var mean_loss: Float32  # The mean loss for the current forward pass.
    var checkpoint_path: String  # The path to the checkpoint file.

    # Memory allocation tracking flags
    var has_allocated_params: Bool
    var has_allocated_acts: Bool
    var has_allocated_grads: Bool
    var has_allocated_optimizer_moments: Bool

    # Cache management
    var kv_cache: KVCache

    # Sharding and parallelism
    var zero_ctx: ZeroContext[Self.target, Self.WORLD_SIZE]
    var optimizer_num_parameters: Int
    var padded_num_parameters: Int
    var sharded_grads_buf: DeviceBuffer[GPT2_DTYPE]
    var sharded_grads_memory: MutMemPtr[GPT2_DTYPE]

    def __init__(
        out self,
        checkpoint_path: String,
        rank: Int,
        zero_stage: Int,
        ctx: DeviceContext,
        cpu_coordinator_ptr: Optional[
            UnsafePointer[CpuCoordinator, MutUntrackedOrigin]
        ] = None,
    ) raises:
        self.ctx = ctx
        self.checkpoint_path = checkpoint_path
        self.has_allocated_params = False
        self.has_allocated_acts = False
        self.has_allocated_grads = False
        self.has_allocated_optimizer_moments = False
        self.kv_cache = KVCache()
        self.zero_ctx = ZeroContext[Self.target, Self.WORLD_SIZE](
            rank, zero_stage, ctx, cpu_coordinator_ptr
        )
        self.optimizer_num_parameters = 0
        self.padded_num_parameters = 0

        self.config = GPT2Config(
            max_seq_len=0,
            vocab_size=0,
            num_layer=0,
            num_heads=0,
            channels=0,
            padded_vocab_size=0,
        )

        var zero = 0
        var NULL_DTYPE_PTR = MutMemPtr[GPT2_DTYPE](unsafe_from_address=zero)
        var NULL_INT32_PTR = MutMemPtr[DType.int32](unsafe_from_address=zero)

        self.sharded_grads_buf = self.ctx.enqueue_create_buffer[GPT2_DTYPE](1)
        self.sharded_grads_memory = NULL_DTYPE_PTR

        self.params_memory = NULL_DTYPE_PTR
        self.grads_memory = NULL_DTYPE_PTR
        self.m_memory = NULL_DTYPE_PTR
        self.v_memory = NULL_DTYPE_PTR
        self.acts_memory = NULL_DTYPE_PTR
        self.grad_acts_memory = NULL_DTYPE_PTR
        self.inputs = NULL_INT32_PTR
        self.targets = NULL_INT32_PTR
        self.bucket_info = NULL_INT32_PTR
        self.workload_indices = NULL_INT32_PTR
        self.num_wte_buckets = 0
        self.wte_bucket_capacity = 0

        self.num_parameters = 0
        self.num_activations = 0
        self.num_grads = 0
        self.batch_size = 0
        self.seq_len = 0
        self.mean_loss = -1.0

        self.params = ParameterTensors[GPT2_DTYPE]()
        self.grads = ParameterTensors[GPT2_DTYPE]()
        self.acts = ActivationTensors[GPT2_DTYPE]()
        self.grad_acts = ActivationTensors[GPT2_DTYPE]()

        self.params_buf = self.ctx.enqueue_create_buffer[GPT2_DTYPE](1)
        self.grads_buf = self.ctx.enqueue_create_buffer[GPT2_DTYPE](1)
        self.m_buf = self.ctx.enqueue_create_buffer[GPT2_DTYPE](1)
        self.v_buf = self.ctx.enqueue_create_buffer[GPT2_DTYPE](1)
        self.acts_buf = self.ctx.enqueue_create_buffer[GPT2_DTYPE](1)
        self.grad_acts_buf = self.ctx.enqueue_create_buffer[GPT2_DTYPE](1)
        self.inputs_buf = self.ctx.enqueue_create_host_buffer[DType.int32](1)
        self.targets_buf = self.ctx.enqueue_create_host_buffer[DType.int32](1)
        self.bucket_info_buf = self.ctx.enqueue_create_host_buffer[DType.int32](
            1
        )
        self.workload_indices_buf = self.ctx.enqueue_create_host_buffer[
            DType.int32
        ](1)
        self.losses_host_buf = self.ctx.enqueue_create_host_buffer[GPT2_DTYPE](
            1
        )
        self.logits_host_buf = self.ctx.enqueue_create_host_buffer[GPT2_DTYPE](
            1
        )
        self.ctx.synchronize()

        self.param_sizes = List[Int]()
        for _ in range(NUM_PARAMETER_TENSORS):
            self.param_sizes.append(0)

        self.act_sizes = List[Int]()
        for _ in range(NUM_ACTIVATION_TENSORS):
            self.act_sizes.append(0)

        var model_file = open(self.checkpoint_path, "r")
        var model_header = alloc[Int32](256)
        read_and_copy[DType.int32](model_file, model_header, 256)

        if model_header.load(0) != GPT2_MAGIC:
            print("Bad magic number in header: " + String(model_header.load(0)))
            model_header.free()
            raise Error("GPT2 error: Invalid magic number in header")
        if model_header.load(1) != 3:
            print("Bad version in header: " + String(model_header.load(1)))
            model_header.free()
            raise Error("GPT2 error: Invalid version in header")

        self.config = GPT2Config(
            max_seq_len=Int(model_header.load(2)),
            vocab_size=Int(model_header.load(3)),
            num_layer=Int(model_header.load(4)),
            num_heads=Int(model_header.load(5)),
            channels=Int(model_header.load(6)),
            padded_vocab_size=Int(model_header.load(7)),
        )
        model_header.free()

        # Allocate parameters.
        self.allocate_parameters(model_file)

        # Allocate weight gradients.
        self.allocate_gradients()

        # Allocate optimizer moments.
        self.allocate_optimizer_moments()

        # Init Activations and activation gradients (allocated dynamically per batch).
        self.acts = ActivationTensors[GPT2_DTYPE]()
        self.acts_memory = MutMemPtr[GPT2_DTYPE](unsafe_from_address=zero)
        self.num_activations = 0

        self.grad_acts = ActivationTensors[GPT2_DTYPE]()
        self.grad_acts_memory = MutMemPtr[GPT2_DTYPE](unsafe_from_address=zero)

        self.batch_size = 0
        self.seq_len = 0
        self.mean_loss = -1.0  # -1.0 designates no loss has been computed yet.
        self.inputs = MutMemPtr[DType.int32](unsafe_from_address=zero)
        self.targets = MutMemPtr[DType.int32](unsafe_from_address=zero)

        # Print out the model summary.
        print("Model Summary:")
        print("--------------------------------")
        print("Model Name: GPT-2")
        print("Model Version: 3")
        print("Model Magic Number: " + String(GPT2_MAGIC))
        print("Model Config:")
        print("--------------------------------")
        print("Max Sequence Length: " + String(self.config.max_seq_len))
        print("Vocab Size: " + String(self.config.vocab_size))
        print("Number of Layers: " + String(self.config.num_layer))
        print("Number of Heads: " + String(self.config.num_heads))
        print("Number of Channels: " + String(self.config.channels))
        print("Padded Vocab Size: " + String(self.config.padded_vocab_size))
        print("--------------------------------")
        print("Number of Parameters: " + String(self.num_parameters))
        print("--------------------------------")
        print("Number of Activations: " + String(self.num_activations))
        print("--------------------------------")
        print("Number of Gradients: " + String(self.num_grads))
        print("--------------------------------")

    def allocate_parameters(mut self, mut model_file: FileHandle) raises:
        var max_T = self.config.max_seq_len
        var V = self.config.vocab_size
        var L = self.config.num_layer
        var NH = self.config.num_heads
        var C = self.config.channels
        var V_p = self.config.padded_vocab_size

        # Alloc space for all the parameters then read them in.
        self.param_sizes[Parameters.wte] = V_p * C
        self.param_sizes[Parameters.wpe] = max_T * C
        self.param_sizes[Parameters.ln_1_gamma] = L * C
        self.param_sizes[Parameters.ln_1_beta] = L * C
        self.param_sizes[Parameters.qkv_weight] = L * (3 * C) * C
        self.param_sizes[Parameters.qkv_bias] = L * (3 * C)
        self.param_sizes[Parameters.attn_proj_weight] = L * C * C
        self.param_sizes[Parameters.attn_proj_bias] = L * C
        self.param_sizes[Parameters.ln_2_gamma] = L * C
        self.param_sizes[Parameters.ln_2_beta] = L * C
        self.param_sizes[Parameters.fc_weight] = L * (4 * C) * C
        self.param_sizes[Parameters.fc_bias] = L * (4 * C)
        self.param_sizes[Parameters.proj_weight] = L * C * (4 * C)
        self.param_sizes[Parameters.proj_bias] = L * C
        self.param_sizes[Parameters.ln_f_gamma] = C
        self.param_sizes[Parameters.ln_f_beta] = C

        # Count the total number of parameters.
        var num_parameters = 0
        for i in range(NUM_PARAMETER_TENSORS):
            num_parameters += self.param_sizes[i]
        self.num_parameters = num_parameters

        # Calculate optimizer sharding parameters
        comptime if Self.WORLD_SIZE > 1:
            # All zero stages 1/2/3 shard the optimizer parameters.
            if self.zero_ctx.zero_stage >= 1:
                self.optimizer_num_parameters = (
                    self.num_parameters + Self.WORLD_SIZE - 1
                ) // Self.WORLD_SIZE
                self.padded_num_parameters = (
                    self.optimizer_num_parameters * Self.WORLD_SIZE
                )
            else:
                self.optimizer_num_parameters = self.num_parameters
                self.padded_num_parameters = self.num_parameters
        else:
            self.optimizer_num_parameters = self.num_parameters
            self.padded_num_parameters = self.num_parameters

        # Allocate parameters using device context host buffer.
        self.params = ParameterTensors[GPT2_DTYPE]()
        var temp_host_buf = self.ctx.enqueue_create_host_buffer[GPT2_DTYPE](
            self.num_parameters
        )
        self.ctx.synchronize()
        var temp_ptr = rebind_mut_mem[GPT2_DTYPE](
            temp_host_buf.unsafe_ptr().as_unsafe_any_origin()
        )
        read_and_copy[GPT2_DTYPE](model_file, temp_ptr, self.num_parameters)
        model_file.close()

        self.params_buf = self.ctx.enqueue_create_buffer[GPT2_DTYPE](
            self.padded_num_parameters
        )

        self.params_buf.enqueue_fill(Scalar[GPT2_DTYPE](0.0))
        self.ctx.enqueue_copy(
            dst_ptr=rebind[UnsafePointer[Scalar[GPT2_DTYPE], MutAnyOrigin]](
                self.params_buf.unsafe_ptr().as_unsafe_any_origin()
            ),
            src_ptr=rebind[UnsafePointer[Scalar[GPT2_DTYPE], ImmutAnyOrigin]](
                temp_ptr.as_unsafe_any_origin()
            ),
            size=self.num_parameters,
        )
        self.ctx.synchronize()
        self.params_memory = rebind_mut_mem[GPT2_DTYPE](
            self.params_buf.unsafe_ptr().as_unsafe_any_origin()
        )
        self.params.point_parameters(self.param_sizes, self.params_memory)

        self.has_allocated_params = True

    def allocate_gradients(mut self) raises:
        self.grads = ParameterTensors[GPT2_DTYPE]()
        self.grads_buf = self.ctx.enqueue_create_buffer[GPT2_DTYPE](
            self.padded_num_parameters
        )
        self.grads_buf.enqueue_fill(Scalar[GPT2_DTYPE](0.0))
        self.ctx.synchronize()
        self.grads_memory = rebind_mut_mem[GPT2_DTYPE](
            self.grads_buf.unsafe_ptr().as_unsafe_any_origin()
        )
        self.grads.point_parameters(self.param_sizes, self.grads_memory)
        self.num_grads = self.num_parameters

        comptime if Self.WORLD_SIZE > 1:
            # ZeRO-2/3 shards the gradient communication (reduce-scatter).
            # ZeRO-1 uses allreduce so gradients stay fully replicated in grads_memory;
            # the optimizer reads grads_memory + rank*opt directly, no shard buffer needed.
            if self.zero_ctx.zero_stage >= 2:
                self.sharded_grads_buf = self.ctx.enqueue_create_buffer[
                    GPT2_DTYPE
                ](self.optimizer_num_parameters)
                self.sharded_grads_buf.enqueue_fill(Scalar[GPT2_DTYPE](0.0))
                self.ctx.synchronize()
                self.sharded_grads_memory = rebind_mut_mem[GPT2_DTYPE](
                    self.sharded_grads_buf.unsafe_ptr().as_unsafe_any_origin()
                )
            else:
                # ZeRO-0 (DDP) and ZeRO-1 keep gradients fully replicated;
                # no sharded_grads_buf is needed.
                self.sharded_grads_buf = self.ctx.enqueue_create_buffer[
                    GPT2_DTYPE
                ](1)
                var zero = 0
                self.sharded_grads_memory = MutMemPtr[GPT2_DTYPE](
                    unsafe_from_address=zero
                )
        else:
            self.sharded_grads_buf = self.ctx.enqueue_create_buffer[GPT2_DTYPE](
                1
            )
            var zero = 0
            self.sharded_grads_memory = MutMemPtr[GPT2_DTYPE](
                unsafe_from_address=zero
            )

        self.has_allocated_grads = True

    def allocate_optimizer_moments(mut self) raises:
        self.m_buf = self.ctx.enqueue_create_buffer[GPT2_DTYPE](
            self.optimizer_num_parameters
        )
        self.v_buf = self.ctx.enqueue_create_buffer[GPT2_DTYPE](
            self.optimizer_num_parameters
        )
        self.m_buf.enqueue_fill(Scalar[GPT2_DTYPE](0.0))
        self.v_buf.enqueue_fill(Scalar[GPT2_DTYPE](0.0))
        self.ctx.synchronize()
        self.m_memory = rebind_mut_mem[GPT2_DTYPE](
            self.m_buf.unsafe_ptr().as_unsafe_any_origin()
        )
        self.v_memory = rebind_mut_mem[GPT2_DTYPE](
            self.v_buf.unsafe_ptr().as_unsafe_any_origin()
        )
        self.has_allocated_optimizer_moments = True

    def allocate_activations(mut self, batch_size: Int, seq_len: Int) raises:
        self.batch_size = batch_size
        self.seq_len = seq_len

        var B = batch_size
        var T = seq_len
        var C = self.config.channels
        var V_p = self.config.padded_vocab_size
        var L = self.config.num_layer
        var NH = self.config.num_heads

        self.act_sizes[Activations.encoded] = B * T * C
        self.act_sizes[Activations.ln_1] = L * B * T * C
        self.act_sizes[Activations.ln_1_mean] = L * B * T
        self.act_sizes[Activations.ln_1_rstd] = L * B * T
        self.act_sizes[Activations.qkv] = L * B * T * 3 * C
        self.act_sizes[Activations.lse] = L * B * NH * T
        self.act_sizes[Activations.attn] = L * B * T * C
        self.act_sizes[Activations.attn_proj] = L * B * T * C
        self.act_sizes[Activations.residual_2] = L * B * T * C
        self.act_sizes[Activations.ln_2] = L * B * T * C
        self.act_sizes[Activations.ln_2_mean] = L * B * T
        self.act_sizes[Activations.ln_2_rstd] = L * B * T
        # With activation recompute on, fch and fch_gelu are not persisted per
        # layer: they collapse to a single-layer scratch slot that forward fills
        # and backward rematerializes via the fused FC matmul (see `recompute`).
        var fch_layers = L
        comptime if Self.recompute:
            fch_layers = 1
        self.act_sizes[Activations.fch] = fch_layers * B * T * 4 * C
        self.act_sizes[Activations.fch_gelu] = fch_layers * B * T * 4 * C
        self.act_sizes[Activations.fc_proj] = L * B * T * C
        self.act_sizes[Activations.residual_3] = L * B * T * C
        self.act_sizes[Activations.ln_f] = B * T * C
        self.act_sizes[Activations.ln_f_mean] = B * T
        self.act_sizes[Activations.ln_f_rstd] = B * T
        self.act_sizes[Activations.logits] = B * T * V_p
        self.act_sizes[Activations.losses] = B * T
        self.act_sizes[Activations.q] = L * B * T * C
        self.act_sizes[Activations.k] = L * B * T * C
        self.act_sizes[Activations.v] = L * B * T * C
        self.act_sizes[Activations.attn_merged] = L * B * T * C

        # Count total activations
        var num_activations = 0
        for i in range(NUM_ACTIVATION_TENSORS):
            num_activations += self.act_sizes[i]
        self.num_activations = num_activations

        print("Number of Activations: " + String(self.num_activations))

        # Re-allocate device memory and point the structures
        self.acts_buf = self.ctx.enqueue_create_buffer[GPT2_DTYPE](
            self.num_activations
        )
        self.grad_acts_buf = self.ctx.enqueue_create_buffer[GPT2_DTYPE](
            self.num_activations
        )

        self.inputs_buf = self.ctx.enqueue_create_host_buffer[DType.int32](
            self.batch_size * self.seq_len
        )
        self.targets_buf = self.ctx.enqueue_create_host_buffer[DType.int32](
            self.batch_size * self.seq_len
        )

        # Encoder backward scratch: one bucket per (token, channel_group).
        var wte_c_per_warp = 128
        var num_channel_groups = (C + wte_c_per_warp - 1) // wte_c_per_warp
        self.wte_bucket_capacity = B * T * num_channel_groups
        self.bucket_info_buf = self.ctx.enqueue_create_host_buffer[DType.int32](
            self.wte_bucket_capacity * 4
        )
        self.workload_indices_buf = self.ctx.enqueue_create_host_buffer[
            DType.int32
        ](B * T)

        self.losses_host_buf = self.ctx.enqueue_create_host_buffer[GPT2_DTYPE](
            self.batch_size * self.seq_len
        )
        self.logits_host_buf = self.ctx.enqueue_create_host_buffer[GPT2_DTYPE](
            self.config.padded_vocab_size
        )

        self.ctx.synchronize()

        self.acts_memory = rebind_mut_mem[GPT2_DTYPE](
            self.acts_buf.unsafe_ptr().as_unsafe_any_origin()
        )
        self.grad_acts_memory = rebind_mut_mem[GPT2_DTYPE](
            self.grad_acts_buf.unsafe_ptr().as_unsafe_any_origin()
        )
        self.inputs = rebind_mut_mem[DType.int32](
            self.inputs_buf.unsafe_ptr().as_unsafe_any_origin()
        )
        self.targets = rebind_mut_mem[DType.int32](
            self.targets_buf.unsafe_ptr().as_unsafe_any_origin()
        )
        self.bucket_info = rebind_mut_mem[DType.int32](
            self.bucket_info_buf.unsafe_ptr().as_unsafe_any_origin()
        )
        self.workload_indices = rebind_mut_mem[DType.int32](
            self.workload_indices_buf.unsafe_ptr().as_unsafe_any_origin()
        )

        self.acts.point_activations(self.act_sizes, self.acts_memory)
        self.grad_acts.point_activations(self.act_sizes, self.grad_acts_memory)
        self.has_allocated_acts = True

    def build_encoder_buckets(mut self, batch_size: Int, seq_len: Int) raises:
        self.num_wte_buckets = build_wte_buckets(
            as_immut_kernel_from_mut[DType.int32](self.inputs),
            as_mut_kernel[DType.int32](self.bucket_info),
            as_mut_kernel[DType.int32](self.workload_indices),
            batch_size,
            seq_len,
            self.config.vocab_size,
            self.config.channels,
            self.wte_bucket_capacity,
        )

    def forward(
        mut self,
        inputs: MutMemPtr[DType.int32],
        targets: MutMemPtr[DType.int32],
        batch_size: Int,
        seq_len: Int,
    ) raises:
        var zero = 0
        var NULL_DTYPE_PTR = MutMemPtr[GPT2_DTYPE](unsafe_from_address=zero)
        var NULL_INT32_PTR = MutMemPtr[DType.int32](unsafe_from_address=zero)

        if (
            not self.has_allocated_params
            or self.params_memory == NULL_DTYPE_PTR
        ):
            raise Error("GPT2 error: Parameters not allocated")
        if not self.has_allocated_grads or self.grads_memory == NULL_DTYPE_PTR:
            raise Error("GPT2 error: Gradients not allocated")
        if (
            not self.has_allocated_optimizer_moments
            or self.m_memory == NULL_DTYPE_PTR
            or self.v_memory == NULL_DTYPE_PTR
        ):
            raise Error("GPT2 error: Optimizer moments not allocated")

        # Convienence variables.
        var vocab_size = self.config.vocab_size
        var vocab_size_padded = self.config.padded_vocab_size
        var num_layers = self.config.num_layer
        var num_heads = self.config.num_heads
        var channels = self.config.channels

        # Lazily allocate activations if needed.
        if (
            not self.has_allocated_acts
            or self.acts_memory == NULL_DTYPE_PTR
            or self.grad_acts_memory == NULL_DTYPE_PTR
        ):
            self.allocate_activations(batch_size, seq_len)
        else:
            # Validate activations and gradients are not larger then the prvious allocations.
            # In the future we could resize and reallocate for now we will just raise an error.
            if seq_len > self.seq_len or batch_size > self.batch_size:
                raise Error(
                    "GPT2 error: Sequence length or batch size is larger than"
                    " the previous allocations"
                )

        # Cache the inputs and targets
        memcpy(dest=self.inputs, src=inputs, count=batch_size * seq_len)

        if targets != NULL_INT32_PTR:
            memcpy(dest=self.targets, src=targets, count=batch_size * seq_len)

        self.build_encoder_buckets(batch_size, seq_len)

        comptime if Self.WORLD_SIZE > 1:
            # ZeRO-3: Gather all parameter shards into params_buf before running forward.
            # Each rank holds its persistent shard at params_memory + rank * optimizer_num_parameters.
            # This is a coarse-grained gather: one allgather per forward pass rather than per layer.
            # It is memory-correct ZeRO-3 sharding of persistent state, but does NOT yet achieve
            # the peak-memory savings of true per-layer streaming (gather/free per tensor).
            if self.zero_ctx.zero_stage >= 3:
                self.zero_ctx.allgather(
                    rebind[UnsafePointer[Scalar[GPT2_DTYPE], MutAnyOrigin]](
                        self.params_memory.as_unsafe_any_origin()
                    ),
                    self.optimizer_num_parameters,
                )
                self.ctx.synchronize()

        #########################################################
        # Forward Pass
        #########################################################

        var residual: MutMemPtr[GPT2_DTYPE]
        # Forward the encoder.
        encoder_fwd[GPT2_DTYPE, Self.target](
            as_mut_kernel[GPT2_DTYPE](self.acts.encoded),
            as_immut_kernel_from_mut[DType.int32](self.inputs),
            as_immut_kernel_from_mut[GPT2_DTYPE](self.params.wte),
            as_immut_kernel_from_mut[GPT2_DTYPE](self.params.wpe),
            batch_size,
            seq_len,
            channels,
            self.ctx,
        )

        for layer in range(num_layers):
            if (
                layer == 0
            ):  # First layer, use the encoded activations as the pointer
                residual = self.acts.encoded
            else:
                residual = (
                    self.acts.residual_3
                    + (layer - 1) * batch_size * seq_len * channels
                )

            # Get the weight (gamma) and bias (beta) pointers for this layer
            var l_ln_1_gamma = self.params.ln_1_gamma + layer * channels
            var l_ln_1_beta = self.params.ln_1_beta + layer * channels
            var l_qkv_weight = (
                self.params.qkv_weight + layer * (3 * channels) * channels
            )
            var l_qkv_bias = self.params.qkv_bias + layer * (3 * channels)
            var l_attn_proj_weight = (
                self.params.attn_proj_weight + layer * channels * channels
            )
            var l_attn_proj_bias = self.params.attn_proj_bias + layer * channels
            var l_ln_2_gamma = self.params.ln_2_gamma + layer * channels
            var l_ln_2_beta = self.params.ln_2_beta + layer * channels
            var l_fc_weight = (
                self.params.fc_weight + layer * (4 * channels) * channels
            )
            var l_fc_bias = self.params.fc_bias + layer * (4 * channels)
            var l_proj_weight = self.params.proj_weight + layer * channels * (
                4 * channels
            )
            var l_proj_bias = self.params.proj_bias + layer * channels

            # Get the pointers for the activations for this layer
            var l_ln_1 = (
                self.acts.ln_1 + layer * batch_size * seq_len * channels
            )
            var l_ln_1_mean = self.acts.ln_1_mean + layer * batch_size * seq_len
            var l_ln_1_rstd = self.acts.ln_1_rstd + layer * batch_size * seq_len
            var l_qkv = self.acts.qkv + layer * batch_size * seq_len * (
                3 * channels
            )
            var l_q = self.acts.q + layer * batch_size * seq_len * channels
            var l_k = self.acts.k + layer * batch_size * seq_len * channels
            var l_v = self.acts.v + layer * batch_size * seq_len * channels
            var l_lse = self.acts.lse + layer * batch_size * num_heads * seq_len
            var l_attn = (
                self.acts.attn + layer * batch_size * seq_len * channels
            )
            var l_attn_merged = (
                self.acts.attn_merged + layer * batch_size * seq_len * channels
            )
            var l_attn_proj = (
                self.acts.attn_proj + layer * batch_size * seq_len * channels
            )
            var l_residual_2 = (
                self.acts.residual_2 + layer * batch_size * seq_len * channels
            )
            var l_ln_2 = (
                self.acts.ln_2 + layer * batch_size * seq_len * channels
            )
            var l_ln_2_mean = self.acts.ln_2_mean + layer * batch_size * seq_len
            var l_ln_2_rstd = self.acts.ln_2_rstd + layer * batch_size * seq_len
            # fch / fch_gelu live in a single-layer scratch slot when recompute
            # is on, so their per-layer offset collapses to 0.
            var fch_layer = layer
            comptime if Self.recompute:
                fch_layer = 0
            var l_fch = self.acts.fch + fch_layer * batch_size * seq_len * (
                4 * channels
            )
            var l_fch_gelu = (
                self.acts.fch_gelu
                + fch_layer * batch_size * seq_len * (4 * channels)
            )
            var l_fc_proj = (
                self.acts.fc_proj + layer * batch_size * seq_len * channels
            )
            var l_residual_3 = (
                self.acts.residual_3 + layer * batch_size * seq_len * channels
            )

            # 1. LayerNorm 1 / Residual Fused
            if layer == 0:
                layernorm_fwd[GPT2_DTYPE, Self.target](
                    as_mut_kernel[GPT2_DTYPE](l_ln_1),
                    as_mut_kernel[GPT2_DTYPE](residual),
                    as_immut_kernel_from_mut[GPT2_DTYPE](l_ln_1_gamma),
                    as_immut_kernel_from_mut[GPT2_DTYPE](l_ln_1_beta),
                    EPSILON,
                    as_mut_kernel[GPT2_DTYPE](l_ln_1_mean),
                    as_mut_kernel[GPT2_DTYPE](l_ln_1_rstd),
                    Int64(batch_size),
                    Int64(seq_len),
                    Int64(channels),
                    self.ctx,
                )
            else:
                var prev_residual_2 = (
                    self.acts.residual_2
                    + (layer - 1) * batch_size * seq_len * channels
                )
                var prev_fc_proj = (
                    self.acts.fc_proj
                    + (layer - 1) * batch_size * seq_len * channels
                )
                layernorm_fused_residual_fwd[GPT2_DTYPE, Self.target](
                    as_mut_kernel[GPT2_DTYPE](residual),
                    as_mut_kernel[GPT2_DTYPE](l_ln_1),
                    as_mut_kernel[GPT2_DTYPE](prev_residual_2),
                    as_mut_kernel[GPT2_DTYPE](prev_fc_proj),
                    as_immut_kernel_from_mut[GPT2_DTYPE](l_ln_1_gamma),
                    as_immut_kernel_from_mut[GPT2_DTYPE](l_ln_1_beta),
                    EPSILON,
                    as_mut_kernel[GPT2_DTYPE](l_ln_1_mean),
                    as_mut_kernel[GPT2_DTYPE](l_ln_1_rstd),
                    Int64(batch_size),
                    Int64(seq_len),
                    Int64(channels),
                    self.ctx,
                )

            # Matmul QKV.
            matmul_fwd[GPT2_DTYPE, Self.target, use_gelu=False](
                as_mut_kernel[GPT2_DTYPE](l_qkv),
                as_mut_kernel[GPT2_DTYPE](NULL_DTYPE_PTR),
                as_mut_kernel[GPT2_DTYPE](l_ln_1),
                as_immut_kernel_from_mut[GPT2_DTYPE](l_qkv_weight),
                as_immut_kernel_from_mut[GPT2_DTYPE](l_qkv_bias),
                Int64(batch_size),
                Int64(seq_len),
                Int64(channels),
                Int64(3 * channels),
                self.ctx,
            )

            # Attention Forward (handles QKV Split/Transpose and Merge Heads internally).
            var head_dim = channels // num_heads
            attention_fwd[GPT2_DTYPE, Self.target, use_soft_exp=True](
                as_mut_kernel[GPT2_DTYPE](l_qkv),
                as_mut_kernel[GPT2_DTYPE](l_q),
                as_mut_kernel[GPT2_DTYPE](l_k),
                as_mut_kernel[GPT2_DTYPE](l_v),
                as_mut_kernel[GPT2_DTYPE](l_attn),
                as_mut_kernel[GPT2_DTYPE](l_attn_merged),
                as_mut_kernel[GPT2_DTYPE](l_lse),
                Int64(batch_size),
                Int64(num_heads),
                Int64(seq_len),
                Int64(head_dim),
                self.ctx,
                cache=rebind[KVCachePtr](UnsafePointer(to=self.kv_cache)),
            )

            # Matmul Attn Proj
            matmul_fwd[GPT2_DTYPE, Self.target, use_gelu=False](
                as_mut_kernel[GPT2_DTYPE](l_attn_proj),
                as_mut_kernel[GPT2_DTYPE](NULL_DTYPE_PTR),
                as_mut_kernel[GPT2_DTYPE](l_attn_merged),
                as_immut_kernel_from_mut[GPT2_DTYPE](l_attn_proj_weight),
                as_immut_kernel_from_mut[GPT2_DTYPE](l_attn_proj_bias),
                Int64(batch_size),
                Int64(seq_len),
                Int64(channels),
                Int64(channels),
                self.ctx,
            )
            # LayerNorm 2 & Residual.
            layernorm_fused_residual_fwd[GPT2_DTYPE, Self.target](
                as_mut_kernel[GPT2_DTYPE](l_residual_2),
                as_mut_kernel[GPT2_DTYPE](l_ln_2),
                as_mut_kernel[GPT2_DTYPE](residual),
                as_mut_kernel[GPT2_DTYPE](l_attn_proj),
                as_immut_kernel_from_mut[GPT2_DTYPE](l_ln_2_gamma),
                as_immut_kernel_from_mut[GPT2_DTYPE](l_ln_2_beta),
                EPSILON,
                as_mut_kernel[GPT2_DTYPE](l_ln_2_mean),
                as_mut_kernel[GPT2_DTYPE](l_ln_2_rstd),
                Int64(batch_size),
                Int64(seq_len),
                Int64(channels),
                self.ctx,
            )

            # Matmul FC (fused GELU).
            matmul_fwd[GPT2_DTYPE, Self.target, use_gelu=True](
                as_mut_kernel[GPT2_DTYPE](l_fch_gelu),
                as_mut_kernel[GPT2_DTYPE](l_fch),
                as_mut_kernel[GPT2_DTYPE](l_ln_2),
                as_immut_kernel_from_mut[GPT2_DTYPE](l_fc_weight),
                as_immut_kernel_from_mut[GPT2_DTYPE](l_fc_bias),
                Int64(batch_size),
                Int64(seq_len),
                Int64(channels),
                Int64(4 * channels),
                self.ctx,
            )

            # Matmul Proj.
            matmul_fwd[GPT2_DTYPE, Self.target, use_gelu=False](
                as_mut_kernel[GPT2_DTYPE](l_fc_proj),
                as_mut_kernel[GPT2_DTYPE](NULL_DTYPE_PTR),
                as_mut_kernel[GPT2_DTYPE](l_fch_gelu),
                as_immut_kernel_from_mut[GPT2_DTYPE](l_proj_weight),
                as_immut_kernel_from_mut[GPT2_DTYPE](l_proj_bias),
                Int64(batch_size),
                Int64(seq_len),
                Int64(4 * channels),
                Int64(channels),
                self.ctx,
            )

        # Final LayerNorm
        var last_residual_3 = (
            self.acts.residual_3
            + (num_layers - 1) * batch_size * seq_len * channels
        )
        var last_residual_2 = (
            self.acts.residual_2
            + (num_layers - 1) * batch_size * seq_len * channels
        )
        var last_fc_proj = (
            self.acts.fc_proj
            + (num_layers - 1) * batch_size * seq_len * channels
        )
        layernorm_fused_residual_fwd[GPT2_DTYPE, Self.target](
            as_mut_kernel[GPT2_DTYPE](last_residual_3),
            as_mut_kernel[GPT2_DTYPE](self.acts.ln_f),
            as_mut_kernel[GPT2_DTYPE](last_residual_2),
            as_mut_kernel[GPT2_DTYPE](last_fc_proj),
            as_immut_kernel_from_mut[GPT2_DTYPE](self.params.ln_f_gamma),
            as_immut_kernel_from_mut[GPT2_DTYPE](self.params.ln_f_beta),
            EPSILON,
            as_mut_kernel[GPT2_DTYPE](self.acts.ln_f_mean),
            as_mut_kernel[GPT2_DTYPE](self.acts.ln_f_rstd),
            Int64(batch_size),
            Int64(seq_len),
            Int64(channels),
            self.ctx,
        )

        # Output Logits (wte has no bias).
        matmul_fwd[GPT2_DTYPE, Self.target, use_gelu=False, has_bias=False](
            as_mut_kernel[GPT2_DTYPE](self.acts.logits),
            as_mut_kernel[GPT2_DTYPE](NULL_DTYPE_PTR),
            as_mut_kernel[GPT2_DTYPE](self.acts.ln_f),
            as_immut_kernel_from_mut[GPT2_DTYPE](self.params.wte),
            as_immut_kernel_from_mut[GPT2_DTYPE](NULL_DTYPE_PTR),
            Int64(batch_size),
            Int64(seq_len),
            Int64(channels),
            Int64(vocab_size_padded),
            self.ctx,
        )
        self.ctx.synchronize()

        # Fused classifier (cross entropy loss).
        if targets != NULL_INT32_PTR:
            fused_classifier[GPT2_DTYPE, Self.target, write_d_logits=False](
                as_mut_kernel[GPT2_DTYPE](self.acts.logits),
                as_mut_kernel[GPT2_DTYPE](self.acts.losses),
                as_immut_kernel[DType.float32](
                    ImmutMemPtr[DType.float32](unsafe_from_address=zero)
                ),
                as_immut_kernel_from_mut[DType.int32](self.targets),
                Int64(batch_size),
                Int64(seq_len),
                Int64(vocab_size),
                Int64(vocab_size_padded),
                self.ctx,
            )

            # Sync context before reading losses on CPU.
            self.ctx.synchronize()

            # Copy losses back to host.
            var count = batch_size * seq_len
            self.ctx.enqueue_copy(
                dst_ptr=rebind[UnsafePointer[Scalar[GPT2_DTYPE], MutAnyOrigin]](
                    self.losses_host_buf.unsafe_ptr().as_unsafe_any_origin()
                ),
                src_ptr=rebind[
                    UnsafePointer[Scalar[GPT2_DTYPE], ImmutAnyOrigin]
                ](self.acts.losses.as_unsafe_any_origin()),
                size=count,
            )
            self.ctx.synchronize()

            # Average loss into self.mean_loss.
            var total_loss: Float32 = 0.0
            for i in range(count):
                total_loss += self.losses_host_buf[i].cast[DType.float32]()
            self.mean_loss = total_loss / Float32(count)

    def zero_gradients(mut self) raises:
        if self.has_allocated_grads:
            self.grads_buf.enqueue_fill(Scalar[GPT2_DTYPE](0.0))
            comptime if Self.WORLD_SIZE > 1:
                if self.zero_ctx.zero_stage >= 2:
                    self.sharded_grads_buf.enqueue_fill(Scalar[GPT2_DTYPE](0.0))
        if self.has_allocated_acts:
            self.grad_acts_buf.enqueue_fill(Scalar[GPT2_DTYPE](0.0))
        self.ctx.synchronize()

    def backward(mut self) raises:
        if self.mean_loss == -1.0:
            raise Error("GPT2 error: must call forward pass first")
        if not self.has_allocated_acts:
            raise Error("GPT2 error: activations not allocated")

        self.zero_gradients()

        var zero = 0
        var NULL_DTYPE_PTR = MutMemPtr[GPT2_DTYPE](unsafe_from_address=zero)

        var batch_size = self.batch_size
        var seq_len = self.seq_len
        var channels = self.config.channels
        var num_layers = self.config.num_layer
        var num_heads = self.config.num_heads
        var vocab_size = self.config.vocab_size
        var vocab_size_padded = self.config.padded_vocab_size
        var head_dim = channels // num_heads
        var layer_stride = batch_size * seq_len * channels
        var qkv_layer_stride = layer_stride * 3
        var fch_layer_stride = layer_stride * 4

        ##############################################################
        # Backward Pass
        ##############################################################

        var dloss_mean = Scalar[DType.float32](1.0) / Scalar[DType.float32](
            batch_size * seq_len
        )
        var dloss_val = dloss_mean.cast[GPT2_DTYPE]()
        for i in range(batch_size * seq_len):
            self.losses_host_buf[i] = dloss_val

        self.ctx.enqueue_copy(
            dst_ptr=rebind[UnsafePointer[Scalar[GPT2_DTYPE], MutAnyOrigin]](
                self.grad_acts.losses.as_unsafe_any_origin()
            ),
            src_ptr=rebind[UnsafePointer[Scalar[GPT2_DTYPE], ImmutAnyOrigin]](
                self.losses_host_buf.unsafe_ptr().as_unsafe_any_origin()
            ),
            size=batch_size * seq_len,
        )

        # Fused classifier backward writes d_logits in-place into acts.logits.
        fused_classifier[GPT2_DTYPE, Self.target, write_d_logits=True](
            as_mut_kernel[GPT2_DTYPE](self.acts.logits),
            as_mut_kernel[GPT2_DTYPE](self.acts.losses),
            as_immut_kernel_from_mut[DType.float32](self.grad_acts.losses),
            as_immut_kernel_from_mut[DType.int32](self.targets),
            Int64(batch_size),
            Int64(seq_len),
            Int64(vocab_size),
            Int64(vocab_size_padded),
            self.ctx,
        )

        # LM head matmul backward (wte has no bias).
        matmul_bwd[GPT2_DTYPE, Self.target, use_gelu=False, has_bias=False](
            as_mut_kernel[GPT2_DTYPE](self.grad_acts.ln_f),
            as_mut_kernel[GPT2_DTYPE](self.grads.wte),
            as_mut_kernel[GPT2_DTYPE](NULL_DTYPE_PTR),
            as_mut_kernel[GPT2_DTYPE](self.acts.logits),
            as_mut_kernel[GPT2_DTYPE](self.acts.ln_f),
            as_immut_kernel_from_mut[GPT2_DTYPE](self.params.wte),
            as_mut_kernel[GPT2_DTYPE](NULL_DTYPE_PTR),
            as_mut_kernel[GPT2_DTYPE](self.grad_acts.logits),
            Int64(batch_size),
            Int64(seq_len),
            Int64(channels),
            Int64(vocab_size_padded),
            self.ctx,
        )

        # Final fused LayerNorm backward.
        var last_layer = num_layers - 1
        var last_residual_3 = self.acts.residual_3 + last_layer * layer_stride
        var last_residual_2 = self.acts.residual_2 + last_layer * layer_stride
        var last_fc_proj = self.acts.fc_proj + last_layer * layer_stride
        var d_last_residual_2 = (
            self.grad_acts.residual_2 + last_layer * layer_stride
        )
        var d_last_fc_proj = self.grad_acts.fc_proj + last_layer * layer_stride

        layernorm_fused_residual_bwd[GPT2_DTYPE, Self.target](
            as_mut_kernel[GPT2_DTYPE](d_last_residual_2),
            as_mut_kernel[GPT2_DTYPE](d_last_fc_proj),
            as_mut_kernel[GPT2_DTYPE](self.grad_acts.ln_f),
            as_mut_kernel[GPT2_DTYPE](last_residual_3),
            as_immut_kernel_from_mut[GPT2_DTYPE](self.params.ln_f_gamma),
            as_mut_kernel[GPT2_DTYPE](self.acts.ln_f_mean),
            as_mut_kernel[GPT2_DTYPE](self.acts.ln_f_rstd),
            as_mut_kernel[GPT2_DTYPE](self.grads.ln_f_gamma),
            as_mut_kernel[GPT2_DTYPE](self.grads.ln_f_beta),
            as_mut_kernel[GPT2_DTYPE](
                self.grad_acts.residual_3 + last_layer * layer_stride
            ),
            Int64(batch_size),
            Int64(seq_len),
            Int64(channels),
            self.ctx,
        )

        # Backward through the transformer blocks.
        for layer in range(num_layers - 1, -1, -1):
            var layer_offset = layer * layer_stride

            # Layer weights.
            var l_ln_1_gamma = self.params.ln_1_gamma + layer * channels
            var l_ln_1_beta = self.params.ln_1_beta + layer * channels
            var l_qkv_weight = (
                self.params.qkv_weight + layer * (3 * channels) * channels
            )
            var l_qkv_bias = self.params.qkv_bias + layer * (3 * channels)
            var l_attn_proj_weight = (
                self.params.attn_proj_weight + layer * channels * channels
            )
            var l_attn_proj_bias = self.params.attn_proj_bias + layer * channels
            var l_ln_2_gamma = self.params.ln_2_gamma + layer * channels
            var l_ln_2_beta = self.params.ln_2_beta + layer * channels
            var l_fc_weight = (
                self.params.fc_weight + layer * (4 * channels) * channels
            )
            var l_fc_bias = self.params.fc_bias + layer * (4 * channels)
            var l_proj_weight = self.params.proj_weight + layer * channels * (
                4 * channels
            )
            var l_proj_bias = self.params.proj_bias + layer * channels

            # Layer weight gradients.
            var d_l_ln_1_gamma = self.grads.ln_1_gamma + layer * channels
            var d_l_ln_1_beta = self.grads.ln_1_beta + layer * channels
            var d_l_qkv_weight = (
                self.grads.qkv_weight + layer * (3 * channels) * channels
            )
            var d_l_qkv_bias = self.grads.qkv_bias + layer * (3 * channels)
            var d_l_attn_proj_weight = (
                self.grads.attn_proj_weight + layer * channels * channels
            )
            var d_l_attn_proj_bias = (
                self.grads.attn_proj_bias + layer * channels
            )
            var d_l_ln_2_gamma = self.grads.ln_2_gamma + layer * channels
            var d_l_ln_2_beta = self.grads.ln_2_beta + layer * channels
            var d_l_fc_weight = (
                self.grads.fc_weight + layer * (4 * channels) * channels
            )
            var d_l_fc_bias = self.grads.fc_bias + layer * (4 * channels)
            var d_l_proj_weight = self.grads.proj_weight + layer * channels * (
                4 * channels
            )
            var d_l_proj_bias = self.grads.proj_bias + layer * channels

            # Saved activations for this layer.
            var l_ln_1 = self.acts.ln_1 + layer_offset
            var l_ln_1_mean = self.acts.ln_1_mean + layer * batch_size * seq_len
            var l_ln_1_rstd = self.acts.ln_1_rstd + layer * batch_size * seq_len
            var l_qkv = self.acts.qkv + layer * qkv_layer_stride
            var l_q = self.acts.q + layer_offset
            var l_k = self.acts.k + layer_offset
            var l_v = self.acts.v + layer_offset
            var l_lse = self.acts.lse + layer * batch_size * num_heads * seq_len
            var l_attn = self.acts.attn + layer_offset
            var l_attn_merged = self.acts.attn_merged + layer_offset
            var l_attn_proj = self.acts.attn_proj + layer_offset
            var l_residual_2 = self.acts.residual_2 + layer_offset
            var l_ln_2 = self.acts.ln_2 + layer_offset
            var l_ln_2_mean = self.acts.ln_2_mean + layer * batch_size * seq_len
            var l_ln_2_rstd = self.acts.ln_2_rstd + layer * batch_size * seq_len
            # With recompute on, fch / fch_gelu (and their grads) share a single
            # scratch slot, so the per-layer offset collapses to 0; backward
            # rematerializes them below before the MLP backward kernels run.
            var fch_layer = layer
            comptime if Self.recompute:
                fch_layer = 0
            var l_fch = self.acts.fch + fch_layer * fch_layer_stride
            var l_fch_gelu = self.acts.fch_gelu + fch_layer * fch_layer_stride
            var l_fc_proj = self.acts.fc_proj + layer_offset

            # Activation gradients for this layer.
            var d_l_ln_1 = self.grad_acts.ln_1 + layer_offset
            var d_l_qkv = self.grad_acts.qkv + layer * qkv_layer_stride
            var d_l_q = self.grad_acts.q + layer_offset
            var d_l_k = self.grad_acts.k + layer_offset
            var d_l_v = self.grad_acts.v + layer_offset
            var d_l_attn = self.grad_acts.attn + layer_offset
            var d_l_attn_merged = self.grad_acts.attn_merged + layer_offset
            var d_l_attn_proj = self.grad_acts.attn_proj + layer_offset
            var d_l_ln_2 = self.grad_acts.ln_2 + layer_offset
            var d_l_fch = self.grad_acts.fch + fch_layer * fch_layer_stride
            var d_l_fch_gelu = (
                self.grad_acts.fch_gelu + fch_layer * fch_layer_stride
            )
            var d_l_fc_proj = self.grad_acts.fc_proj + layer_offset

            var d_block_input: MutMemPtr[GPT2_DTYPE]
            if layer == 0:
                d_block_input = self.grad_acts.encoded
            else:
                d_block_input = (
                    self.grad_acts.residual_3 + (layer - 1) * layer_stride
                )

            # Recompute checkpointing: rematerialize fch (pre-GELU) and fch_gelu
            # (post-GELU) into the scratch slot by re-running the same fused FC
            # matmul the forward used, reading the still-resident ln_2. Both MLP
            # backward kernels below then read them exactly as in the no-recompute
            # path. Bit-identical to having stored them, since the GEMM + GELU are
            # deterministic in the same inputs.
            comptime if Self.recompute:
                matmul_fwd[GPT2_DTYPE, Self.target, use_gelu=True](
                    as_mut_kernel[GPT2_DTYPE](l_fch_gelu),
                    as_mut_kernel[GPT2_DTYPE](l_fch),
                    as_mut_kernel[GPT2_DTYPE](l_ln_2),
                    as_immut_kernel_from_mut[GPT2_DTYPE](l_fc_weight),
                    as_immut_kernel_from_mut[GPT2_DTYPE](l_fc_bias),
                    Int64(batch_size),
                    Int64(seq_len),
                    Int64(channels),
                    Int64(4 * channels),
                    self.ctx,
                )

            # MLP projection backward.
            var loop_t0 = global_perf_counter_ns()
            matmul_bwd[GPT2_DTYPE, Self.target, use_gelu=False](
                as_mut_kernel[GPT2_DTYPE](d_l_fch_gelu),
                as_mut_kernel[GPT2_DTYPE](d_l_proj_weight),
                as_mut_kernel[GPT2_DTYPE](d_l_proj_bias),
                as_mut_kernel[GPT2_DTYPE](d_l_fc_proj),
                as_mut_kernel[GPT2_DTYPE](l_fch_gelu),
                as_immut_kernel_from_mut[GPT2_DTYPE](l_proj_weight),
                as_mut_kernel[GPT2_DTYPE](NULL_DTYPE_PTR),
                as_mut_kernel[GPT2_DTYPE](d_l_attn),
                Int64(batch_size),
                Int64(seq_len),
                Int64(4 * channels),
                Int64(channels),
                self.ctx,
            )

            # MLP FC backward (fused GELU).
            matmul_bwd[GPT2_DTYPE, Self.target, use_gelu=True](
                as_mut_kernel[GPT2_DTYPE](d_l_ln_2),
                as_mut_kernel[GPT2_DTYPE](d_l_fc_weight),
                as_mut_kernel[GPT2_DTYPE](d_l_fc_bias),
                as_mut_kernel[GPT2_DTYPE](d_l_fch_gelu),
                as_mut_kernel[GPT2_DTYPE](l_ln_2),
                as_immut_kernel_from_mut[GPT2_DTYPE](l_fc_weight),
                as_mut_kernel[GPT2_DTYPE](l_fch),
                as_mut_kernel[GPT2_DTYPE](d_l_fch),
                Int64(batch_size),
                Int64(seq_len),
                Int64(channels),
                Int64(4 * channels),
                self.ctx,
            )

            # LayerNorm 2 fused residual backward.
            layernorm_fused_residual_bwd[GPT2_DTYPE, Self.target](
                as_mut_kernel[GPT2_DTYPE](d_block_input),
                as_mut_kernel[GPT2_DTYPE](d_l_attn_proj),
                as_mut_kernel[GPT2_DTYPE](d_l_ln_2),
                as_mut_kernel[GPT2_DTYPE](l_residual_2),
                as_immut_kernel_from_mut[GPT2_DTYPE](l_ln_2_gamma),
                as_mut_kernel[GPT2_DTYPE](l_ln_2_mean),
                as_mut_kernel[GPT2_DTYPE](l_ln_2_rstd),
                as_mut_kernel[GPT2_DTYPE](d_l_ln_2_gamma),
                as_mut_kernel[GPT2_DTYPE](d_l_ln_2_beta),
                as_mut_kernel[GPT2_DTYPE](
                    self.grad_acts.residual_2 + layer_offset
                ),
                Int64(batch_size),
                Int64(seq_len),
                Int64(channels),
                self.ctx,
            )

            # Attention projection backward.
            matmul_bwd[GPT2_DTYPE, Self.target, use_gelu=False](
                as_mut_kernel[GPT2_DTYPE](d_l_attn_merged),
                as_mut_kernel[GPT2_DTYPE](d_l_attn_proj_weight),
                as_mut_kernel[GPT2_DTYPE](d_l_attn_proj_bias),
                as_mut_kernel[GPT2_DTYPE](d_l_attn_proj),
                as_mut_kernel[GPT2_DTYPE](l_attn_merged),
                as_immut_kernel_from_mut[GPT2_DTYPE](l_attn_proj_weight),
                as_mut_kernel[GPT2_DTYPE](NULL_DTYPE_PTR),
                as_mut_kernel[GPT2_DTYPE](d_l_qkv),
                Int64(batch_size),
                Int64(seq_len),
                Int64(channels),
                Int64(channels),
                self.ctx,
            )

            # Attention backward.
            attention_bwd[GPT2_DTYPE, Self.target, use_soft_exp=True](
                as_mut_kernel[GPT2_DTYPE](d_l_qkv),
                as_mut_kernel[GPT2_DTYPE](d_l_q),
                as_mut_kernel[GPT2_DTYPE](d_l_k),
                as_mut_kernel[GPT2_DTYPE](d_l_v),
                as_mut_kernel[GPT2_DTYPE](d_l_attn),
                as_mut_kernel[GPT2_DTYPE](d_l_attn_merged),
                as_mut_kernel[GPT2_DTYPE](l_q),
                as_mut_kernel[GPT2_DTYPE](l_k),
                as_mut_kernel[GPT2_DTYPE](l_v),
                as_mut_kernel[GPT2_DTYPE](l_attn),
                as_mut_kernel[GPT2_DTYPE](l_lse),
                Int64(batch_size),
                Int64(num_heads),
                Int64(seq_len),
                Int64(head_dim),
                self.ctx,
                cache=rebind[KVCachePtr](UnsafePointer(to=self.kv_cache)),
            )

            # QKV matmul backward.
            matmul_bwd[GPT2_DTYPE, Self.target, use_gelu=False](
                as_mut_kernel[GPT2_DTYPE](d_l_ln_1),
                as_mut_kernel[GPT2_DTYPE](d_l_qkv_weight),
                as_mut_kernel[GPT2_DTYPE](d_l_qkv_bias),
                as_mut_kernel[GPT2_DTYPE](d_l_qkv),
                as_mut_kernel[GPT2_DTYPE](l_ln_1),
                as_immut_kernel_from_mut[GPT2_DTYPE](l_qkv_weight),
                as_mut_kernel[GPT2_DTYPE](NULL_DTYPE_PTR),
                as_mut_kernel[GPT2_DTYPE](d_l_fch_gelu),
                Int64(batch_size),
                Int64(seq_len),
                Int64(channels),
                Int64(3 * channels),
                self.ctx,
            )

            # LayerNorm 1 backward.
            if layer == 0:
                layernorm_bwd[GPT2_DTYPE, Self.target](
                    as_mut_kernel[GPT2_DTYPE](d_l_ln_1),
                    as_mut_kernel[GPT2_DTYPE](l_ln_1),
                    as_immut_kernel_from_mut[GPT2_DTYPE](l_ln_1_gamma),
                    as_mut_kernel[GPT2_DTYPE](l_ln_1_mean),
                    as_mut_kernel[GPT2_DTYPE](l_ln_1_rstd),
                    as_mut_kernel[GPT2_DTYPE](d_block_input),
                    as_mut_kernel[GPT2_DTYPE](d_l_ln_1_gamma),
                    as_mut_kernel[GPT2_DTYPE](d_l_ln_1_beta),
                    Int64(batch_size),
                    Int64(seq_len),
                    Int64(channels),
                    self.ctx,
                )
            else:
                var prev_layer_offset = (layer - 1) * layer_stride
                layernorm_fused_residual_bwd[GPT2_DTYPE, Self.target](
                    as_mut_kernel[GPT2_DTYPE](
                        self.grad_acts.residual_2 + prev_layer_offset
                    ),
                    as_mut_kernel[GPT2_DTYPE](
                        self.grad_acts.fc_proj + prev_layer_offset
                    ),
                    as_mut_kernel[GPT2_DTYPE](d_l_ln_1),
                    as_mut_kernel[GPT2_DTYPE](
                        self.acts.residual_3 + prev_layer_offset
                    ),
                    as_immut_kernel_from_mut[GPT2_DTYPE](l_ln_1_gamma),
                    as_mut_kernel[GPT2_DTYPE](l_ln_1_mean),
                    as_mut_kernel[GPT2_DTYPE](l_ln_1_rstd),
                    as_mut_kernel[GPT2_DTYPE](d_l_ln_1_gamma),
                    as_mut_kernel[GPT2_DTYPE](d_l_ln_1_beta),
                    as_mut_kernel[GPT2_DTYPE](
                        self.grad_acts.residual_3 + prev_layer_offset
                    ),
                    Int64(batch_size),
                    Int64(seq_len),
                    Int64(channels),
                    self.ctx,
                )

        # Encoder backward: scatter token grads into wte, sum position grads into wpe.
        encoder_bwd[GPT2_DTYPE, Self.target](
            as_mut_kernel[GPT2_DTYPE](self.grads.wte),
            as_mut_kernel[GPT2_DTYPE](self.grads.wpe),
            as_immut_kernel_from_mut[DType.int32](self.bucket_info),
            as_immut_kernel_from_mut[DType.int32](self.workload_indices),
            as_immut_kernel_from_mut[GPT2_DTYPE](self.grad_acts.encoded),
            self.num_wte_buckets,
            batch_size,
            seq_len,
            channels,
            self.ctx,
        )

        comptime if Self.WORLD_SIZE > 1:
            # ZeRO-0 (DDP) and ZeRO-1 use allreduce: full gradient sum is replicated
            # to all ranks (2N communication). ZeRO-1 then reads grads_memory + rank*opt
            # in the optimizer step.
            # ZeRO-2/3 use reduce-scatter: each rank receives only its gradient shard
            # (N communication), which is stored in sharded_grads_memory.
            if self.zero_ctx.zero_stage <= 1:
                self.zero_ctx.allreduce(
                    rebind[UnsafePointer[Scalar[GPT2_DTYPE], MutAnyOrigin]](
                        self.grads_memory.as_unsafe_any_origin()
                    ),
                    self.num_parameters,
                )
            else:
                self.zero_ctx.reducescatter(
                    rebind[UnsafePointer[Scalar[GPT2_DTYPE], MutAnyOrigin]](
                        self.grads_memory.as_unsafe_any_origin()
                    ),
                    rebind[UnsafePointer[Scalar[GPT2_DTYPE], MutAnyOrigin]](
                        self.sharded_grads_memory.as_unsafe_any_origin()
                    ),
                    self.optimizer_num_parameters,
                )
            self.ctx.synchronize()

    def update(
        mut self,
        t: UInt32,
        learning_rate: Scalar[DType.float32],
        beta1: Scalar[DType.float32] = Scalar[DType.float32](0.9),
        beta2: Scalar[DType.float32] = Scalar[DType.float32](0.999),
        eps: Scalar[DType.float32] = Scalar[DType.float32](1e-8),
        weight_decay: Scalar[DType.float32] = Scalar[DType.float32](0.0),
        grad_scale: Scalar[DType.float32] = Scalar[DType.float32](1.0),
    ) raises:
        var scaled_grad_scale = grad_scale
        comptime if Self.WORLD_SIZE > 1:
            scaled_grad_scale = grad_scale / Scalar[DType.float32](
                Self.WORLD_SIZE
            )

        var config = AdamWConfig[GPT2_DTYPE](
            learning_rate=learning_rate,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            weight_decay=weight_decay,
            grad_scale=scaled_grad_scale,
        )

        comptime if is_cpu[Self.target]():
            var p_ptr: MutMemPtr[GPT2_DTYPE] = self.params_memory
            var g_ptr: MutMemPtr[GPT2_DTYPE] = self.grads_memory
            var update_num_params = self.num_parameters

            comptime if Self.WORLD_SIZE > 1:
                if self.zero_ctx.zero_stage >= 1:
                    var offset = (
                        self.zero_ctx.rank * self.optimizer_num_parameters
                    )
                    var local_num_params = min(
                        self.num_parameters - offset,
                        self.optimizer_num_parameters,
                    )
                    p_ptr = self.params_memory + offset
                    # ZeRO-1: grads were allreduced — each rank reads its shard directly
                    # from the replicated grads_memory (no sharded_grads_buf allocated).
                    # ZeRO-2/3: grads were reduce-scattered into sharded_grads_memory.
                    if self.zero_ctx.zero_stage == 1:
                        g_ptr = self.grads_memory + offset
                    else:
                        g_ptr = self.sharded_grads_memory
                    update_num_params = local_num_params

            if update_num_params > 0:
                comptime simd_w = simd_width_of[GPT2_DTYPE]()
                adamw_update[GPT2_DTYPE, Self.target, width=simd_w](
                    update_num_params,
                    p_ptr.as_unsafe_any_origin(),
                    g_ptr.as_unsafe_any_origin(),
                    self.m_memory.as_unsafe_any_origin(),
                    self.v_memory.as_unsafe_any_origin(),
                    t,
                    config,
                    self.ctx,
                )

            self.ctx.synchronize()

            comptime if Self.WORLD_SIZE > 1:
                # ZeRO-3 does NOT allgather after the optimizer step — the next
                # forward() call gathers param shards just-in-time.
                # ZeRO-1 and ZeRO-2 allgather here so params_memory is consistent
                # before the next forward (they do not pre-gather in forward).
                if (
                    self.zero_ctx.zero_stage >= 1
                    and self.zero_ctx.zero_stage < 3
                ):
                    self.zero_ctx.allgather(
                        rebind[UnsafePointer[Scalar[GPT2_DTYPE], MutAnyOrigin]](
                            self.params_memory.as_unsafe_any_origin()
                        ),
                        self.optimizer_num_parameters,
                    )
                    self.ctx.synchronize()
        else:
            comptime if Self.WORLD_SIZE > 1:
                if self.zero_ctx.zero_stage == 0:
                    # ZeRO-0 (DDP): every rank updates all params using full grads.
                    adamw_update[GPT2_DTYPE, Self.target](
                        self.num_parameters,
                        as_mut_kernel[GPT2_DTYPE](self.params_memory),
                        as_mut_kernel[GPT2_DTYPE](self.grads_memory),
                        as_mut_kernel[GPT2_DTYPE](self.m_memory),
                        as_mut_kernel[GPT2_DTYPE](self.v_memory),
                        t,
                        config,
                        self.ctx,
                    )
                elif self.zero_ctx.zero_stage == 1:
                    # ZeRO-1: grads were allreduced — read shard from grads_memory + offset.
                    var offset = (
                        self.zero_ctx.rank * self.optimizer_num_parameters
                    )
                    var local_num_params = min(
                        self.num_parameters - offset,
                        self.optimizer_num_parameters,
                    )
                    if local_num_params > 0:
                        adamw_update[GPT2_DTYPE, Self.target](
                            local_num_params,
                            as_mut_kernel[GPT2_DTYPE](
                                self.params_memory + offset
                            ),
                            as_mut_kernel[GPT2_DTYPE](
                                self.grads_memory + offset
                            ),
                            as_mut_kernel[GPT2_DTYPE](self.m_memory),
                            as_mut_kernel[GPT2_DTYPE](self.v_memory),
                            t,
                            config,
                            self.ctx,
                        )
                    self.ctx.synchronize()
                    # Allgather so params_memory is consistent before the next forward.
                    self.zero_ctx.allgather(
                        rebind[UnsafePointer[Scalar[GPT2_DTYPE], MutAnyOrigin]](
                            self.params_memory.as_unsafe_any_origin()
                        ),
                        self.optimizer_num_parameters,
                    )
                    self.ctx.synchronize()
                else:
                    # ZeRO 2/3: grads were reduce-scattered into sharded_grads_memory.
                    var offset = (
                        self.zero_ctx.rank * self.optimizer_num_parameters
                    )
                    var local_num_params = min(
                        self.num_parameters - offset,
                        self.optimizer_num_parameters,
                    )
                    if local_num_params > 0:
                        adamw_update[GPT2_DTYPE, Self.target](
                            local_num_params,
                            as_mut_kernel[GPT2_DTYPE](
                                self.params_memory + offset
                            ),
                            as_mut_kernel[GPT2_DTYPE](
                                self.sharded_grads_memory
                            ),
                            as_mut_kernel[GPT2_DTYPE](self.m_memory),
                            as_mut_kernel[GPT2_DTYPE](self.v_memory),
                            t,
                            config,
                            self.ctx,
                        )
                    self.ctx.synchronize()
                    # ZeRO-2 allgathers here so params_memory is immediately consistent.
                    if self.zero_ctx.zero_stage == 2:
                        self.zero_ctx.allgather(
                            rebind[
                                UnsafePointer[Scalar[GPT2_DTYPE], MutAnyOrigin]
                            ](self.params_memory.as_unsafe_any_origin()),
                            self.optimizer_num_parameters,
                        )
                        self.ctx.synchronize()
            else:
                adamw_update[GPT2_DTYPE, Self.target](
                    self.num_parameters,
                    as_mut_kernel[GPT2_DTYPE](self.params_memory),
                    as_mut_kernel[GPT2_DTYPE](self.grads_memory),
                    as_mut_kernel[GPT2_DTYPE](self.m_memory),
                    as_mut_kernel[GPT2_DTYPE](self.v_memory),
                    t,
                    config,
                    self.ctx,
                )

    def _checkpoint_config(self) -> CheckpointConfig:
        return CheckpointConfig(
            max_seq_len=self.config.max_seq_len,
            vocab_size=self.config.vocab_size,
            num_layer=self.config.num_layer,
            num_heads=self.config.num_heads,
            channels=self.config.channels,
            padded_vocab_size=self.config.padded_vocab_size,
        )

    def _copy_device_to_host(
        mut self,
        host: HostBuffer[GPT2_DTYPE],
        src: MutMemPtr[GPT2_DTYPE],
        n: Int,
    ) raises:
        self.ctx.enqueue_copy(
            dst_ptr=rebind[UnsafePointer[Scalar[GPT2_DTYPE], MutAnyOrigin]](
                host.unsafe_ptr().as_unsafe_any_origin()
            ),
            src_ptr=rebind[UnsafePointer[Scalar[GPT2_DTYPE], ImmutAnyOrigin]](
                src.as_unsafe_any_origin()
            ),
            size=n,
        )

    def _copy_host_to_device(
        mut self,
        dst: MutMemPtr[GPT2_DTYPE],
        host: HostBuffer[GPT2_DTYPE],
        n: Int,
    ) raises:
        self.ctx.enqueue_copy(
            dst_ptr=rebind[UnsafePointer[Scalar[GPT2_DTYPE], MutAnyOrigin]](
                dst.as_unsafe_any_origin()
            ),
            src_ptr=rebind[UnsafePointer[Scalar[GPT2_DTYPE], ImmutAnyOrigin]](
                host.unsafe_ptr().as_unsafe_any_origin()
            ),
            size=n,
        )

    def write_checkpoint(
        mut self,
        model_path: String,
        state_path: String,
        step: Int,
        loader: DataLoader,
        sampler_rng_state: UInt64,
    ) raises:
        """Write a model + optimizer-state checkpoint to disk.

        Rank 0 writes the full model file; every rank writes its own optimizer
        shard (m, v) to its `state_path`. Mirrors llm.c's split of model_*.bin
        (shared) and state_*_rank.bin (per-process).
        """
        # ZeRO-3 keeps only this rank's parameter shard resident in
        # params_memory after update() (the next forward re-gathers it). Gather
        # the full parameter set before snapshotting the model.
        comptime if Self.WORLD_SIZE > 1:
            if self.zero_ctx.zero_stage >= 3:
                self.zero_ctx.allgather(
                    rebind[UnsafePointer[Scalar[GPT2_DTYPE], MutAnyOrigin]](
                        self.params_memory.as_unsafe_any_origin()
                    ),
                    self.optimizer_num_parameters,
                )
                self.ctx.synchronize()

        if self.zero_ctx.rank == 0:
            var host_params = self.ctx.enqueue_create_host_buffer[GPT2_DTYPE](
                self.num_parameters
            )
            self.ctx.synchronize()
            self._copy_device_to_host(
                host_params, self.params_memory, self.num_parameters
            )
            self.ctx.synchronize()
            write_model_checkpoint(
                model_path,
                self._checkpoint_config(),
                rebind_mut_mem[GPT2_DTYPE](
                    host_params.unsafe_ptr().as_unsafe_any_origin()
                ),
                self.num_parameters,
            )

        var n_opt = self.optimizer_num_parameters
        var host_m = self.ctx.enqueue_create_host_buffer[GPT2_DTYPE](n_opt)
        var host_v = self.ctx.enqueue_create_host_buffer[GPT2_DTYPE](n_opt)
        self.ctx.synchronize()
        self._copy_device_to_host(host_m, self.m_memory, n_opt)
        self._copy_device_to_host(host_v, self.v_memory, n_opt)
        self.ctx.synchronize()
        write_state_checkpoint(
            state_path,
            make_training_state(loader, step, sampler_rng_state),
            rebind_mut_mem[GPT2_DTYPE](
                host_m.unsafe_ptr().as_unsafe_any_origin()
            ),
            rebind_mut_mem[GPT2_DTYPE](
                host_v.unsafe_ptr().as_unsafe_any_origin()
            ),
            n_opt,
        )

    def load_checkpoint(
        mut self, model_path: String, state_path: String
    ) raises -> TrainingState:
        """Restore params and optimizer moments from a checkpoint into device
        memory; return the TrainingState (step, rng, dataloader position).

        The model must already be allocated with the same config that wrote the
        checkpoint (build it from the same base checkpoint first). Each rank
        reads the shared model file plus its own optimizer-state shard.
        """
        var host_params = self.ctx.enqueue_create_host_buffer[GPT2_DTYPE](
            self.num_parameters
        )
        self.ctx.synchronize()
        _ = read_model_checkpoint(
            model_path,
            rebind_mut_mem[GPT2_DTYPE](
                host_params.unsafe_ptr().as_unsafe_any_origin()
            ),
            self.num_parameters,
        )
        self._copy_host_to_device(
            self.params_memory, host_params, self.num_parameters
        )
        self.ctx.synchronize()

        var n_opt = self.optimizer_num_parameters
        var host_m = self.ctx.enqueue_create_host_buffer[GPT2_DTYPE](n_opt)
        var host_v = self.ctx.enqueue_create_host_buffer[GPT2_DTYPE](n_opt)
        self.ctx.synchronize()
        var state = read_state_checkpoint(
            state_path,
            rebind_mut_mem[GPT2_DTYPE](
                host_m.unsafe_ptr().as_unsafe_any_origin()
            ),
            rebind_mut_mem[GPT2_DTYPE](
                host_v.unsafe_ptr().as_unsafe_any_origin()
            ),
            n_opt,
        )
        self._copy_host_to_device(self.m_memory, host_m, n_opt)
        self._copy_host_to_device(self.v_memory, host_v, n_opt)
        self.ctx.synchronize()
        return state^


# ===----------------------------------------------------------------------=== #
# The Main Training Loop!
# ===----------------------------------------------------------------------=== #


def train[
    target: StaticString,
    WORLD_SIZE: Int = 1,
](
    rank: Int = 0,
    cpu_coord: Optional[
        UnsafePointer[CpuCoordinator, MutUntrackedOrigin]
    ] = Optional[UnsafePointer[CpuCoordinator, MutUntrackedOrigin]](),
    zero_stage: Int = 0,
) raises:
    var ctx: DeviceContext
    comptime if is_cpu[target]():
        ctx = DeviceContext(api="cpu")
    else:
        ctx = DeviceContext()

    # Zero sharding.
    # Build the GPT-2 model from my checkpoint.
    var model = GPT2[target, WORLD_SIZE](
        "gpt2_124M.bin",
        rank,
        zero_stage,
        ctx,
        cpu_coordinator_ptr=cpu_coord,
    )

    # Disk checkpointing config (opt-in via env, mirroring llm.c's CLI flags):
    #   LLMM_SAVE_EVERY=N      write a checkpoint every N steps (0 = off)
    #   LLMM_OUTPUT_DIR=dir    directory for model_*/state_* files
    #   LLMM_RESUME_FROM=step  resume params/optimizer/dataloader from a step
    var save_every = 0
    if getenv("LLMM_SAVE_EVERY") != "":
        save_every = atol(getenv("LLMM_SAVE_EVERY"))
    var output_dir = getenv("LLMM_OUTPUT_DIR")
    if output_dir == "":
        output_dir = String("checkpoints")
    var resume_from = -1
    if getenv("LLMM_RESUME_FROM") != "":
        resume_from = atol(getenv("LLMM_RESUME_FROM"))
    if save_every > 0 and rank == 0:
        var os = Python.import_module("os")
        _ = os.makedirs(output_dir, exist_ok=True)

    # Build the dataloaders from tokens files.
    # TODO: Add tiny stories dataset.
    var tiny_shakespeare_train = (
        "./data/.tinyshakespeare/tiny_shakespeare_train.bin"
    )
    var tiny_shakespeare_val = (
        "./data/.tinyshakespeare/tiny_shakespeare_val.bin"
    )

    var train_tokens = tiny_shakespeare_train
    var val_tokens = tiny_shakespeare_val

    try:
        var file = open(tiny_shakespeare_train, "r")
        file.close()
    except:
        raise Error("Failed to open train tokens file: " + train_tokens)

    comptime BATCH_SIZE = 4
    comptime SEQ_LEN = 1024

    var train_loader = DataLoader(
        train_tokens, BATCH_SIZE, SEQ_LEN, rank, WORLD_SIZE
    )
    if rank == 0:
        print("Loaded train tokens from " + train_tokens)
        print("Number of tokens: " + String(train_loader.num_tokens))
    var val_loader = DataLoader(
        val_tokens, BATCH_SIZE, SEQ_LEN, rank, WORLD_SIZE
    )
    if rank == 0:
        print("Loaded val tokens from " + val_tokens)
        print("Number of tokens: " + String(val_loader.num_tokens))

    # Build the tokenizer from the tokens file.
    var tokenizer = Tokenizer("gpt2_tokenizer.bin")
    if rank == 0:
        print("Loaded tokenizer from " + "gpt2_tokenizer.bin")

    # Build the learning rate scheduler.
    var num_iters = 40
    var learning_rate_scheduler = LearningRateScheduler(
        "constant",
        learning_rate=Scalar[DType.float32](1e-4),
        warmup_steps=100,
        train_num_batches=num_iters,
        final_learning_rate_fraction=Scalar[DType.float32](1e-2),
    )

    # Initialize some memory for generating samples from the model.
    var rng_state = UInt64(1337)
    var gen_max_length = 64
    var gen_tokens = alloc[Scalar[DType.int32]](gen_max_length)
    var zero = 0
    var null_int32_ptr = MutMemPtr[DType.int32](unsafe_from_address=zero)

    # Optionally resume params, optimizer moments, RNG and dataloader position
    # from a prior checkpoint. Restoring the loader before the priming
    # next_batch() below means training continues over the exact data stream it
    # would have without the interruption.
    var start_step = 0
    if resume_from >= 0:
        var model_path = output_dir + "/model_" + String(resume_from) + ".bin"
        var state_path = (
            output_dir
            + "/state_"
            + String(resume_from)
            + "_"
            + String(rank)
            + ".bin"
        )
        var resumed = model.load_checkpoint(model_path, state_path)
        restore_dataloader_state(train_loader, resumed)
        rng_state = resumed.sampler_rng_state
        start_step = resumed.step
        if rank == 0:
            print("Resumed from checkpoint at step " + String(start_step))

    #####################################################################################
    # Training Loop
    #####################################################################################

    var val_loss_every = 10
    var val_max_steps = 10
    var sample_every = 20

    var elapsed_time_ms_total = 0.0

    train_loader.next_batch()

    for step in range(start_step, num_iters + 1):
        # Occasionally estimate validation loss.
        if val_loss_every > 0 and step % val_loss_every == 0:
            var time_val_start = global_perf_counter_ns()
            var val_loss = Float32(0.0)
            val_loader.reset()
            for _ in range(val_max_steps):
                val_loader.next_batch()
                model.forward(
                    val_loader.inputs, val_loader.targets, BATCH_SIZE, SEQ_LEN
                )
                val_loss += model.mean_loss
            val_loss /= Float32(val_max_steps)
            var time_val_end = global_perf_counter_ns()
            if rank == 0:
                print(
                    "validation step "
                    + String(step // val_loss_every)
                    + ": validation loss: "
                    + String(val_loss)
                    + " | total: "
                    + String(Float64(time_val_end - time_val_start) / 1e9)
                    + "s"
                )

        if sample_every > 0 and step > 0 and step % sample_every == 0:
            if rank == 0:
                gen_tokens[0] = Scalar[DType.int32](
                    tokenizer.eot_token
                )  # The GPT-2 EOT token begins generation.

                print("Generating:\n---")
                for t in range(1, gen_max_length):
                    # NOTE: Inference is wasteful here because for each t we recompute all activations between 0 and t.
                    # In a real inference setting we would use a separate code for this anyway.
                    # Inference is here only for sanity checking.
                    model.forward(gen_tokens, null_int32_ptr, 1, t)
                    var dev_logits_ptr = (
                        model.acts.logits
                        + (t - 1) * model.config.padded_vocab_size
                    )
                    model.ctx.enqueue_copy(
                        dst_ptr=rebind[
                            UnsafePointer[Scalar[GPT2_DTYPE], MutAnyOrigin]
                        ](
                            model.logits_host_buf.unsafe_ptr().as_unsafe_any_origin()
                        ),
                        src_ptr=rebind[
                            UnsafePointer[Scalar[GPT2_DTYPE], ImmutAnyOrigin]
                        ](dev_logits_ptr.as_unsafe_any_origin()),
                        size=model.config.vocab_size,
                    )
                    model.ctx.synchronize()

                    var logits = rebind[ImmutMemPtr[DType.float32]](
                        model.logits_host_buf.unsafe_ptr().as_unsafe_any_origin()
                    )
                    var coin = random_f32(rng_state)
                    var next_token = sample_softmax(
                        logits, model.config.vocab_size, coin
                    )
                    gen_tokens[t] = Scalar[DType.int32](next_token)
                    var token_str = tokenizer.decode(next_token)
                    safe_print(token_str)
                print("\n---")

        if step == num_iters:
            break

        var time_forward = global_perf_counter_ns()
        model.forward(
            train_loader.inputs, train_loader.targets, BATCH_SIZE, SEQ_LEN
        )
        model.ctx.synchronize()
        var time_backward = global_perf_counter_ns()

        model.backward()
        model.ctx.synchronize()
        var time_update = global_perf_counter_ns()

        model.update(
            UInt32(step + 1), learning_rate_scheduler.get_learning_rate(step)
        )
        model.ctx.synchronize()
        var time_load_batch = global_perf_counter_ns()

        # Checkpoint after the step's optimizer update but before advancing the
        # loader, so the saved dataloader position (current_sample_idx) is the
        # batch the next step will consume. The stored step is the next step to
        # run, matching how `start_step` resumes the loop.
        if save_every > 0 and (
            (step + 1) % save_every == 0 or step + 1 == num_iters
        ):
            var model_path = output_dir + "/model_" + String(step + 1) + ".bin"
            var state_path = (
                output_dir
                + "/state_"
                + String(step + 1)
                + "_"
                + String(rank)
                + ".bin"
            )
            model.write_checkpoint(
                model_path, state_path, step + 1, train_loader, rng_state
            )
            if rank == 0:
                print("Saved checkpoint at step " + String(step + 1))

        train_loader.next_batch()
        var time_end = global_perf_counter_ns()

        var elapsed_time = Float64(time_end - time_forward) / 1e9
        elapsed_time_ms_total += elapsed_time
        if rank == 0:
            print(
                "step "
                + String(step)
                + ": train loss "
                + String(model.mean_loss)
                + " | forward: "
                + String(Float64(time_backward - time_forward) / 1e9)
                + "s | backward: "
                + String(Float64(time_update - time_backward) / 1e9)
                + "s | update: "
                + String(Float64(time_load_batch - time_update) / 1e9)
                + "s | loader: "
                + String(Float64(time_end - time_load_batch) / 1e9)
                + "s | total: "
                + String(elapsed_time)
                + "s"
            )

    train_loader.close()
    val_loader.close()


def _dispatch_world_size[
    target: StaticString
](
    rank: Int,
    world_size: Int,
    cpu_coord: Optional[UnsafePointer[CpuCoordinator, MutUntrackedOrigin]],
    zero_stage: Int,
) raises:
    # train() is monomorphized on WORLD_SIZE (a comptime parameter), so a runtime
    # world_size can only dispatch to an instantiation compiled into this binary:
    # the common sizes below, plus whatever this binary was built for
    # (-D WORLD_SIZE=..., exposed as WORLD_SIZE_DEF). Listing WORLD_SIZE_DEF as
    # one branch is what makes larger sizes (16/32/64/...) reachable, without
    # compiling an instantiation for every possible size.
    if world_size == 1:
        train[target, 1](rank, cpu_coord, zero_stage)
    elif world_size == 2:
        train[target, 2](rank, cpu_coord, zero_stage)
    elif world_size == 4:
        train[target, 4](rank, cpu_coord, zero_stage)
    elif world_size == 8:
        train[target, 8](rank, cpu_coord, zero_stage)
    elif world_size == WORLD_SIZE_DEF:
        train[target, WORLD_SIZE_DEF](rank, cpu_coord, zero_stage)
    else:
        raise Error(
            "Unsupported world_size: "
            + String(world_size)
            + ". Supported values: 1, 2, 4, 8, or the compile-time WORLD_SIZE: "
            + String(WORLD_SIZE_DEF)
        )


def main() raises:
    var args = argv()
    var rank = 0
    var world_size = 1
    var zero_stage = 0

    var env_rank = getenv("RANK")
    if env_rank != "":
        rank = atol(env_rank)
    var env_world_size = getenv("WORLD_SIZE")
    if env_world_size != "":
        world_size = atol(env_world_size)
    var env_zero_stage = getenv("ZERO_STAGE")
    if env_zero_stage != "":
        zero_stage = atol(env_zero_stage)

    # Command line args override env variables:
    if len(args) > 1:
        world_size = atol(args[1])
    if len(args) > 2:
        zero_stage = atol(args[2])
    if len(args) > 3:
        rank = atol(args[3])

    print("Rank:", rank, "World size:", world_size, "ZeRO stage:", zero_stage)

    var use_cpu = getenv("LLMM_USE_CPU")
    var run_on_cpu = (
        (use_cpu != "") or not has_accelerator() or has_apple_gpu_accelerator()
    )

    if run_on_cpu:
        if has_apple_gpu_accelerator() and use_cpu == "":
            print(
                "==============================================================================="
            )
            print(
                "WARNING: Apple Silicon GPU training is disabled — using CPU"
                " instead."
            )
            print("")
            print(
                "An Apple Metal accelerator is present, but this trainer does"
                " not run on it yet."
            )
            print("Known blockers on Apple Silicon today:")
            print(
                "  • Metal / KGEN: several llmm GPU kernels compile in Mojo but"
                " fail at the"
            )
            print(
                '    metallib stage ("could not elaborate the generated KGEN")'
                " or miscompile"
            )
            print("    when lowered through MAX's Apple-GPU path.")
            print("")
            print(
                "CPU training is correct and fast enough for dev; GPU support"
                " here is WIP."
            )
            print(
                "==============================================================================="
            )

        print("Training on CPU.")
        if world_size == 1:
            _dispatch_world_size["cpu"](
                0,
                world_size,
                Optional[UnsafePointer[CpuCoordinator, MutUntrackedOrigin]](),
                zero_stage,
            )
        else:
            var cpu_coord_ptr = alloc[CpuCoordinator](1)
            cpu_coord_ptr[] = CpuCoordinator(world_size)

            @parameter
            def _run_rank(rank: Int):
                try:
                    _dispatch_world_size["cpu"](
                        rank, world_size, cpu_coord_ptr, zero_stage
                    )
                except e:
                    print("Rank", rank, "failed:", e)

            sync_parallelize[_run_rank](world_size)
            cpu_coord_ptr[].free()
            cpu_coord_ptr.free()
    elif has_accelerator():
        print("GPU detected — training on GPU.")
        _dispatch_world_size["gpu"](
            rank,
            world_size,
            Optional[UnsafePointer[CpuCoordinator, MutUntrackedOrigin]](),
            zero_stage,
        )
