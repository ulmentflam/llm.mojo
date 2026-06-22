from std.os import getenv
from std.time import global_perf_counter_ns
from std.gpu.host.info import is_cpu, is_gpu
from std.gpu.host import DeviceContext, HostBuffer, DeviceBuffer
from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from std.sys import has_accelerator, has_apple_gpu_accelerator
from std.memory import alloc, UnsafePointer, memcpy, memset_zero

from llmm.io import read_and_copy
from llmm.dataloader import DataLoader
from llmm.encoder import encoder_fwd, encoder_bwd, build_wte_buckets
from llmm.layernorm import (
    layernorm_fwd,
    layernorm_bwd,
    layernorm_fused_residual_fwd,
    layernorm_fused_residual_bwd,
)
from llmm.attention import attention_fwd, attention_bwd
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


# ===----------------------------------------------------------------------=== #
# Constants and Comptime Variables
# ===----------------------------------------------------------------------=== #


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


struct GPT2[target: StaticString]:
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

    def __init__(out self, checkpoint_path: String, ctx: DeviceContext) raises:
        self.ctx = ctx
        self.checkpoint_path = checkpoint_path
        self.has_allocated_params = False
        self.has_allocated_acts = False
        self.has_allocated_grads = False
        self.has_allocated_optimizer_moments = False

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

        self.params_buf = ctx.enqueue_create_buffer[GPT2_DTYPE](1)
        self.grads_buf = ctx.enqueue_create_buffer[GPT2_DTYPE](1)
        self.m_buf = ctx.enqueue_create_buffer[GPT2_DTYPE](1)
        self.v_buf = ctx.enqueue_create_buffer[GPT2_DTYPE](1)
        self.acts_buf = ctx.enqueue_create_buffer[GPT2_DTYPE](1)
        self.grad_acts_buf = ctx.enqueue_create_buffer[GPT2_DTYPE](1)
        self.inputs_buf = ctx.enqueue_create_host_buffer[DType.int32](1)
        self.targets_buf = ctx.enqueue_create_host_buffer[DType.int32](1)
        self.bucket_info_buf = ctx.enqueue_create_host_buffer[DType.int32](1)
        self.workload_indices_buf = ctx.enqueue_create_host_buffer[DType.int32](
            1
        )
        self.losses_host_buf = ctx.enqueue_create_host_buffer[GPT2_DTYPE](1)
        self.logits_host_buf = ctx.enqueue_create_host_buffer[GPT2_DTYPE](1)
        ctx.synchronize()

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
            self.num_parameters
        )
        self.ctx.enqueue_copy(dst_buf=self.params_buf, src_ptr=temp_ptr)
        self.ctx.synchronize()
        self.params_memory = rebind_mut_mem[GPT2_DTYPE](
            self.params_buf.unsafe_ptr().as_unsafe_any_origin()
        )
        self.params.point_parameters(self.param_sizes, self.params_memory)

        self.has_allocated_params = True

    def allocate_gradients(mut self) raises:
        self.grads = ParameterTensors[GPT2_DTYPE]()
        self.grads_buf = self.ctx.enqueue_create_buffer[GPT2_DTYPE](
            self.num_parameters
        )
        self.grads_buf.enqueue_fill(Scalar[GPT2_DTYPE](0.0))
        self.ctx.synchronize()
        self.grads_memory = rebind_mut_mem[GPT2_DTYPE](
            self.grads_buf.unsafe_ptr().as_unsafe_any_origin()
        )
        self.grads.point_parameters(self.param_sizes, self.grads_memory)
        self.num_grads = self.num_parameters
        self.has_allocated_grads = True

    def allocate_optimizer_moments(mut self) raises:
        self.m_buf = self.ctx.enqueue_create_buffer[GPT2_DTYPE](
            self.num_parameters
        )
        self.v_buf = self.ctx.enqueue_create_buffer[GPT2_DTYPE](
            self.num_parameters
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
        self.act_sizes[Activations.fch] = L * B * T * 4 * C
        self.act_sizes[Activations.fch_gelu] = L * B * T * 4 * C
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
            var l_fch = self.acts.fch + layer * batch_size * seq_len * (
                4 * channels
            )
            var l_fch_gelu = (
                self.acts.fch_gelu
                + layer * batch_size * seq_len * (4 * channels)
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

            # TODO: Race condition fix: Synchronize context before reading logits on CPU.
            self.ctx.synchronize()

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

        # TODO: Race condition fix: Synchronize context before reading logits on CPU.
        self.ctx.synchronize()

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
            var l_fch = self.acts.fch + layer * fch_layer_stride
            var l_fch_gelu = self.acts.fch_gelu + layer * fch_layer_stride
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
            var d_l_fch = self.grad_acts.fch + layer * fch_layer_stride
            var d_l_fch_gelu = (
                self.grad_acts.fch_gelu + layer * fch_layer_stride
            )
            var d_l_fc_proj = self.grad_acts.fc_proj + layer_offset

            var d_block_input: MutMemPtr[GPT2_DTYPE]
            if layer == 0:
                d_block_input = self.grad_acts.encoded
            else:
                d_block_input = (
                    self.grad_acts.residual_3 + (layer - 1) * layer_stride
                )

            # MLP projection backward.
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
        var config = AdamWConfig[GPT2_DTYPE](
            learning_rate=learning_rate,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            weight_decay=weight_decay,
            grad_scale=grad_scale,
        )
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


# ===----------------------------------------------------------------------=== #
# The Main Training Loop!
# ===----------------------------------------------------------------------=== #


def train[target: StaticString]() raises:
    var ctx: DeviceContext
    comptime if is_cpu[target]():
        ctx = DeviceContext(api="cpu")
    else:
        ctx = DeviceContext()

    # Build the GPT-2 model from my checkpoint.
    var model = GPT2[target]("gpt2_124M.bin", ctx)

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
    comptime SEQ_LEN = 64

    var train_loader = DataLoader(train_tokens, BATCH_SIZE, SEQ_LEN)
    print("Loaded train tokens from " + train_tokens)
    print("Number of tokens: " + String(train_loader.num_tokens))
    var val_loader = DataLoader(val_tokens, BATCH_SIZE, SEQ_LEN)
    print("Loaded val tokens from " + val_tokens)
    print("Number of tokens: " + String(val_loader.num_tokens))

    # Build the tokenizer from the tokens file.
    var tokenizer = Tokenizer("gpt2_tokenizer.bin")
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

    #####################################################################################
    # Training Loop
    #####################################################################################

    var val_loss_every = 10
    var val_max_steps = 10
    var grad_accum_steps = 10
    var sample_every = 20

    var elapsed_time_ms_total = 0.0

    for step in range(num_iters + 1):
        # Occasionally estimate validation loss.
        if val_loss_every > 0 and step % val_loss_every == 0:
            var val_loss = Float32(0.0)
            val_loader.reset()
            for _ in range(val_max_steps):
                val_loader.next_batch()
                model.forward(
                    val_loader.inputs, val_loader.targets, BATCH_SIZE, SEQ_LEN
                )
                val_loss += model.mean_loss
            val_loss /= Float32(val_max_steps)
            print("Validation loss: " + String(val_loss))

        if sample_every > 0 and step > 0 and step % sample_every == 0:
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
                    model.acts.logits + (t - 1) * model.config.padded_vocab_size
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

        var start_time = global_perf_counter_ns()
        train_loader.next_batch()
        model.forward(
            train_loader.inputs, train_loader.targets, BATCH_SIZE, SEQ_LEN
        )
        model.backward()
        model.update(
            UInt32(step + 1), learning_rate_scheduler.get_learning_rate(step)
        )
        model.ctx.synchronize()
        var elapsed_time = Float64(global_perf_counter_ns() - start_time) / 1e9
        elapsed_time_ms_total += elapsed_time
        print(
            "step "
            + String(step)
            + ": train loss "
            + String(model.mean_loss)
            + " | elapsed time "
            + String(elapsed_time)
            + " ms, average time "
            + String(elapsed_time_ms_total / Float64(step + 1))
            + " ms"
        )

    train_loader.close()
    val_loader.close()


def main() raises:
    var use_cpu = getenv("LLMM_USE_CPU")
    if use_cpu != "":
        print("LLMM_USE_CPU is set — forcing CPU training.")
        train["cpu"]()
        return
    comptime if has_apple_gpu_accelerator():
        print(
            "==============================================================================="
        )
        print(
            "WARNING: Apple Silicon GPU training is disabled — using CPU"
            " instead."
        )
        print("")
        print(
            "An Apple Metal accelerator is present, but this trainer does not"
            " run on it yet."
        )
        print("Known blockers on Apple Silicon today:")
        print(
            "  • Metal / KGEN: several llmm GPU kernels compile in Mojo but"
            " fail at the"
        )
        print(
            '    metallib stage ("could not elaborate the generated KGEN") or'
            " miscompile"
        )
        print("    when lowered through MAX's Apple-GPU path.")
        print("")
        print(
            "CPU training is correct and fast enough for dev; GPU support here"
            " is WIP."
        )
        print(
            "==============================================================================="
        )
        train["cpu"]()
    elif has_accelerator():
        print("GPU detected — training on GPU.")
        train["gpu"]()
    else:
        print("No GPU detected — training on CPU.")
        train["cpu"]()
