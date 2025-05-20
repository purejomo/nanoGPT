import math
import inspect
import sys
import re
from rich import print
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
import time
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
# Config
from gpt_conf import GPTConfig
# Checkpointing
import torch.utils.checkpoint as checkpoint
# Variations
from variations.lsv_variations import lsv_dictionary
from variations.softmax_variations import softmax_dictionary
from variations.norm_variations import norm_dictionary
from variations.position_encoding_variations import QuantizedEmbedding, RotaryEmbedding, SymmetricalOverlapAngularPositions, FIRE
from variations.activation_variations import activation_dictionary
from variations.linear_variations import linear_dictionary
from variations.router_variations import router_dictionary
from quantization.quantize import quantize_dictionary, dequantize, fake_quantize_act

# Attention Layer 전체에서 최대/최소 값 추적 (전역 변수)
global_attention_max = {
    "Q": -float("inf"),
    "K": -float("inf"),
    "V": -float("inf"),
    "Attention Score (Before Softmax)": -float("inf"),
    "Attention Score (After Softmax)": -float("inf"),
    "Attention Output (QK * V)": -float("inf"),
    "Projection Output": -float("inf"),
}

global_attention_min = {
    "Q": float("inf"),
    "K": float("inf"),
    "V": float("inf"),
    "Attention Score (Before Softmax)": float("inf"),
    "Attention Score (After Softmax)": float("inf"),
    "Attention Output (QK * V)": float("inf"),
    "Projection Output": float("inf"),
}

# 📌 2️⃣ 전역 함수로 정의 (forward() 밖)
def update_global_min_max(tensor, name):
    """
    주어진 Attention Layer의 tensor 값을 받아서
    전역 min/max 값을 업데이트하는 함수
    """
    global global_attention_max, global_attention_min
    
    # 만약 tensor가 None이거나 빈 tensor라면 업데이트하지 않음
    if tensor is None or tensor.numel() == 0:
        return

    tensor_max = torch.max(tensor).item()
    tensor_min = torch.min(tensor).item()

    # 최댓값 업데이트
    global_attention_max[name] = max(global_attention_max[name], tensor_max)
    # 최솟값 업데이트
    global_attention_min[name] = min(global_attention_min[name], tensor_min)
    
softmax_total_latency = 0  # 전체 softmax 연산 시간
softmax_total_count = 0  # 전체 softmax 실행 횟수
attention_total_latency = 0  # 전체 softmax 연산 시간
attention_total_count = 0  # 전체 softmax 실행 횟수
gpt_count = 0  # 전체 GPT 실행 횟수

def print_global_attention_statistics():
    """
    전체 Attention Layer의 tensor들에 대한 최댓값/최솟값을 출력하는 함수
    """
    print("\n🔍 Attention Layer Tensor Min/Max Statistics (Global Across Inference):")
    for key in global_attention_max.keys():
        print(f"📌 {key}:")
        print(f"   - 최대값: {global_attention_max[key]:.6f}")
        print(f"   - 최소값: {global_attention_min[key]:.6f}")
        
def log_softmax_latency(start_time, end_time):
    """Softmax 실행 시간과 횟수를 Global 변수에 저장"""
    global softmax_total_latency, softmax_total_count
    softmax_total_latency += (end_time - start_time)
    softmax_total_count += 1

def log_attention_time(start_time, end_time):
    global attention_total_latency, attention_total_count
    attention_total_latency += (end_time - start_time)
    attention_total_count += 1

def print_softmax_stats():
    """프로그램 종료 시 Softmax 실행 통계 출력"""
    global softmax_total_latency, softmax_total_count, attention_total_latency, attention_total_count, gpt_count
    avg_softmax = softmax_total_latency / softmax_total_count if softmax_total_count > 0 else 0
    avg_attn = attention_total_latency / attention_total_count if attention_total_count > 0 else 0
    print("\n🔥 Performance Statistics 🔥")
    print(f"📌 Total Attention Latency: {attention_total_latency:.12f} s ({attention_total_count} calls)")
    print(f"📌 Average Attention Time: {avg_attn:.12f} s/call")
    print(f"📌 Total Softmax Latency: {softmax_total_latency:.12f} s ({softmax_total_count} calls)")
    print(f"📌 Average Softmax Time: {avg_softmax:.12f} s/call")
    print(f"📌 Total GPT Forward Passes: {gpt_count}")

sparsity_list = []

def calculate_sparsity(att):
    """하삼각 행렬(Lower triangular matrix) 부분에서의 sparsity(0 값 비율) 계산"""
    B, nh, T, _ = att.shape  # Batch, num_heads, seq_len, seq_len
    # 하삼각 mask 만들기 (T x T)
    lower_triangular_mask = torch.tril(torch.ones(T, T, device=att.device))  # (T, T)
    # 전체 하삼각 요소 개수 (batch와 head 수를 고려하여 확장)
    total_elements = lower_triangular_mask.sum().item() * B * nh
    # 하삼각 부분만 선택한 후 0 값 개수 카운트
    zero_elements = torch.sum((att == 0) * lower_triangular_mask.unsqueeze(0).unsqueeze(0)).item()
    # sparsity 계산 (퍼센트 값)
    sparsity = (zero_elements / total_elements) * 100
    # ✅ sparsity 값을 리스트에 저장
    sparsity_list.append(sparsity)
    return sparsity  # 이제 즉시 출력하지 않고 리스트에만 저장

# 기존에 있던 sparsity_list처럼, QK/SV/Proj별로 따로 리스트를 둠
sparsity_qk_list = []
sparsity_sv_list = []
sparsity_proj_list = []

def calculate_sparsity_anyshape(tensor):
    """
    텐서 전체에서 0값이 차지하는 비율을 구해서 float으로 반환
    """
    zero_count = (tensor == 0).sum().item()
    total_count = tensor.numel()
    sparsity = (zero_count / total_count) * 100
    return sparsity

def print_average_sparsities():
    """ QK/SV/Projection 각각에 대해 누적된 sparsity의 평균을 출력 """
    # QK
    if sparsity_qk_list:
        avg_qk = sum(sparsity_qk_list) / len(sparsity_qk_list)
        print(f"🔥 평균 Q*K sparsity: {avg_qk:.12f}% (총 {len(sparsity_qk_list)}회)")
    else:
        print("⚠️ 저장된 Q*K sparsity 데이터가 없습니다.")

    # SV
    if sparsity_sv_list:
        avg_sv = sum(sparsity_sv_list) / len(sparsity_sv_list)
        print(f"🔥 평균 S*V sparsity: {avg_sv:.12f}% (총 {len(sparsity_sv_list)}회)")
    else:
        print("⚠️ 저장된 S*V sparsity 데이터가 없습니다.")

    # Projection
    if sparsity_proj_list:
        avg_proj = sum(sparsity_proj_list) / len(sparsity_proj_list)
        print(f"🔥 평균 Projection sparsity: {avg_proj:.12f}% (총 {len(sparsity_proj_list)}회)")
    else:
        print("⚠️ 저장된 Projection sparsity 데이터가 없습니다.")



def print_average_sparsity():
    """전체 실행이 끝난 후 평균 sparsity 값을 출력"""
    if sparsity_list:
        avg_sparsity = sum(sparsity_list) / len(sparsity_list)
        print(f"\n🔥 평균 Attention probabiltiy Sparsity: {avg_sparsity:.12f}%")
    else:
        print("\n⚠️ 저장된 sparsity 데이터가 없습니다.")

def create_shared_param_group(layer_type, config):
    # explore MoE layers being reflected symmetrically
    shared_size = None
    shared_sym = None # if true, output array is symmetrical
    layer_block = None
    shared_group = []

    if layer_type == "mlp":
        shared_size = config.shared_mlp_size
        shared_sym = config.shared_mlp_sym
    elif layer_type == "attn":
        shared_size = config.shared_attn_size
        shared_sym = config.shared_attn_sym
    else:
        sys.exit(f"{layer_type} not supported, exiting")

    # if attn layer check if using shared fire embeddings
    fire_pos_enc = None
    if layer_type == "attn" and config.shared_fire_embeddings:
        fire_pos_enc = FIRE(config, num_heads=config.n_head)

    for i in range (config.n_layer):
        # Create new layer block every "shared_size"
        if i % shared_size == 0:
            if layer_type == "mlp":
                if config.use_moe and i % config.moe_layer_freq == 0:
                    # this iter is an moe layer iter
                    layer_block = MoELayer(config)
                else:
                    layer_block = MLP(config)
            elif layer_type == "attn":
                layer_block = CausalSelfAttention(config, fire_pos_enc=fire_pos_enc)
            else:
                sys.exit(f"{layer_type} not supported, exiting")

        # Add layer block
        shared_group.append(layer_block)

        # If symmetrical and halfway, then mirror extend and exit
        if shared_sym:
            # Even
            if config.n_layer % 2 == 0:
                if i == (config.n_layer // 2 - 1):
                    # Append going backwards
                    for j in range(i+1):
                        shared_group.append(shared_group[i - j])
                    return shared_group
            # Odd
            else:
                if i == (config.n_layer // 2):
                    # Append going backwards
                    for j in range(i):
                        shared_group.append(shared_group[i - j])
                    return shared_group
    return shared_group

def set_variant(variant, default_variant):
    # If variant is false or None, then set to provided default value
    if not variant:
        return default_variant
    return variant

def create_activation_buffers(obj, arg):
    arg_str = arg.split("quantize_")[1]
    obj.register_buffer(arg_str, None)
    obj.register_buffer(f"{arg_str}_scale", None)
    obj.register_buffer(f"{arg_str}_zero_point", None)


class CausalSelfAttention(nn.Module):
    def __init__(self, config, fire_pos_enc=None):
        super().__init__()

        self.full_quant_iteration = config.full_quant_iteration
        self.eval_interval = config.eval_interval
        self.start_quant_level = config.start_quant_level
        self.quant_scheduler = config.quant_scheduler

        if (config.n_kv_group == None):
            config.n_kv_group = config.n_head
        else:
            assert config.n_embd % config.n_kv_group == 0

        self.quantization_attn_dict = {}
        self.quantization_attn_dict["activations_quant_method"] = config.activations_quant_method
        for arg, val in vars(config).items():
            # Set each attention Activation precision and method
            if arg.startswith("quantize_") and "attn_act" in arg and arg.endswith("_bits"):
                self.quantization_attn_dict[arg] = set_variant(val, config.quantize_attn_act_bits)
            elif arg.startswith("quantize_") and "attn_act" in arg:
                self.quantization_attn_dict[arg] = set_variant(val, config.quantize_attn_act)
                if config.store_activations and arg != "quantize_attn_act" and self.quantization_attn_dict[arg]:
                    create_activation_buffers(self, arg)
            # Set each attention Linear precision and method
            elif arg.startswith("quantize_") and "linear_attn" in arg and arg.endswith("_bits"):
                self.quantization_attn_dict[arg] = set_variant(val, config.quantize_linear_bits)
            elif arg.startswith("quantize_") and "linear_attn" in arg and arg.endswith("_method"):
                self.quantization_attn_dict[arg] = set_variant(val, config.quantize_linear_method)

        self.linear_variant_q = linear_dictionary[set_variant(config.linear_variant_q, config.linear_variant_attn)]
        self.linear_variant_k = linear_dictionary[set_variant(config.linear_variant_k, config.linear_variant_attn)]
        self.linear_variant_v = linear_dictionary[set_variant(config.linear_variant_v, config.linear_variant_attn)]
        self.linear_variant_attn_proj = linear_dictionary[set_variant(config.linear_variant_attn_proj, config.linear_variant_attn)]

        # key, query, value projections for all heads, but in a batch
        self.c_attn_q = self.linear_variant_q(config.n_embd, config.n_embd, config, self.quantization_attn_dict["quantize_linear_attn_q_method"], self.quantization_attn_dict["quantize_linear_attn_q_bits"], bias=config.bias)

        self.n_head = config.n_head
        if config.n_kv_group == None:
            self.n_kv_group = config.n_head
        else:
            assert config.n_head % config.n_kv_group == 0
            self.n_kv_group = config.n_kv_group

        self.kv_dim = (config.n_embd // config.n_head) * self.n_kv_group
        self.c_attn_k = self.linear_variant_k(config.n_embd, self.kv_dim, config, self.quantization_attn_dict["quantize_linear_attn_k_method"], self.quantization_attn_dict["quantize_linear_attn_k_bits"], bias=config.bias)
        self.c_attn_v = self.linear_variant_v(config.n_embd, self.kv_dim, config, self.quantization_attn_dict["quantize_linear_attn_v_method"], self.quantization_attn_dict["quantize_linear_attn_v_bits"], bias=config.bias)
        self.c_proj = self.linear_variant_attn_proj(config.n_embd, config.n_embd, config, self.quantization_attn_dict["quantize_linear_attn_proj_method"], self.quantization_attn_dict["quantize_linear_attn_proj_bits"], bias=config.bias)

        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout

        # Embedding
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.n_embd = config.n_embd
        self.gate = config.gate
        self.use_fire_embeddings = None
        ## self.disable_flash_attention = config.disable_flash_attention        ## GH
        self.disable_flash_attention = True                                     ## GH
        if config.use_fire_embeddings:
            self.use_fire_embeddings = config.use_fire_embeddings
            if fire_pos_enc is not None:
                self.fire_pos_enc = fire_pos_enc
                print("shared fire")
            else:
                self.fire_pos_enc = FIRE(config, num_heads=config.n_head)
                print("indiv fire")

        # Rotary Positional Embeddings
        self.rotary_emb_q = None
        self.rotary_emb_k = None
        if config.use_rotary_embeddings:
            # Note: size is the size of the head dimension
            if config.rope_variant == "soap":
                self.sym_rot_num_angles = config.sym_rot_num_angles
                self.rotary_emb_q = SymmetricalOverlapAngularPositions(config, size=config.n_embd // self.n_head, num_angles=self.sym_rot_num_angles)
                self.rotary_emb_k = SymmetricalOverlapAngularPositions(config, size=config.n_embd // self.n_head, num_angles=self.sym_rot_num_angles)
            elif config.rope_variant == "rope":
                self.rotary_emb_q = RotaryEmbedding(config, size=config.n_embd // self.n_head)
                self.rotary_emb_k = RotaryEmbedding(config, size=config.n_embd // self.n_head)

        # Sliding window size
        self.window_size = config.window_size
        ## print(f"sliding window size: {self.window_size}")

        # Using flex attention
        self.use_flex_attn = config.use_flex_attn

        # Gating
        self.gate = config.gate

        # Fire Embeddings
        self.use_fire_embeddings = None
        if config.use_fire_embeddings:
            self.use_fire_embeddings = config.use_fire_embeddings
            if fire_pos_enc is not None:
                self.fire_pos_enc = fire_pos_enc
                print("shared fire")
            else:
                self.fire_pos_enc = FIRE(config, num_heads=config.n_head)
                print("indiv fire")

        # Rotary Positional Embeddings
        self.rotary_emb_q = None
        self.rotary_emb_k = None
        if config.use_rotary_embeddings:
            # Note: size is the size of the head dimension
            if config.rope_variant == "soap":
                self.sym_rot_num_angles = config.sym_rot_num_angles
                self.rotary_emb_q = SymmetricalOverlapAngularPositions(config, size=config.n_embd // self.n_head, num_angles=self.sym_rot_num_angles)
                self.rotary_emb_k = SymmetricalOverlapAngularPositions(config, size=config.n_embd // self.n_head, num_angles=self.sym_rot_num_angles)
            elif config.rope_variant == "rope":
                self.rotary_emb_q = RotaryEmbedding(config, size=config.n_embd // self.n_head)
                self.rotary_emb_k = RotaryEmbedding(config, size=config.n_embd // self.n_head)

        self.flash = True
        if self.window_size is not None:
            # TODO: look into supporting sliding window attn for flash attn
            self.flash = False
            print("flash attention removed due to windowed attention")

        if self.n_kv_group != self.n_head:
            self.flash = False
            print("flash attention removed due to GQA")

        if self.use_fire_embeddings:
            self.flash = False
            print("flash attention removed due to FIRE")

        # Can't use flash attention if we want to manually quantize most input/output activations in attn
        for key, val in self.quantization_attn_dict.items():
            if key.startswith("quantize_") and val == True:
                self.flash = False
                print("flash attention removed due to Quantization")
                break

        if self.disable_flash_attention:
            self.flash = False

        # Softmax Variant Selection
        self.softmax_variant_attn = config.softmax_variant_attn
        if self.softmax_variant_attn == "softmax":
            # Enable flash attention, which is compatible with 'softmax'
            if self.disable_flash_attention or self.flash == False:
                print("setting non-flash softmax attn")
            else:
                self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
                print("setting flash attn")
        else:
            # Remove flash attention (only compatible with 'softmax')
            print("flash attention removed due to softmax alternative")
            self.flash = False
            # Set softmax_layer_attn to custom softmax alternative
            self.softmax_layer_attn = softmax_dictionary[config.softmax_variant_attn](config)

        if (not self.flash) and (not self.use_flex_attn):
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))


    # Flex Attention Related
    def sliding_window_causal(self, b, h, q_idx, kv_idx):
        causal_mask = q_idx >= kv_idx
        window_mask = q_idx - kv_idx <= self.window_size
        return causal_mask & window_mask

    def get_block_mask(self, T, device):
        if T not in self.block_masks:
            block_mask = create_block_mask(
                    self.sliding_window_causal,
                    B=None,
                    H=None,
                    Q_LEN=T,
                    KV_LEN=T,
                    device=device
                    )
            self.block_masks[T] = block_mask
        else:
            block_mask = self.block_masks[T]
        return block_mask
    # End Flex Attention Related

    def forward(self, x, iter_num):
        global attention_total_latency, attention_total_count               # Global variables to track attention latency
        B, T, C = x.size()      # batch size, sequence length, embedding dimensionality (n_embd)

        if self.quantization_attn_dict["quantize_attn_act_input"]:
            num_bits = self.quantization_attn_dict["quantize_attn_act_input_bits"]
            quant_method = self.quantization_attn_dict["activations_quant_method"]
            x = fake_quantize_act(self, "attn_act_input", x, num_bits, quant_method, iter_num)

        q = self.c_attn_q(x)
        k = self.c_attn_k(x)
        v = self.c_attn_v(x)

        if self.window_size is not None:
            if self.use_flex_attn is not None:
                self.block_masks = {}
            else:
                self.window_mask = torch.ones((1, 1, T, T), device=x.device)
                self.window_mask = torch.triu(self.window_mask, diagonal=-self.window_size)
                self.window_mask = self.bias[:,:,:T,:T] * self.window_mask

        if self.gate:
            if self.n_kv_group == self.n_head:
                Gating = nn.Linear(self.n_embd, self.n_embd, bias=True, device=x.device)
                gate_ = torch.sigmoid(Gating(x))
                q = q * gate_
                k = k * gate_
                v = v * gate_
            else:
                # TODO: Test more methods to merge Attention Gates with GQA
                # TODO: Evaluate each method's ability to even out parameter sizes
                Gating_q = nn.Linear(self.n_embd, self.n_embd, bias=True, device=x.device)
                Gating_kv = nn.Linear(self.n_embd, self.kv_dim, bias=True, device=x.device)
                gate_qx = Gating_q(x)
                gate_q = torch.sigmoid(gate_qx)
                gate_kv = torch.sigmoid(Gating_kv(gate_qx))
                q = q * gate_q
                k = k * gate_kv
                v = v * gate_kv

        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, n_h, T, hs)
        k = k.view(B, T, self.n_kv_group, C // self.n_head).transpose(1, 2) # (B, n_kv, T, hs)
        v = v.view(B, T, self.n_kv_group, C // self.n_head).transpose(1, 2) # (B, n_kv, T, hs)

        # rotate q and k before evaluating with the heads
        if (self.rotary_emb_q is not None) and (self.rotary_emb_k is not None):
            q = self.rotary_emb_q(q)
            k = self.rotary_emb_k(k)

        start_time = time.perf_counter()  # Start timing
        y = None
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        elif self.use_flex_attn and self.window_size is not None:
            block_mask = self.get_block_mask(T, x.device)
            y = torch.nn.attention.flex_attention.flex_attention(q, k, v, block_mask=block_mask)
        else:
            if self.quantization_attn_dict["quantize_attn_act_qk_mult_q_input"]:
                num_bits = self.quantization_attn_dict["quantize_attn_act_qk_mult_q_input_bits"]
                quant_method = self.quantization_attn_dict["activations_quant_method"]
                q = fake_quantize_act(self, "attn_act_qk_mult_q_input", q, num_bits, quant_method, iter_num)
            if self.quantization_attn_dict["quantize_attn_act_qk_mult_k_input"]:
                num_bits = self.quantization_attn_dict["quantize_attn_act_qk_mult_k_input_bits"]
                quant_method = self.quantization_attn_dict["activations_quant_method"]
                k = fake_quantize_act(self, "attn_act_qk_mult_k_input", k, num_bits, quant_method, iter_num)

            att = None
            # manual implementation of attention
            # start_time = time.perf_counter()
            if self.n_head != self.n_kv_group:
                k_repeated = k.repeat_interleave(self.n_head // self.n_kv_group, dim=1)
                att = (q @ k_repeated.transpose(-2, -1)) / math.sqrt(k.size(-1))
            else:
                att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

            qk_sparsity = calculate_sparsity_anyshape(att)
            sparsity_qk_list.append(qk_sparsity)

            end_time = time.perf_counter()
            # print(f"time {end_time - start_time}")

            # apply masks
            if self.window_size is not None:
                # add mask for sliding window attention
                att = att.masked_fill(self.window_mask == 0, float('-inf'))
            else:
                # regular lower triangle attention
                att = att.masked_fill(self.bias[:,:,:T,:T].to(x.device) == 0, float('-inf'))

            # fire position embeddings
            if self.use_fire_embeddings is not None:
                # add learned fire bias
                att = att + self.fire_pos_enc(x)

            if self.quantization_attn_dict["quantize_attn_act_softmax_input"]:
                num_bits = self.quantization_attn_dict["quantize_attn_act_softmax_input_bits"]
                quant_method = self.quantization_attn_dict["activations_quant_method"]
                att = fake_quantize_act(self, "attn_act_softmax_input", att, num_bits, quant_method, iter_num, causal_mask=True)
            
            # softmax variation
            start_time = time.perf_counter()
            if self.softmax_variant_attn != 'softmax':
                att = self.softmax_layer_attn(att)
            else:
                att = F.softmax(att, dim=-1)  # Softmax 연산
                print("eager attention")
            end_time = time.perf_counter()
            if att is None:
                print("⚠️ Softmax 이후 att 값이 None입니다!")
            sparsity = calculate_sparsity(att)

            # ✅ Softmax 이후 0 값이 있는지 체크
            zero_count = (att == 0).sum().item()
            print(f"⚠️ Softmax 이후 0값 개수: {zero_count} / {att.numel()} ({zero_count / att.numel() * 100:.2f}%)")

            # ✅ Softmax 이후 최소/최대값 확인
            print(f"🔍 Softmax 후 att 최소값: {att.min().item()}, 최대값: {att.max().item()}")

            # ✅ 0에 가까운 값이 얼마나 있는지 확인
            near_zero_count = (att < 1e-10).sum().item()
            print(f"⚠️ 0에 가까운 값 개수 (< 1e-10): {near_zero_count} / {att.numel()}")


            # ✅ Global 변수 업데이트
            log_softmax_latency(start_time, end_time)

            att = self.attn_dropout(att)

            if self.quantization_attn_dict["quantize_attn_act_pv_mult_p_input"]:
                num_bits = self.quantization_attn_dict["quantize_attn_act_pv_mult_p_input_bits"]
                quant_method = self.quantization_attn_dict["activations_quant_method"]
                att = fake_quantize_act(self, "attn_act_pv_mult_p_input", att, num_bits, quant_method, iter_num)
            if self.quantization_attn_dict["quantize_attn_act_pv_mult_v_input"]:
                num_bits = self.quantization_attn_dict["quantize_attn_act_pv_mult_v_input_bits"]
                quant_method = self.quantization_attn_dict["activations_quant_method"]
                v = fake_quantize_act(self, "attn_act_pv_mult_v_input", v, num_bits, quant_method, iter_num)

            if self.n_head != self.n_kv_group:
                v_repeated = v.repeat_interleave(self.n_head // self.n_kv_group, dim=1)
                y = att @ v_repeated # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
            else:
                y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
            sv_sparsity = calculate_sparsity_anyshape(y)
            sparsity_sv_list.append(sv_sparsity)

        if self.quantization_attn_dict["quantize_attn_act_pv_mult_output"]:
            num_bits = self.quantization_attn_dict["quantize_attn_act_pv_mult_output_bits"]
            quant_method = self.quantization_attn_dict["activations_quant_method"]
            y = fake_quantize_act(self, "attn_act_pv_mult_output", y, num_bits, quant_method, iter_num)

        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))

        proj_sparsity = calculate_sparsity_anyshape(y)
        sparsity_proj_list.append(proj_sparsity)

        if self.quantization_attn_dict["quantize_attn_act_output"]:
            num_bits = self.quantization_attn_dict["quantize_attn_act_output_bits"]
            quant_method = self.quantization_attn_dict["activations_quant_method"]
            y = fake_quantize_act(self, "attn_act_output", y, num_bits, quant_method, iter_num)

        end_time = time.perf_counter()  # End timing
        attention_total_latency += (end_time - start_time)
        attention_total_count += 1

        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.full_quant_iteration = config.full_quant_iteration
        self.eval_interval = config.eval_interval

        # Select "mlp variant"
        self.mlp_variant = config.mlp_variant

        self.start_quant_level = config.start_quant_level
        self.quant_scheduler = config.quant_scheduler

        # If "MLP Variant" is KAN, then we skip MLP specific items
        if self.mlp_variant == "kan":
            self.kan = linear_dictionary["kan"](config.n_embd, config.n_embd, config=config)
        else:
            # Select activation variant
            self.activation_variant = activation_dictionary[config.activation_variant](config=config)

            # Sets the class of linear for MLP
            self.linear_variant_mlp_up = linear_dictionary[set_variant(config.linear_variant_mlp_up, config.linear_variant_mlp)]
            self.linear_variant_mlp_down = linear_dictionary[set_variant(config.linear_variant_mlp_down, config.linear_variant_mlp)]

            self.quantization_mlp_dict = {}
            self.quantization_mlp_dict["activations_quant_method"] = config.activations_quant_method

            # Set quantization parameters for MLP
            for arg, val in vars(config).items():
                # Set MLP Activation precision and quantization method
                if arg.startswith("quantize_") and "mlp_act" in arg and arg.endswith("_bits"):
                    self.quantization_mlp_dict[arg] = set_variant(val, config.quantize_mlp_act_bits)
                elif arg.startswith("quantize_") and "mlp_act" in arg:
                    self.quantization_mlp_dict[arg] = set_variant(val, config.quantize_mlp_act)
                    if config.store_activations and arg != "quantize_mlp_act" and self.quantization_mlp_dict[arg]:
                        create_activation_buffers(self, arg)
                # Set MLP Linear Weight precision and quantization method
                elif arg.startswith("quantize_") and "linear_mlp" in arg and arg.endswith("_bits"):
                    self.quantization_mlp_dict[arg] = set_variant(val, config.quantize_linear_bits)
                elif arg.startswith("quantize_") and "linear_mlp" in arg and arg.endswith("_method"):
                    self.quantization_mlp_dict[arg] = set_variant(val, config.quantize_linear_method)

            # Instantiate Linear Layers
            if self.mlp_variant == "mlp":
                self.c_fc = self.linear_variant_mlp_up(config.n_embd, config.mlp_expansion_factor * config.n_embd, config, self.quantization_mlp_dict["quantize_linear_mlp_up_method"], self.quantization_mlp_dict["quantize_linear_mlp_up_bits"], bias=config.bias)
                self.c_proj = self.linear_variant_mlp_down(config.mlp_expansion_factor * config.n_embd, config.n_embd, config, self.quantization_mlp_dict["quantize_linear_mlp_down_method"], self.quantization_mlp_dict["quantize_linear_mlp_down_bits"], bias=config.bias)
            elif self.mlp_variant == "swiglu":
                self.c_fc_in1 = self.linear_variant_mlp_up(config.n_embd, config.mlp_expansion_factor * config.n_embd, config, self.quantization_mlp_dict["quantize_linear_mlp_up_method"], self.quantization_mlp_dict["quantize_linear_mlp_up_bits"])
                self.c_fc_in2 = self.linear_variant_mlp_up(config.n_embd, config.mlp_expansion_factor * config.n_embd, config, self.quantization_mlp_dict["quantize_linear_mlp_up_method"], self.quantization_mlp_dict["quantize_linear_mlp_up_bits"])
                self.c_fc_out = self.linear_variant_mlp_down(config.mlp_expansion_factor * config.n_embd, config.n_embd, config, self.quantization_mlp_dict["quantize_linear_mlp_down_method"], self.quantization_mlp_dict["quantize_linear_mlp_down_bits"])

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, iter_num):

        if self.quantization_mlp_dict["quantize_mlp_act_input"]:
            num_bits = self.quantization_mlp_dict["quantize_mlp_act_input_bits"]
            quant_method = self.quantization_mlp_dict["activations_quant_method"]
            x = fake_quantize_act(self, "mlp_act_input", x, num_bits, quant_method, iter_num)

        if self.mlp_variant == "kan":
            x = self.kan(x)

        elif self.mlp_variant == "mlp":
            x = self.c_fc(x)

            if self.quantization_mlp_dict["quantize_mlp_act_activation_input"]:
                num_bits = self.quantization_mlp_dict["quantize_mlp_act_activation_input_bits"]
                quant_method = self.quantization_mlp_dict["activations_quant_method"]
                x = fake_quantize_act(self, "mlp_act_activation_input", x, num_bits, quant_method, iter_num)

            x = self.activation_variant(x)

            if self.quantization_mlp_dict["quantize_mlp_act_activation_output"]:
                num_bits = self.quantization_mlp_dict["quantize_mlp_act_activation_output_bits"]
                quant_method = self.quantization_mlp_dict["activations_quant_method"]
                x = fake_quantize_act(self, "mlp_act_activation_output", x, num_bits, quant_method, iter_num)

            x = self.c_proj(x)

        elif self.mlp_variant == "swiglu":
            x_in1 = self.c_fc_in1(x)

            if self.quantization_mlp_dict["quantize_mlp_act_activation_input"]:
                num_bits = self.quantization_mlp_dict["quantize_mlp_act_activation_input_bits"]
                quant_method = self.quantization_mlp_dict["activations_quant_method"]
                x_in1 = fake_quantize_act(self, "mlp_act_activation_input", x_in1, num_bits, quant_method, iter_num)

            x_in1 = self.activation_variant(x_in1)

            if self.quantization_mlp_dict["quantize_mlp_act_activation_output"]:
                num_bits = self.quantization_mlp_dict["quantize_mlp_act_activation_output_bits"]
                quant_method = self.quantization_mlp_dict["activations_quant_method"]
                x_in1 = fake_quantize_act(self, "mlp_act_activation_output", x_in1, num_bits, quant_method, iter_num)

            x_in2 = self.c_fc_in2(x)
            x_out = x_in1 * x_in2
            x = self.c_fc_out(x_out)

        x = self.dropout(x)

        if self.quantization_mlp_dict["quantize_mlp_act_output"]:
            num_bits = self.quantization_mlp_dict["quantize_mlp_act_output_bits"]
            quant_method = self.quantization_mlp_dict["activations_quant_method"]
            x = fake_quantize_act(self, "mlp_act_output", x, num_bits, quant_method, iter_num)
        return x

class Block(nn.Module):
    def __init__(self, config, mlp=None, attn=None):
        super().__init__()

        # Initialize and set attn normalization (e.g. rmsnorm)
        norm_variant_attn = norm_dictionary[config.norm_variant_attn]
        self.ln_1 = norm_variant_attn(config)
        if not config.use_parallel_mlp:
            self.ln_2 = norm_variant_attn(config)

        self.use_post_ln = config.use_post_ln
        self.use_parallel_mlp = config.use_parallel_mlp
        self.use_gradient_checkpointing = config.use_gradient_checkpointing

        # Allow for sharing attn between blocks
        if attn is None:
            self.attn = CausalSelfAttention(config)
        else:
            self.attn = attn

        # Allow for sharing mlp between blocks
        if mlp is None:
            self.mlp = MLP(config)
        else:
            self.mlp = mlp
        
    def forward(self, x, iter_num):
        def custom_forward(*inputs):
            x = inputs[0]
            if self.use_post_ln:
                if self.use_parallel_mlp:
                    x = self.ln_1(x + self.attn(x, iter_num) + self.mlp(x, iter_num))
                else:
                    x = self.ln_1(x + self.attn(x, iter_num))
                    x = self.ln_2(x + self.mlp(x, iter_num))
            else:
                if self.use_parallel_mlp:
                    ln_1 = self.ln_1(x)
                    x = x + self.attn(ln_1, iter_num) + self.mlp(ln_1, iter_num)
                else:
                    x = (x + self.attn(self.ln_1(x), iter_num))
                    x = (x + self.mlp(self.ln_2(x), iter_num))
            return x

        if self.use_gradient_checkpointing and x.requires_grad:
            return checkpoint.checkpoint(custom_forward, x, use_reentrant=False)
        else:
            return custom_forward(x)

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None

        self.config = config

        # Shared Parameters MLP
        shared_mlp_array = create_shared_param_group("mlp", config)
        # Shared Parameters Attention
        shared_attn_array = create_shared_param_group("attn", config)

        # Factorization Parameters
        self.n_embd_wte = config.n_embd_wte
        self.n_embd_wte_scale_tying = config.n_embd_wte_scale_tying

        # Learned Steering Vectors
        self.use_lsv = config.use_lsv
        self.lsv_index = config.lsv_index
        self.lsv_dataset_num = config.lsv_dataset_num

        if config.lsv_dataset_num is not None and config.use_lsv:
            self.num_datasets = config.lsv_dataset_num
            print(config.lsv_variant)
            self.lsv_variant = config.lsv_variant
            self.lsv_matrix = lsv_dictionary[self.lsv_variant](config)

        # Configure wte, with optional quantization and factoring
        if config.quantize_wte:
            if config.n_embd_wte:
                # If factorization is set
                word_embd = QuantizedEmbedding(config.vocab_size, config.n_embd_wte, config.quantize_wte_method, config.quantize_wte_bits)
            else:
                # no factorization
                word_embd = QuantizedEmbedding(config.vocab_size, config.n_embd, config.quantize_wte_method, config.quantize_wte_bits)
        else:
            if config.n_embd_wte:
                # If factorization is set
                word_embd = nn.Embedding(config.vocab_size, config.n_embd_wte)
            else:
                # no factorization
                word_embd = nn.Embedding(config.vocab_size, config.n_embd)

        self.transformer = nn.ModuleDict(dict(
            wte = word_embd,
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config, mlp=shared_mlp_array[i], attn=shared_attn_array[i]) for i in range(config.n_layer)]),
            ln_f = norm_dictionary[config.norm_variant_output](config),
        ))

        if self.config.use_abs_pos_embeddings:
            if config.quantize_wpe:
                pos_embd = QuantizedEmbedding(config.block_size, config.n_embd, config.quantize_wpe_method, config.quantize_wpe_bits)
            else:
                pos_embd = nn.Embedding(config.block_size, config.n_embd)
            self.transformer['wpe'] = pos_embd

        # Select softmax variant for output layer
        self.softmax_variant_output = config.softmax_variant_output
        if self.softmax_variant_output != "softmax":
            self.softmax_layer_output = softmax_dictionary[config.softmax_variant_output](config)

        if config.n_embd_wte:
            self.lm_head = nn.Linear(config.n_embd_wte, config.vocab_size, bias=False)
        else:
            self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # Initialize and possibly import scale_up and scale_down matrices, if factorization is set
        if self.n_embd_wte:
            # TODO: make this linear set from variant dictionary
            # TODO: make this linear quantizable
            self.transformer['scale_up'] = nn.Linear(config.n_embd_wte, config.n_embd, bias=False)
            self.transformer['scale_down'] = nn.Linear(config.n_embd_wte, config.n_embd, bias=False)

            if self.n_embd_wte_scale_tying:
                self.transformer.scale_up.weight = self.transformer.scale_down.weight # Weight tying

            if config.import_scale_matrices_freeze:
                self.transformer.scale_up.weight.requires_grad = False
                self.transformer.scale_down.weight.requires_grad = False

        # init all weights
        self.apply(self._init_weights)

        # import wte
        if self.config.import_wte_npy:
            # Replace wte with values from numpy and retie weights
            self.import_wte(self.config.import_wte_npy)

        # import scale_matrices
        if config.import_scale_matrices_npz:
            self.import_scale_matrices(config.import_scale_matrices_npz, config.n_embd_wte_scale_tying)

        for pn, p in self.named_parameters():
            # apply special scaled init to the residual projections, per GPT-2 paper
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))


    def get_num_params(self, non_embedding=True):       ## 여기서 True 좀 잘 봐야할듯 이거 좀 잘 봐야할듯, rotary positional embedding 하게되면 분명 차이가 있을것으로 생각     GH
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and self.config.use_abs_pos_embeddings:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def update_block_size(self, new_block_size):
        # Function to increase block size dynamically
        if new_block_size > self.config.block_size:
            self.config.block_size = new_block_size
            if self.config.use_abs_pos_embeddings:
                if self.config.quantize_wpe:
                    pos_embd = QuantizedEmbedding(new_block_size, self.config.n_embd, self.config.quantize_wpe_method, self.config.quantize_wpe_bits)
                else:
                    pos_embd = nn.Embedding(new_block_size, self.config.n_embd)
                self.transformer.wpe = pos_embd
            for block in self.transformer.h:
                if hasattr(block.attn, 'bias'):
                    block.attn.bias = torch.tril(torch.ones(new_block_size, new_block_size)).view(1, 1, new_block_size, new_block_size)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=self.config.linear_mean_init, std=self.config.linear_std_init)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=self.config.embedding_mean_init, std=self.config.embedding_std_init)

    def update_num_angles(self, num_angles):
        """Update the number of angles for rotary embeddings in all attention layers."""
        device = next(self.parameters()).device
        for block in self.transformer.h:
            if hasattr(block.attn, 'rotary_emb_q') and hasattr(block.attn, 'rotary_emb_k'):
                block.attn.rotary_emb_q.update_num_angles(num_angles, device)
                block.attn.rotary_emb_k.update_num_angles(num_angles, device)

    def update_rope_length(self, rope_length):
        """Update the number of angles for rotary embeddings in all attention layers."""
        for block in self.transformer.h:
            if hasattr(block.attn, 'rotary_emb_q') and hasattr(block.attn, 'rotary_emb_k'):
                block.attn.rotary_emb_q.update_rope_length(rope_length)
                block.attn.rotary_emb_k.update_rope_length(rope_length)

    def import_wte(self, file_path):
        """ Replace wte with values from numpy and retie weights """

        #Load and format weights
        initial_embeddings = np.load(self.config.import_wte_npy)
        initial_embeddings_tensor = torch.from_numpy(initial_embeddings).float()

        # Initialize imported wte
        self.transformer.wte = nn.Embedding.from_pretrained(
                initial_embeddings_tensor,
                freeze=self.config.import_wte_freeze
                )

        # Redo the Weight tying
        self.lm_head.weight = self.transformer.wte.weight

    def export_wte(self, file_path):
        # TODO: Determine strategy with this and other means of export, possibly
        # replacing this with composition of existing means
        embedding_table = self.transformer.wte.weight.detach().cpu().numpy()
        np.save(file_path, embedding_table)
        print(f"Embedding table saved to {file_path}")

    def import_scale_matrices(self, file_path, weight_tying=False):
        """Import scale_up and scale_down matrices from a numpy file."""
        scale_matrices = np.load(file_path)
        scale_up_tensor = torch.from_numpy(scale_matrices['scale_up']).float().T
        scale_down_tensor = torch.from_numpy(scale_matrices['scale_down']).float().T

        print(scale_up_tensor.size())
        print(scale_down_tensor.size())
        self.transformer.scale_up.weight.data.copy_(scale_up_tensor)
        self.transformer.scale_down.weight.data.copy_(scale_down_tensor)

        if weight_tying:
            self.transformer.scale_up.weight = self.transformer.scale_down.weight

        print(f"Scale matrices loaded from {file_path} with weight tying: {weight_tying}")

    def export_scale_matrices(self, file_path):
        """Export scale_up and scale_down matrices to a numpy file."""
        scale_up_matrix = self.transformer.scale_up.weight.detach().cpu().numpy()
        scale_down_matrix = self.transformer.scale_down.weight.detach().cpu().numpy()

        np.savez(file_path, scale_up=scale_up_matrix, scale_down=scale_down_matrix)
        print(f"Scale matrices saved to {file_path}")

    def forward(self, idx, targets=None, iter_num=None):
        global gpt_count
        device = idx.device
        b, t = idx.size()
        # assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        x = None

        if self.n_embd_wte:
            tok_emb = self.transformer.scale_up(tok_emb)
        if self.config.use_abs_pos_embeddings:
            pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)
            pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
            x = self.transformer.drop(tok_emb + pos_emb)
        else:
            x = self.transformer.drop(tok_emb)

        x.requires_grad_(True)  # Ensure requires_grad is True

        if self.use_lsv and self.config.apply_lsv_at_layer_idx == 0:
            x = self.lsv_matrix(x)

        layer = 1
        for block in self.transformer.h:
            # Propagate tokens through layers
            if self.config.use_gradient_checkpointing:
                x = checkpoint.checkpoint(block, x, iter_num, use_reentrant=self.config.recompute_backward_pass)
            else:
                x = block(x, iter_num)

            # Intercept for Learned Steering Vectors
            if self.use_lsv and layer == self.config.apply_lsv_at_layer_idx:
                x = self.lsv_matrix(x)
                # x = self.apply_learned_vector_to_layer_output(x)

            # Intercept for Steering Vectors
            if self.config.apply_vector_at_layer_idx is not None and layer == self.config.apply_vector_at_layer_idx:
                x = self.apply_vector_to_layer_output(x)
            if self.config.obtain_vector_at_layer_idx is not None and layer == self.config.obtain_vector_at_layer_idx:
                print(layer, self.config.obtain_vector_at_layer_idx)
                x = self.obtain_vector_from_layer_output(x)

            layer +=1

        x = self.transformer.ln_f(x)

        if self.n_embd_wte:
            x = F.linear(x, self.transformer.scale_down.weight.t())

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None
        gpt_count += 1

        return logits, loss

    def set_lsv_scaling_factor(self, factor):
        self.lsv_matrix.update_lsv_scaling_factor(factor)

    def set_lsv_mode(self, mode):
        self.lsv_matrix.set_mode(mode)

    def set_lsv_mixture(self, mixture):
        """ Mixture is a list, allowing for mixing steering vectors """
        self.lsv_matrix.set_mixture(mixture)

    def get_lsv_scaling_factor(self):
        return self.lsv_matrix.get_lsv_scaling_factor()

    def set_lsv_index(self, index):
        self.lsv_matrix.update_lsv_index(index)

    def freeze_non_lsv_parameters(self):
        """Freeze all parameters except for lsv_matrix if lsv_focused_training is enabled."""

        print("Freezing all parameters except for lsv_matrix")

        # Freeze all parameters by setting requires_grad to False
        for name, param in self.named_parameters():
            if name != "lsv_matrix":
                param.requires_grad = False
            else:
                param.requires_grad = True  # Ensure lsv_matrix can still be trained

    def apply_learned_vector_to_layer_output(self, x):
        """Conditionally add a vector based on dataset index to the output of a specific layer."""

        # Use one-hot vector for the dataset and multiply by the learned parameter matrix
        one_hot_vector = torch.zeros(self.lsv_matrix.size(0), device=x.device)
        one_hot_vector[self.lsv_index] = 1.0

        # Multiply the one-hot vector by the learned parameter matrix
        selected_vector = torch.matmul(one_hot_vector, self.lsv_matrix)

        x = x + selected_vector

        return x

    def apply_vector_to_layer_output(self, x):
        """Conditionally add a vector from a file to the output of a specific layer."""

        # require this method has the vector file
        assert self.config.apply_vector_file is not None

        vector = np.load(self.config.apply_vector_file)
        vector_tensor = torch.from_numpy(vector).float().to(x.device)
        x = x + self.config.apply_vector_scaling_factor * vector_tensor

        return x

    def obtain_vector_from_layer_output(self, x):
        """Append a vector to an existing .npy file."""

        # Convert the tensor back to a numpy array
        y = x
        y = torch.mean(y, dim=1, keepdim=True)
        result_vector = y.detach().cpu().numpy()

        # Save the vector to file
        np.save(self.config.obtain_vector_file, result_vector)
        print(f"Updated avg vector saved to {self.config.obtain_vector_file}")

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        if self.config.use_abs_pos_embeddings:
            self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, config, model_type):
        # assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel

        print(f"loading weights from pretrained gpt: {model_type}")

        # create a from-scratch initialized minGPT model
        model = GPT(config)
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)

        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        # NOTE: the assert below will fail because we split out the c_attn linears!
        # assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for key in sd_keys_hf:
            if any(key.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[key].shape[::-1] == sd[key].shape
                with torch.no_grad():
                    sd[key].copy_(sd_hf[key].t())
            elif key.endswith('attn.c_attn.weight') or key.endswith('attn.c_attn.bias'):
                # split into c_attn_q/k/v
                q, k, v  = sd_hf[key].t().split(config.n_embd, dim=0)
                q_key_str = key.replace("c_attn", "c_attn_q")
                k_key_str = key.replace("c_attn", "c_attn_k")
                v_key_str = key.replace("c_attn", "c_attn_v")
                sd[q_key_str] = q
                sd[k_key_str] = k
                sd[v_key_str] = v
            else:
                # vanilla copy over the other parameters
                print(key)
                if config.n_embd_wte:
                    if key == "transformer.wte.weight":
                        continue
                    if key == "lm_head.weight":
                        continue

                if not config.use_abs_pos_embeddings:
                    if key == "transformer.wpe.weight":
                        continue

                assert sd_hf[key].shape == sd[key].shape
                with torch.no_grad():
                    print(key)
                    sd[key].copy_(sd_hf[key])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        
        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        #flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        flops_promised = 71.2e12 # RTX3090 GPU float16 peak flops      ## GH
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = None
            if self.config.softmax_variant_output != 'softmax':
                probs = self.softmax_layer_output(logits)
            else:
                probs = F.softmax(logits, dim=-1)
            assert probs != None
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    @torch.no_grad()
    def generate_with_stop(self, idx, max_new_tokens, stop_string, decode, temperature=1.0, top_k=None):
        """
        Generate tokens and stop on fixed string match, return the state for further input.
        """
        generated_text = ""
        buffer = ""
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

            next_token_text = decode(idx_next[0].tolist())
            generated_text += next_token_text
            buffer += next_token_text

            # Check if the buffer ends with the stop_string
            if buffer.endswith(stop_string):
                break

        return idx, generated_text


class MoELayer(nn.Module):
    """ Mixture of Experts layer to replace FFN (or every other FFN) """

    def __init__(self, config):
        super().__init__()
        self.top_k = config.moe_top_k
        # TODO: implement expert capacity throttling
        # self.expert_capacity = config.expert_capacity
        self.num_experts = config.n_experts
        self.router = router_dictionary[config.moe_router_scheme](config)
        self.experts = nn.ModuleList([MLP(config) for _ in range(config.n_experts)])

    def forward(self, x):
        # Assuming x has shape [batch_size, seq_len, n_embd]
        batch_size, seq_len, _ = x.shape
        gating_output, indices = self.router(x)
        # print(f"gating_output.shape: {gating_output.shape}")
        # print(f"indices 1 count: {indices}")
        final_output = torch.zeros_like(x)

        # Flatten the batch and sequence dimensions to treat each token independently
        flat_x = x.view(-1, x.size(-1))
        # print(f"x.shape() = {x.shape}")
        # print(f"flat_x = {flat_x.shape}")
        flat_gating_output = gating_output.view(-1, gating_output.size(-1))
        # print(f"flat_gating_output.shape = {flat_gating_output.shape}")

        # Process each expert in parallel
        for i, expert in enumerate(self.experts):
            # Create a mask for the inputs where the current expert is in top-k
            expert_mask = (indices == i).any(dim=-1)
            flat_mask = expert_mask.view(-1)
            # print(f"expert_mask shape = {expert_mask.shape}")
            # print(f"flat_mask shape = {flat_mask.shape}")

            if flat_mask.any():
                expert_input = flat_x[flat_mask]
                expert_output = expert(expert_input)

                # Extract and apply gating scores
                gating_scores = flat_gating_output[flat_mask, i].unsqueeze(1)
                weighted_output = expert_output * gating_scores

                # Update final output additively by indexing and adding
                final_output[expert_mask] += weighted_output.squeeze(1)
        # print(f"final_output.shape = {final_output.shape}\n")
        return final_output
