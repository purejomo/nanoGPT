## omniquant methods for symmetric quantization
import torch

def set_dtype(bits):
    if bits > 16:
        return torch.int32
    if bits > 8:
        return torch.int16
    else:
        return torch.int8
    
def ternary_quantize(tensor, bits, causal_mask=False):
    if causal_mask:
        lower_triangular = torch.tril(tensor)
        scale = lower_triangular.abs().mean().clamp(min=1e-5)
    else:
        scale = tensor.abs().mean().clamp(min=1e-5)
    result = (tensor / scale).round().clamp(-1, 1).to(dtype=torch.int8)
    return torch.tensor([0], device=tensor.device), scale, result
    
def calculate_quant_level(training, quant_scheduler, start_quant_level, full_quant_iter, iter_num):
    if full_quant_iter == None:
        raise ValueError("Full quant iteration was not specified.")
    if iter_num == None:
        raise ValueError("Iter_num was not passed to GPT model")
    if not training:
        return 1
    if quant_scheduler == "static":
        return start_quant_level
    elif quant_scheduler == "linear":
        return min(iter_num / full_quant_iter + (full_quant_iter * start_quant_level), 1)
    
# def symmetric_quantize(tensor, bits, causal_mask=False):
#     """
#     Symmetric quantization function
#     :param tensor: Tensor to be quantized
#     :param bits: Number of bits of quantization
#     :return: zero point, scale, quantized tensor
#     """
#     bit_max = (1 << (bits - 1)) - 1
#     bit_min = -bit_max - 1
#     if causal_mask:
#         # Apply torch.tril to get the lower triangular part (including diagonal)
#         lower_triangular = torch.tril(tensor)

#         # Find the maximum value
#         abs_max = lower_triangular.abs().max()
#     else:
#         abs_max = tensor.abs().max()
#     scale = abs_max / bit_max
#     xi_array = torch.round(tensor / scale)
#     clamped_array = torch.clamp(xi_array, min=bit_min, max=bit_max).to(dtype=set_dtype(bits))
#     return torch.tensor([0], device=tensor.device), scale, clamped_array

## classes for omniquant

class LearnableEquivalentTransformation(torch.nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        # ✅ scale_factor 초기값을 낮춰서 안정적인 학습 유도
        self.scale_factor = torch.nn.Parameter(torch.ones(num_channels) * 0.1)  
        self.shift_factor = torch.nn.Parameter(torch.zeros(num_channels))  

    def forward(self, activation):
        transformed = (activation - self.shift_factor) / (self.scale_factor + 1e-6)  
        return transformed


class LearnableWeightClipping(torch.nn.Module):
    def __init__(self, init_gamma=1.0, init_beta=1.0):
        super().__init__()
        self.gamma = torch.nn.Parameter(torch.tensor(init_gamma))  
        self.beta = torch.nn.Parameter(torch.tensor(init_beta))  

    def forward(self, tensor, bits):
        bit_max = (1 << (bits - 1)) - 1
        bit_min = -bit_max - 1

        max_val = tensor.max()
        min_val = tensor.min()

        # ✅ NaN 체크
        if torch.isnan(max_val) or torch.isnan(min_val):
            print("❌ [ERROR] max_val or min_val has NaN!")

        # ✅ 안정적인 학습을 위해 gamma, beta 값 제한
        gamma = torch.clamp(self.gamma, min=0.5, max=2.0)
        beta = torch.clamp(self.beta, min=0.5, max=2.0)

        clipped_max = max_val * gamma  
        clipped_min = min_val * beta  

        # ✅ scale 값이 0이 되는 걸 방지
        scale = max(clipped_max - clipped_min, 1e-2) / bit_max  

        # ✅ scale NaN 체크
        if torch.isnan(scale).any():
            print("❌ [ERROR] scale has NaN values!")

        quantized = torch.round((tensor - clipped_min) / scale).clamp(bit_min, bit_max).to(dtype=set_dtype(bits))

        # ✅ NaN 체크
        if torch.isnan(quantized).any():
            print("❌ [ERROR] quantized tensor has NaN!")

        return torch.tensor([0], device=tensor.device), scale, quantized


def symmetric_quantize(tensor, bits, causal_mask=False):
    """
    Symmetric quantization function with NaN debugging
    """
    bit_max = (1 << (bits - 1)) - 1
    bit_min = -bit_max - 1

    if causal_mask:
        lower_triangular = torch.tril(tensor)
        abs_max = lower_triangular.abs().max()
    else:
        abs_max = tensor.abs().max()

    # ✅ NaN 체크
    if torch.isnan(tensor).any():
        print("❌ [ERROR] Input tensor has NaN values before quantization!")

    if torch.isnan(abs_max):
        print("❌ [ERROR] abs_max has NaN values! Check tensor values before this step.")
        print("tensor:", tensor)

    # ✅ scale 값이 0이 되는 걸 방지
    scale = max(abs_max, 1e-6) / bit_max  # NaN 방지

    xi_array = torch.round(tensor / scale)
    clamped_array = torch.clamp(xi_array, min=bit_min, max=bit_max).to(dtype=set_dtype(bits))

    # ✅ NaN 체크
    if torch.isnan(clamped_array).any():
        print("❌ [ERROR] clamped_array has NaN values after quantization!")

    return torch.tensor([0], device=tensor.device), scale, clamped_array




def affine_quantize(tensor, bits, causal_mask=False):
    """
    Affine (asymmetric) quantization function with optional causal masking
    :param tensor: Tensor to be quantized
    :param bits: Number of bits of quantization
    :param causal_mask: Whether to apply a causal mask (default: False)
    :return: zero point, scale, quantized tensor
    """
    bit_max = (1 << (bits - 1)) - 1
    bit_min = -bit_max - 1

    if causal_mask:
        tensor = torch.tril(tensor)  # ✅ 하삼각행렬만 남기기 for error debugging

    max_val = tensor.max()
    min_val = tensor.min()
    scale = (max_val - min_val + 1e-6) / ((1 << bits) - 1)  # ✅ 스케일 안정화
    zero_point = (-torch.round(min_val / scale)).clamp(bit_min, bit_max)
    xi_array = torch.round(tensor / scale + zero_point)

    return zero_point, scale, torch.clamp(xi_array, min=bit_min, max=bit_max).to(dtype=set_dtype(bits))


def stochastic_quantize(tensor, bits):
    """
    Stochastic quantization function
    :param tensor: Tensor to be quantized
    :param bits: Number of bits of quantization
    :return: zero point, scale, quantized tensor
    Source: https://github.com/Alexstrasza98/Transformer-Quantization/blob/main
    Source License: MIT
    """
    s = (1 << bits) - 1

    norm = tensor.abs().max()
    sign_array = torch.sign(tensor).to(dtype=torch.int8)
    l_array = torch.abs(tensor) / norm * s
    l_array_floored = l_array.to(dtype=torch.int)

    prob_array = l_array - l_array_floored
    prob_array = torch.clamp(prob_array, min=0.0, max=1.0)
    mask = torch.bernoulli(prob_array)
    xi_array = l_array_floored + mask
    xi_array = xi_array.to(dtype=torch.int32)
    sign_xi_array = (sign_array * xi_array).to(dtype=set_dtype(bits))
    norm = norm / s

    return torch.tensor([0], device=tensor.device), norm, sign_xi_array

def dequantize(zero_point, scale, tensor, causal_mask=False):
    """
    Dequantize the quantizated tensor
    :param zero_point: zero point of tensor
    :param scale: scale of tensor
    :param tensor: quantized tensor
    :return: Dequantized weights
    """
    dequantized = (tensor - zero_point) * scale
    return dequantized

# def fake_quantize_act(obj, activation, tensor, num_bits, quant_method, iter_num, causal_mask=False):
#     zero_point, scale, act = quantize_dictionary[quant_method](tensor, num_bits, causal_mask=causal_mask)
#     setattr(obj, activation, act)
#     setattr(obj, f"{activation}_scale", scale)
#     setattr(obj, f"{activation}_zero_point", zero_point)
#     dequantized = dequantize(zero_point, scale, act, causal_mask=causal_mask)
#     if causal_mask:
#         # Create a mask for the upper triangular part
#         upper_tri_mask = torch.triu(torch.ones_like(tensor), diagonal=1).bool()

#         # Set the upper triangular part to -inf
#         tensor[upper_tri_mask] = 0

#     # If scheduler is set, then we need to calculate the current quantization level
#     if obj.quant_scheduler != None:
#         quant_level = calculate_quant_level(obj.training, obj.quant_scheduler, obj.start_quant_level, obj.full_quant_iteration, iter_num)
#         # print quantization level for every evaluation interval
#         if obj.training and iter_num % obj.eval_interval == 0:
#             print("quant level: ", quant_level)
#         # adds quantization error to the original tensor
#         result = tensor + quant_level * (dequantized - tensor).detach()
#     else:
#         result = dequantized

#     if causal_mask:
#         result[upper_tri_mask] = -float('inf')

#     return result

def fake_quantize_act(obj, activation, tensor, num_bits, quant_method, iter_num, causal_mask=False):
    # Learnable Equivalent Transformation 적용
    let = LearnableEquivalentTransformation(tensor.shape[-1]).to(tensor.device)
    transformed_tensor = let(tensor)  # activation 변환 수행

    # 변환된 activation을 기반으로 양자화 수행
    zero_point, scale, act = quantize_dictionary[quant_method](transformed_tensor, num_bits, causal_mask=causal_mask)

    setattr(obj, activation, act)
    setattr(obj, f"{activation}_scale", scale)
    setattr(obj, f"{activation}_zero_point", zero_point)

    dequantized = dequantize(zero_point, scale, act, causal_mask=causal_mask)

    return dequantized


class FakeLinearQuantizationFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, training, quant_scheduler, start_quant_level, full_quant_iter, eval_interval, steps, bits=7, quantization_method="symmetric_quant"):
        # ✅ 입력 데이터 NaN 체크
        if torch.isnan(input).any():
            print("❌ [ERROR] Input tensor has NaN!")

        # LET 적용: activation 변환
        let = LearnableEquivalentTransformation(input.shape[-1]).to(input.device)
        transformed_input = let(input)

        # ✅ NaN 체크
        if torch.isnan(transformed_input).any():
            print("❌ [ERROR] Transformed input has NaN!")

        # LWC 적용하여 학습 가능한 weight clipping 수행
        lwc = LearnableWeightClipping().to(input.device)
        zero_point, scale, quantized_weight = lwc(transformed_input, bits)

        # ✅ NaN 체크
        if torch.isnan(quantized_weight).any():
            print("❌ [ERROR] Quantized weight has NaN!")

        # Dequantization 수행
        dequantized = dequantize(zero_point, scale, quantized_weight)

        # ✅ NaN 체크
        if torch.isnan(dequantized).any():
            print("❌ [ERROR] Dequantized tensor has NaN!")

        # QAT 방식 유지
        if quant_scheduler is not None:
            quant_level = calculate_quant_level(training, quant_scheduler, start_quant_level, full_quant_iter, steps)
            if training and steps % eval_interval == 0:
                print("quant level: ", quant_level)
            return input + quant_level * (dequantized - input).detach()

        return dequantized




quantize_dictionary = {
    "ternary_quant": ternary_quantize,
    "symmetric_quant": symmetric_quantize,
    "affine_quant": affine_quantize,
    "stochastic_quant": stochastic_quantize
}

_fake_quantize = FakeLinearQuantizationFunction.apply