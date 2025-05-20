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
    


def symmetric_quantize(tensor, bits, causal_mask=False):
    """
    Symmetric quantization function
    :param tensor: Tensor to be quantized
    :param bits: Number of bits of quantization
    :return: zero point, scale, quantized tensor
    """
    bit_max = (1 << (bits - 1)) - 1
    bit_min = -bit_max - 1
    if causal_mask:
        # Apply torch.tril to get the lower triangular part (including diagonal)
        lower_triangular = torch.tril(tensor)

        # Find the maximum value
        abs_max = lower_triangular.abs().max()
    else:
        abs_max = tensor.abs().max()
    scale = abs_max / bit_max
    xi_array = torch.round(tensor / scale)
    clamped_array = torch.clamp(xi_array, min=bit_min, max=bit_max).to(dtype=set_dtype(bits))
    return torch.tensor([0], device=tensor.device), scale, clamped_array

def adaptive_clamp(x, min_val, max_val, alpha=0.1):
    """
    Adaptive Clamping 적용
    - Outlier를 너무 강하게 제거하지 않고 부드럽게 조정
    - min_val보다 작은 값은 min_val로, max_val보다 큰 값은 max_val * alpha로 조정
    
    :param x: 입력 텐서
    :param min_val: 하한값 (Percentile Clamping에 의해 결정됨)
    :param max_val: 상한값 (Percentile Clamping에 의해 결정됨)
    :param alpha: Clamping 강도 (기본값 0.1 = Clamping을 조금만 적용)
    :return: Clamping이 적용된 텐서
    """
    return x * (x > min_val) * (x < max_val) + min_val * (x <= min_val) + max_val * (x >= max_val) * alpha

# def affine_quantize(tensor, bits, percentile=99.9, alpha=0.1, per_channel=True, causal_mask=False):
#     """
#     ✅ 개선된 Affine Quantization (Percentile Clamping + Adaptive Clamping + Per-Channel Quantization)
    
#     :param tensor: 입력 텐서
#     :param bits: 양자화 비트 수
#     :param percentile: Outlier 제거를 위한 Percentile Clamping 값 (기본값 99.9%)
#     :param alpha: Adaptive Clamping 강도 (기본값 0.1)
#     :param per_channel: 채널별 양자화 적용 여부 (True이면 각 채널별 독립적인 scale 적용)
#     :param causal_mask: Causal Mask 적용 여부
#     :return: (zero_point, scale, 양자화된 텐서)
#     """
#     bit_max = (1 << (bits - 1)) - 1  # 예: INT4 -> 7
#     bit_min = -bit_max - 1  # 예: INT4 -> -8

#     if causal_mask:
#         tensor = torch.tril(tensor)  # Lower triangular part 적용

#     # ✅ 입력 텐서를 float으로 변환 (중요!)
#     tensor = tensor.to(torch.float32)

#     # ✅ 채널별 (Per-Channel) 또는 전체 (Per-Tensor) 단위로 Percentile Clamping 적용
#     if per_channel:
#         min_val = torch.quantile(tensor, (100 - percentile) / 100, dim=0, keepdim=True)
#         max_val = torch.quantile(tensor, percentile / 100, dim=0, keepdim=True)
#     else:
#         min_val = torch.quantile(tensor, (100 - percentile) / 100)
#         max_val = torch.quantile(tensor, percentile / 100)

#     # ✅ Adaptive Clamping 적용
#     tensor = adaptive_clamp(tensor, min_val, max_val, alpha)

#     # ✅ Scale & Zero-Point 계산
#     scale = (max_val - min_val) / ((1 << bits) - 1)
#     zero_point = torch.round(-min_val / scale).clamp(bit_min, bit_max)

#     # ✅ Quantization 수행
#     xi_array = torch.round(tensor / scale + zero_point)
#     quantized_tensor = torch.clamp(xi_array, min=bit_min, max=bit_max).to(dtype=set_dtype(bits))

#     return zero_point, scale, quantized_tensor


def affine_quantize(tensor, bits, causal_mask=False):           
    bit_max = (1 << (bits - 1)) - 1
    bit_min = -bit_max - 1

    max_val = tensor.max()
    min_val = tensor.min()
    scale = (max_val - min_val) / ((1 << bits) - 1)
    zero_point = torch.round(-min_val / scale)  # 수정된 부분
    zero_point = zero_point.clamp(bit_min, bit_max)  # 범위 제한 추가
    xi_array = torch.round(tensor / scale + zero_point)
    return zero_point, scale, torch.clamp(xi_array, min=bit_min, max=bit_max).to(dtype=set_dtype(bits))

# def affine_quantize(tensor, bits):
#     """
#     Affine (asymmetric) quantization function
#     :param tensor: Tensor to be quantized
#     :param bits: Number of bits of quantization
#     :return: zero point, scale, quantized tensor
#     """
#     bit_max = (1 << (bits - 1)) - 1
#     bit_min = -bit_max - 1
#     max = tensor.max()
#     min = tensor.min()
#     scale = (max - min) / ((1 << bits) - 1)
#     zero_point = -torch.round(min / scale) + bit_min
#     xi_array = torch.round(tensor / scale) + zero_point
#     return zero_point, scale, torch.clamp(xi_array, min=bit_min, max=bit_max).to(dtype=set_dtype(bits))



def stochastic_quantize(tensor, bits, causal_mask=False):       ## GH causal mask added
    if causal_mask:
        tensor = torch.tril(tensor)  # Lower triangular part
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



def fake_quantize_act(obj, activation, tensor, num_bits, quant_method, iter_num, causal_mask=False):
    zero_point, scale, act = quantize_dictionary[quant_method](tensor, num_bits, causal_mask=causal_mask)
    setattr(obj, activation, act)
    setattr(obj, f"{activation}_scale", scale)
    setattr(obj, f"{activation}_zero_point", zero_point)
    dequantized = dequantize(zero_point, scale, act, causal_mask=causal_mask)
    # 먼저 causal mask 적용
    if causal_mask:
        upper_tri_mask = torch.triu(torch.ones_like(tensor), diagonal=1).bool()
        tensor.masked_fill_(upper_tri_mask, -float('inf'))  # 선처리
        # Set the upper triangular part to -inf
        tensor[upper_tri_mask] = 0
    # If scheduler is set, then we need to calculate the current quantization level
    if obj.quant_scheduler != None:
        quant_level = calculate_quant_level(obj.training, obj.quant_scheduler, obj.start_quant_level, obj.full_quant_iteration, iter_num)
        # print quantization level for every evaluation interval
        if obj.training and iter_num % obj.eval_interval == 0:
            print("quant level: ", quant_level)
        # adds quantization error to the original tensor
        result = tensor + quant_level * (dequantized - tensor).detach()
    else:
        result = dequantized
    if causal_mask:
        result[upper_tri_mask] = -float('inf')

    return result



class FakeLinearQuantizationFunction(torch.autograd.Function):
    """Simulates error caused by quantization. Uses Straight-Through Estimator for Back prop
    Source: https://github.com/Alexstrasza98/Transformer-Quantization/blob/main
    Source License: MIT
    """
    @staticmethod
    def forward(ctx, input, training, quant_scheduler, start_quant_level, full_quant_iter, eval_interval, steps, bits=7, quantization_method="affine_quant"):
        """
        Forward pass
        :param ctx: Context object to store information for the backward pass (not used in this case)
        :param input: The input tensor to be quantized
        :param bits: The number of bits for quantization (default is 7)
        :return: Dequantized tensor
        """
        # steps:
        # Quantize the input tensor using the quantize function.
        # Dequantize the quantized values using the dequantize function.
        # Return the dequantized tensor, which approximates the input tensor but includes the quantization error.
        zero_point, norm, quantized_weight = quantize_dictionary[quantization_method](input, bits)
        # If scheduler is set, then we need to calculate the current quantization level
        dequantized = dequantize(zero_point, norm, quantized_weight)
        if quant_scheduler != None:
            quant_level = calculate_quant_level(training, quant_scheduler, start_quant_level, full_quant_iter, steps)
            if training and steps % eval_interval == 0:
                print("quant level: ", quant_level)
            
            return input + quant_level * (dequantized - input).detach()
        return dequantized
    @staticmethod
    def backward(ctx, grad_output):
        # Straight-Through Estimator (STE): passes grad_output through as the gradient with respect to the input
        # gradient is approximated by simply passing the gradient from the output directly to the input, 
        # ignoring the quantization operation
        return grad_output, None, None, None, None, None, None, None, None



quantize_dictionary = {
    "ternary_quant": ternary_quantize,
    "symmetric_quant": symmetric_quantize,
    "affine_quant": affine_quantize,
    "stochastic_quant": stochastic_quantize,
}



_fake_quantize = FakeLinearQuantizationFunction.apply
