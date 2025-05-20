import torch
import torch.nn as nn
import math
import time
import os
import math
from typing import Tuple, Optional
import torch.nn.functional as F

# Softmax base 2, with option to remove max subtraction
class Softermax(nn.Module):
    """ Base-2 Softmax with option to remove max subtraction"""
    def __init__(self, config, dim=-1):
        super().__init__()
        self.dim = dim
        self.subtract_max = config.softermax_use_xmax

    def forward(self, x):
        torch.cuda.nvtx.range_push("Softermax")
        softmaxPV_start_time = time.perf_counter()
        if self.subtract_max:
            max_x = x.max(dim=self.dim, keepdim=True).values
            x = x - max_x
        e_x = torch.pow(2.0, x)
        result = e_x / e_x.sum(dim=self.dim, keepdim=True)
        softmaxPV_end_time = time.perf_counter()
        print(f"softmaPV time: {(softmaxPV_end_time - softmaxPV_start_time) * 1000000}")
        torch.cuda.nvtx.range_pop()

        return result


# Softmax variation with learnable constant parameters for xmax and denominator
class ConSmax(nn.Module):
    """ Constant learnable parameters for xmax and denominator """
    def __init__(self, config, dim=-1):
        super().__init__()
        self.dim = dim
        self.softmax_io_logging = config.softmax_io_logging
        self.softmax_io_log_interval = config.softmax_io_log_interval
        self.iter_num = 0

        if self.softmax_io_logging:
            self.inputs = []
            self.outputs = []

        # learnable 'xmax' - beta
        self.beta = nn.Parameter(torch.Tensor([config.consmax_initial_beta]))

        # denominator - gamma
        self.gamma = nn.Parameter(torch.Tensor([config.consmax_initial_gamma]))

        # Set the base of the exponent
        if config.consmax_use_euler_base:
            self.consmax_base = math.e
        else:
            self.consmax_base = config.consmax_base

    def forward(self, x):
        x_adj = x - self.beta
        e_x = torch.pow(self.consmax_base, x_adj)
        result = e_x / self.gamma

        if self.training and self.softmax_io_logging and self.iter_num % self.softmax_io_log_interval == 0:
            self.inputs = x
            self.outputs = result

        if self.training:
            self.iter_num += 1
        return result


# Softmax variation with per-head learnable constant parameters for xmax and denominator
class ConSmaxV2(nn.Module):
    def __init__(self, config, dim=-1):
        super().__init__()
        self.n_head = config.n_head
        self.softmax_io_logging = config.softmax_io_logging
        self.softmax_io_log_interval = config.softmax_io_log_interval
        self.iter_num = 0

        if self.softmax_io_logging:
            self.inputs = []
            self.outputs = []

        self.beta_init = config.consmax_initial_beta
        self.gamma_init = config.consmax_initial_gamma
        self.beta_factor = nn.Parameter(torch.ones(self.n_head, 1, 1))
        self.gamma_factor = nn.Parameter(torch.ones(self.n_head, 1, 1))

        # Set beta and gamma as fields for backwards compatibility
        self.beta = self.beta_init * self.beta_factor
        self.gamma = self.beta_init * self.gamma_factor

        # Set optional clamping (on by default)
        self.clamp_inputs = config.consmax_v2_clamping
        self.clamp_value = config.consmax_v2_clamp_value

        # Set the base of the exponent
        if config.consmax_use_euler_base:
            self.consmax_base = math.e
        else:
            self.consmax_base = config.consmax_base

    def forward(self, x):
        torch.cuda.nvtx.range_push("ConSmax")
        

        self.beta = self.beta_factor * self.beta_init
        self.gamma = self.gamma_factor * self.gamma_init
        x_adj = x - self.beta
        if self.clamp_inputs:
            x_adj[x_adj > self.clamp_value] = self.clamp_value
        # e_x = torch.pow(self.consmax_base, x_adj)
        softmaxPV_start_time = time.perf_counter()
        e_x = torch.exp(x_adj * math.log(self.consmax_base))
        result = e_x / self.gamma

        softmaxPV_end_time = time.perf_counter()
        print(f"softmaPV time: {(softmaxPV_end_time - softmaxPV_start_time) * 1000000}")

        if self.training and self.softmax_io_logging and self.iter_num % self.softmax_io_log_interval == 0:
            self.inputs = x
            self.outputs = result   

        if self.training:
            self.iter_num += 1
        torch.cuda.nvtx.range_pop()

        return result


# Softmax variation with per-head learnable constant parameters for xmax and denominator
class ConSmaxV3(nn.Module):
    def __init__(self, config, dim=-1):
        super().__init__()
        self.n_head = config.n_head
        self.softmax_io_logging = config.softmax_io_logging
        self.softmax_io_log_interval = config.softmax_io_log_interval
        self.iter_num = 0

        if self.softmax_io_logging:
            self.inputs = []
            self.outputs = []

        self.beta_init = config.consmax_initial_beta
        self.gamma_init = config.consmax_initial_gamma
        self.beta_factor = nn.Parameter(torch.ones(self.n_head, 256, 1))
        self.gamma_factor = nn.Parameter(torch.ones(self.n_head, 256, 1))

        # Set beta and gamma as fields for backwards compatibility
        self.beta = self.beta_init * self.beta_factor
        self.gamma = self.beta_init * self.gamma_factor

        # Set optional clamping (on by default)
        self.clamp_inputs = config.consmax_v2_clamping
        self.clamp_value = config.consmax_v2_clamp_value

        # Set the base of the exponent
        if config.consmax_use_euler_base:
            self.consmax_base = math.e
        else:
            self.consmax_base = config.consmax_base

    def forward(self, x):
        self.beta = self.beta_factor * self.beta_init
        self.gamma = self.gamma_factor * self.gamma_init

        x_adj = x - self.beta
        if self.clamp_inputs:
            x_adj[x_adj > self.clamp_value] = self.clamp_value

        e_x = torch.pow(self.consmax_base, x_adj)

        result = e_x / self.gamma

        if self.training and self.softmax_io_logging and self.iter_num % self.softmax_io_log_interval == 0:
            self.inputs = x
            self.outputs = result

        if self.training:
            self.iter_num += 1

        return result

# Constantmax Quantized
## Quantization Methods Utilized for Separate Forward and Backward Passes
def quantize(tensor,scale):
    tensor = tensor.mul(scale)
    tensor = torch.round(tensor)
    return tensor
def dequantize(tensor,scale):
    tensor = tensor.div(scale)
    return tensor

## helper class for Constantmax_quan
class const_quan(torch.autograd.Function):
    """Simulates error caused by quantization. Uses Straight-Through Estimator for Back prop"""
    @staticmethod
    def forward(ctx, beta=None, gamma=None):
        #scaling factor for beta and gamma while doing quantization
        scale_beta=100 #scaling factor for quantization, should make it as parameter
        scale_gamma=10
        beta = quantize(beta, scale_beta)
        gamma = quantize(gamma, scale_gamma)
        return dequantize(beta, scale_beta),dequantize(gamma,scale_gamma)

    @staticmethod
    def backward(ctx, grad_gamma, grad_beta):
        return grad_gamma, grad_beta

_const_quan=const_quan.apply

# Softmax variation with quantized xmax and denominator
class ConSmaxQuan(nn.Module):
    """ Quantized version with learnable beta and gamma """
    def __init__(self, config, dim=-1):
        super().__init__()
        self.dim = dim
        self.softmax_io_logging = config.softmax_io_logging
        self.softmax_io_log_interval = config.softmax_io_log_interval
        self.iter_num = 0

        self.gamma = nn.Parameter(torch.Tensor([config.consmax_initial_gamma]))
        self.beta = nn.Parameter(torch.Tensor([config.consmax_initial_beta]))

        self.fake_beta = None
        self.fake_gamma = None

    def forward(self, x):
        if self.training:
            self.fake_beta, self.fake_gamma = _const_quan(self.beta, self.gamma)
            x = x - self.fake_beta
            e_x = torch.exp(x)
            result = e_x / self.fake_gamma
        else:
            scale_beta = 100
            scale_gamma = 10
            x = x - dequantize(quantize(self.beta, scale_beta), scale_beta)
            e_x = torch.exp(x)
            result = e_x / dequantize(quantize(self.gamma, scale_gamma), scale_gamma)

        if self.training and self.softmax_io_logging and self.iter_num % self.softmax_io_log_interval == 0:
            self.inputs = x
            self.outputs = result

        if self.training:
            self.iter_num += 1

        return result


# Like softmax, but parameterized to permit exploration
class Strongermax(nn.Module):
    """ Softmax with ability to increase to 'stronger' bases """
    def __init__(self, config, dim=-1):
        super().__init__()
        self.dim = dim

        # Strongermax Params
        self.strength = config.strongermax_strength
        self.subtract_max = config.strongermax_use_xmax
        self.xmax_guess = config.strongermax_xmax_guess
        self.sum_to_1 = config.strongermax_sum_to_1
        self.divisor = config.strongermax_divisor
        self.div_by_seq_len = config.div_by_seq_len
        self.overflow_recompute = config.strongermax_overflow_recompute
        self.softmax_io_log_interval = config.softmax_io_log_interval
        self.iter_num = 0

        if self.overflow_recompute:
            assert self.xmax_guess is not None, "for overflow recompute, xmax_guess must be set"

        # Input and Output Logging
        self.softmax_io_logging = config.softmax_io_logging
        if self.softmax_io_logging:
            self.inputs = []
            self.outputs = []

    def forward(self, x):
        x_adj = None

        if self.subtract_max:
            # Guessing correctly instead of subtracting real max can save a pass
            # else we use real xmax
            max_x = x.max(dim=self.dim, keepdim=True).values
            if self.overflow_recompute:
                if (torch.max(x - self.xmax_guess)) > 88:
                    x_adj = x - max_x
                else:
                    x_adj = x - self.xmax_guess
            else:
                if self.xmax_guess:
                    x_adj = x - self.xmax_guess
                else:
                    x_adj = x - max_x
        else:
            x_adj = x

        result = torch.pow(self.strength, x_adj)

        if self.sum_to_1:
            result = result / result.sum(dim=self.dim, keepdim=True)
        if self.div_by_seq_len:
            seq_len = x.shape[self.dim]
            result = result / seq_len

        result = result / self.divisor

        if self.training and self.softmax_io_logging and self.iter_num % self.softmax_io_log_interval == 0:
            self.inputs = x
            self.outputs = result

        if self.training:
            self.iter_num += 1

        return result

# Using polynomial instead of exponential for Softmax separation non-linearity
class Polymax(nn.Module):
    def __init__(self, config, dim=-1):
        super().__init__()

        assert(config.polymax_x_intercept < 0) # ensure x_intercept is strictly left of the y-axis

        self.dim = dim

        self.div_by_seq_len = config.div_by_seq_len

        self.x_intercept = config.polymax_x_intercept # where to transition from y=0 to m*x+b
        self.y_intercept = config.polymax_y_intercept # where the graph crosses y-axis
        self.linear_slope = (self.y_intercept - 0)/(0 - self.x_intercept) # aka 'slope', also x intercept !=0

        self.power = config.polymax_power
        self.divisor = config.polymax_divisor
        self.div_by_seq_len = config.div_by_seq_len
        self.softmax_io_logging = config.softmax_io_logging
        self.softmax_io_log_interval = config.softmax_io_log_interval
        self.iter_num = 0

        if self.softmax_io_logging:
            self.inputs = []
            self.outputs = []

    def forward(self, x):
        # Overview:
        # Flat section:       -inf < x < x_intercept
        # Linear section:     x_intercept <= x <= 0
        # Polynomial section: 0 < x < inf
        # Flat section
        flat_piece = torch.where(x < self.x_intercept, torch.tensor(0.0, device=x.device), torch.tensor(0.0, device=x.device))

        # Linear section
        linear_piece = torch.where((x >= self.x_intercept) & (x <= 0), self.linear_slope * x + self.y_intercept, torch.tensor(0.0, device=x.device))

        # Polynomial section
        poly_piece = torch.where(x > 0, x**self.power + self.y_intercept, torch.tensor(0.0, device=x.device))

        # Combine sections
        result = (poly_piece + linear_piece + flat_piece)/self.divisor

        # Divide by sequence length
        if self.div_by_seq_len:
            seq_len = x.shape[self.dim]
            result = result / seq_len

        if self.training and self.softmax_io_logging and self.iter_num % self.softmax_io_log_interval == 0:
            self.inputs = x
            self.outputs = result

        if self.training:
            self.iter_num += 1

        return result

class VPolymax(nn.Module):
    """ variation of polymax with a v-shape, and is non-monotonically increasing"""
    def __init__(self, config, dim=-1):
        super().__init__()

        assert(config.polymax_x_intercept < 0) # ensure x_intercept is strictly left of the y-axis
        self.dim = dim
        self.div_by_seq_len = config.div_by_seq_len

        self.x_intercept = config.polymax_x_intercept # where to transition from y=0 to m*x+b
        self.y_intercept = config.polymax_y_intercept # where the graph crosses y-axis
        self.linear_slope = (self.y_intercept - 0)/(self.x_intercept - 0) # vpoly uses reverse slope

        self.power = config.polymax_power
        self.divisor = config.polymax_divisor

    def forward(self, x):
        # Overview:
        # Flat section:       -inf < x < x_intercept
        # Linear section:     x_intercept <= x <= 0
        # Polynomial section: 0 < x < inf

        # Flat section
        flat_piece = torch.where(x < self.x_intercept, torch.tensor(0.0, device=x.device), torch.tensor(0.0, device=x.device))

        # Linear section
        linear_piece = torch.where((x >= self.x_intercept) & (x <= 0), self.linear_slope * x + self.y_intercept, torch.tensor(0.0, device=x.device))

        # Polynomial section
        poly_piece = torch.where(x > 0, x**self.power + self.y_intercept, torch.tensor(0.0, device=x.device))

        # Combine sections
        result = (poly_piece + linear_piece + flat_piece)/self.divisor

        # divide by sequence length
        if self.div_by_seq_len:
            seq_len = x.shape[self.dim]
            result = result / seq_len

        return result

# Merging of ConSmax body for gradient prop and Polymax head for numerical stability
class SaturatingConSmax(nn.Module):
    def __init__(self, config, dim=-1):
        super().__init__()

        self.dim = dim

        if config.consmax_learnable_beta:
            # learnable 'xmax' is beta
            self.beta = nn.Parameter(torch.Tensor([config.consmax_initial_beta]))
        else:
            self.beta = config.consmax_initial_beta

        if config.consmax_learnable_gamma:
            # denominator is gamma
            self.gamma = nn.Parameter(torch.Tensor([config.consmax_initial_gamma]))
        else:
            self.gamma = config.consmax_initial_gamma

        if config.consmax_use_euler_base:
            self.consmax_base = math.e
        else:
            self.consmax_base = config.consmax_base

        self.div_by_seq_len = config.div_by_seq_len

        # ConSmax saturation is like ReLU6 but happens where e^x normally would overflow
        # Since we're subtracting x by beta, we only need to guard at "beta + x_sat_value)
        # Note: for e^x this is around 11 for fp16 precision
        self.x_sat = config.consmax_saturation + config.consmax_initial_beta

    def forward(self, x):
        # Overview:
        # exponential section:    -inf < x < (sat_point)
        # flat section:           (sat_point) <= x < inf

        # Exponential section
        exponential_piece = torch.where(
            (x < (self.x_sat)),
            torch.pow(self.consmax_base, x - self.beta),
            torch.tensor(0.0, device=x.device))

        # flat section
        flat_piece = torch.where(x >= (self.x_sat), torch.tensor(self.x_sat, device=x.device), torch.tensor(0.0, device=x.device))

        # Combine sections
        result = (exponential_piece + flat_piece)/self.gamma

        # divide by sequence length
        if self.div_by_seq_len:
            seq_len = x.shape[self.dim]
            result = result / seq_len

        return result

# Merging of ConSmax body for gradient prop and Polymax head for numerical stability
class ExpPolymax(nn.Module):
    def __init__(self, config, dim=-1):
        super().__init__()

        self.dim = dim

        self.div_by_seq_len = config.div_by_seq_len

        # Base selection
        if config.exppolymax_use_euler_base:
            self.exppolymax_base = math.e
        else:
            self.exppolymax_base = config.exppolymax_base

        self.y_intercept = config.exppolymax_y_intercept # where the graph crosses y-axis
        self.power = config.exppolymax_power
        self.divisor = config.exppolymax_divisor
        # Assumes Euler Base:
        # Shift of x to move poly portion forward to obtain continuous derivative at x=0
        # derivative of poly at 0 should equal a^0
        # d(x^n + y-int) = d(a^x|x=0) = ln(a) * a^0 = ln(a)
        # n * x^(n-1) = ln(a)
        # x = (ln(a) * ( 1 / n )) ** (1/(n-1))
        # Note: if n==1 (straight line) match is already attained, and calculation would nan, so test this case first
        if config.exppolymax_power == 1.0:
            # Note: this only works with y=x and e^x, since we'd have to implement a multiplier or shift teh exponent otherwise.
            self.x_derivative_match_shift = 0
        elif config.exppolymax_use_euler_base:
            # ln(e) = 1
            self.x_derivative_match_shift = (1.0 / config.exppolymax_power)**(1/(config.exppolymax_power - 1))
        else:
            # ln(a) must be calculated, note torch.log is the natural log 'ln'
            self.x_derivative_match_shift = (torch.log(config.exppolymax_base) * (1.0 / config.exppolymax_power))**(1/(config.exppolymax_power - 1))

    def forward(self, x):
        # Overview:
        # exponential section:    -inf < x < 0
        # Polynomial section:     0 < x < inf

        # Exponential section
        exponential_piece = torch.where((x < 0), torch.pow(self.exppolymax_base, x), torch.tensor(0.0, device=x.device))

        # Polynomial section
        poly_piece = torch.where(x >= 0, (x + self.x_derivative_match_shift)**self.power + self.y_intercept, torch.tensor(0.0, device=x.device))

        # Combine sections
        result = (poly_piece + exponential_piece)/self.divisor

        # divide by sequence length
        if self.div_by_seq_len:
            seq_len = x.shape[self.dim]
            result = result / seq_len

        return result


# SigSoftmax from https://arxiv.org/abs/1805.10829
class SigSoftmax(nn.Module):
    """ Softmax variant based on arxiv 1805.10829 with added handles for base """
    def __init__(self, config, dim=-1):
        super().__init__()
        self.dim = dim

        # Set the base of the exponent
        if config.sigsoftmax_use_euler_base:
          self.sigsoftmax_base = math.e
        else:
          # custom base
          self.sigsoftmaxmax_base = config.sigsoftmax_base

    def forward(self, inputs):

        # Set exponent
        exp_x = torch.pow(self.sigsoftmax_base, inputs)

        # Similarly set sigmoid approximation
        sig_x = 1 / (1 + torch.pow(self.sigsoftmax_base, -inputs))

        # calculation of numerator and denominator
        numerator = exp_x * sig_x
        denominator = torch.sum(exp_x * sig_x, dim=self.dim, keepdim=True)

        return numerator / denominator

class SigmoidMax(nn.Module):
    def __init__(self, config, dim=-1):
        super().__init__()
        self.dim = dim
        self.sigmoid = nn.Sigmoid()
        self.sigmoidmax_divisor = config.sigmoidmax_divisor
        self.div_by_seq_len = config.div_by_seq_len

    def forward(self, x):

        result = self.sigmoid(x) / self.sigmoidmax_divisor

        # divide by sequence length
        if self.div_by_seq_len:
            seq_len = x.shape[self.dim]
            result = result / seq_len

        return result
    
### https://arxiv.org/pdf/2302.06461
### 논문 아님. 이 논문에서는 relu를 하고 gamma와 sequence length로 나눠줌. gamma는 학습 가능한 모델 상수 (graunuality : model)
class ReLUMax(nn.Module):
    def __init__(self, config, dim=-1):
        super().__init__()
        self.dim = dim
        self.relumax = nn.ReLU()
        self.relumax_divisor = config.relumax_divisor
        self.div_by_seq_len = config.div_by_seq_len

    def forward(self, x):

        result = self.relumax(x) / self.relumax_divisor

        # divide by sequence length
        if self.div_by_seq_len:
            seq_len = x.shape[self.dim]
            result = result / seq_len

        return result

class ReLU2Max(nn.Module):
    def __init__(self, config, dim=-1):
        super().__init__()
        self.dim = dim
        self.relu2max_divisor = config.relu2max_divisor
        self.div_by_seq_len = config.div_by_seq_len

    def forward(self, x):

        result = torch.relu(x) ** 2 / self.relu2max_divisor

        # divide by sequence length
        if self.div_by_seq_len:
            seq_len = x.shape[self.dim]
            result = result / seq_len

        return result


class Softplus(nn.Module):
    """ Softmax variant based on arxiv 1805.10829 with added handles for base """
    def __init__(self, config, dim=-1):
        super().__init__()
        self.dim = dim
        self.softplus = nn.Softplus()
        self.softplus_divisor = config.softplus_divisor
        self.div_by_seq_len = config.div_by_seq_len
        

    def forward(self, x):
        torch.cuda.nvtx.range_push("LSSA")
        softmaxPV_start_time = time.perf_counter()
        result = self.softplus(x) / self.softplus_divisor
        # divide by sequence length
        if self.div_by_seq_len:
            seq_len = x.shape[self.dim]
            result = result / seq_len
        softmaxPV_end_time = time.perf_counter()
        print(f"softmaPV time: {(softmaxPV_end_time - softmaxPV_start_time) * 1000000}")
        torch.cuda.nvtx.range_pop()

        return result


class Squareplus(nn.Module):
    def __init__(self, config, dim=-1, b=4.0*math.log(2)**2):
        super().__init__()
        self.b = b
        self.squareplus_divisor = config.squareplus_divisor
        self.div_by_seq_len = config.div_by_seq_len

    def forward(self, x):
        result = 0.5 * (x + torch.sqrt(x**2 + self.b)) / self.squareplus_divisor

        # divide by sequence length
        if self.div_by_seq_len:
            seq_len = x.shape[self.dim]
            result = result / seq_len

        return result

class LinearMax(nn.Module):
    def __init__(self, config, dim=-1):
        super().__init__()
        self.dim = dim
        self.n_head = config.n_head
        self.relu = nn.ReLU()

        ## x shape : (n_head, T, T) 
        self.beta_factor = nn.Parameter(torch.ones(self.n_head, 256, 1))
        self.gamma_factor = nn.Parameter(torch.ones(self.n_head, 256, 1))

    def forward(self, x):
        result = self.beta_factor * self.relu(x) + self.gamma_factor
        return result

# class LinearMax(nn.Module):               ## mask again for generative LLM's causal masking
#     def __init__(self, config, dim=-1):
#         super().__init__()
#         self.dim = dim
#         self.n_head = config.n_head
        
#         # x shape : (n_head, 256, 256)
#         self.size = 256  # Assuming 256x256 matrix

#         # Initialize parameters
#         self.beta_factor = nn.Parameter(torch.ones(self.n_head, self.size, self.size))
#         self.gamma_factor = nn.Parameter(torch.ones(self.n_head, self.size, self.size))
        
#         # Create lower triangular mask (1 for lower triangle, 0 elsewhere)
#         self.register_buffer('mask', torch.tril(torch.ones(self.size, self.size)).unsqueeze(0))

#     def forward(self, x):
#         # Apply mask to parameters
#         beta_masked = self.beta_factor * self.mask
#         gamma_masked = self.gamma_factor * self.mask
        
#         # Apply mask to input
#         x_masked = x * self.mask  # Ensure only lower triangular part of x is used
        
#         # Element-wise operation
#         result = beta_masked * x_masked + gamma_masked
        
#         return result


class EleMax(nn.Module):
    def __init__(self, config, dim=-1):
        super().__init__()
        self.dim = dim
        self.n_head = config.n_head
        self.relu = nn.ReLU()

        ## x shape : (n_head, T, T) 
        self.beta_factor = nn.Parameter(torch.ones(self.n_head, 1024, 1))

    def forward(self, x):
        result = self.beta_factor * self.relu(x)

        return result


class HeadMax(nn.Module):
    def __init__(self, config, dim=-1):
        super().__init__()
        self.dim = dim
        self.n_head = config.n_head
        self.relu = nn.ReLU()

        ## x shape : (n_head, T, T) 
        self.beta_factor = nn.Parameter(torch.ones(self.n_head, 1 , 1))

    def forward(self, x):
        result = self.beta_factor * self.relu(x)
        return result


### https://arxiv.org/pdf/2302.06461 : ReLUformer
# class ReLUMaxPaper(nn.Module):
#     """ReLU attention score with sqrt(n/2) scaling, where n is the current sequence length."""
#     def __init__(self, dim=-1):
#         super().__init__()
#         self.dim = dim
#         self.gamma = nn.Parameter(torch.tensor(1.0))
#         self.register_buffer('current_seq_len', torch.tensor(0))

#     def increment_seq_len(self):
#         self.current_seq_len += 1

#     def reset_seq_len(self):
#         self.current_seq_len.zero_()

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         torch.cuda.nvtx.range_push("ReLUmax")
#         #softmaxPV_start_time = time.perf_counter()
#         relu_out = torch.relu(x)
#         seq_len = self.current_seq_len.item()
#         if seq_len == 0:
#             seq_len = 1
#         result = relu_out / (self.gamma * math.sqrt(seq_len / 2) + 1e-6)
#         #softmaxPV_end_time = time.perf_counter()
#         #print(f"softmaPV time: {(softmaxPV_end_time - softmaxPV_start_time) * 1000000}")
#         torch.cuda.nvtx.range_pop()

#         return result

class ReLUMaxPaper(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim
        self.gamma = nn.Parameter(torch.tensor(1.0))
        # simple Python counter
        self.seq_len = 0

    def increment_seq_len(self):
        self.seq_len += 1

    def reset_seq_len(self):
        self.seq_len = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        softmaxPV_start_time = time.perf_counter()
        relu_out = F.relu(x)
        L = max(self.seq_len, 1)  # avoid zero
        # Python sqrt, but denom stays a float → tensor ops on GPU
        denom = (self.gamma * math.sqrt(L / 2.0) + 1e-6)
        softmaxPV_end_time = time.perf_counter()
        print(f"softmaPV time: {(softmaxPV_end_time - softmaxPV_start_time) * 1000000}")
        return relu_out / denom



from entmax import entmax_bisect
## 1.5-Entmax via Hugging Face entmax_bisect with α-scheduling
# class Entmax15(nn.Module):
#     def __init__(self, config, dim=-1):
#         super().__init__()
#         self.dim = dim
#         # scheduling parameters (defaults: softmax → pure entmax15 over 10000 steps)
#         self.alpha_start = getattr(config, 'entmax_alpha_start', 1.0)
#         self.alpha_end = getattr(config, 'entmax_alpha_end', 1.5)
#         self.schedule_steps = getattr(config, 'entmax_schedule_steps', 10000)
#         self.register_buffer('step', torch.tensor(0, dtype=torch.long))

#     def forward(self, x):
#         torch.cuda.nvtx.range_push("Entmax15")
#         softmaxPV_start_time = time.perf_counter()
#         # 1) 안정적 max-뺄셈
#         max_val, _ = x.max(dim=self.dim, keepdim=True)
#         x_stable = x - max_val

#         # 2) softmax 확률
#         p_soft = torch.exp(x_stable) / torch.sum(torch.exp(x_stable), dim=self.dim, keepdim=True)

#         # 3) entmax15 확률 (bisect-based)
#         # entmax_bisect returns probabilities summing to 1 along dim
#         p_ent = entmax_bisect(x_stable, alpha=1.5, dim=self.dim)

#         # 4) α 스케줄링: step 증가 후 비율 계산
#         step = min(self.step.item(), self.schedule_steps)
#         t = float(step) / float(self.schedule_steps)
#         alpha = self.alpha_start + t * (self.alpha_end - self.alpha_start)
#         lam = (alpha - self.alpha_start) / (self.alpha_end - self.alpha_start) if self.alpha_end != self.alpha_start else 1.0
#         self.step += 1

#         # 5) softmax↔entmax15 선형 보간
#         out = (1 - lam) * p_soft + lam * p_ent
#         softmaxPV_end_time = time.perf_counter()
#         print(f"softmaPV time: {(softmaxPV_end_time - softmaxPV_start_time) * 1000000}")
#         torch.cuda.nvtx.range_pop()
        
#         return out


class Entmax15(nn.Module):
    def __init__(self, config, dim=-1):
        super().__init__()
        self.dim = dim
        self.alpha_start = getattr(config, "entmax_alpha_start", 1.0)
        self.alpha_end   = getattr(config, "entmax_alpha_end",   1.5)
        self.schedule_steps = getattr(config, "entmax_schedule_steps", 10_000)
        self.register_buffer("step", torch.tensor(0, dtype=torch.long))

    def forward(self, x):
        # 1) α 값 계산 (스텝별)
        t = min(self.step.item(), self.schedule_steps) / self.schedule_steps
        alpha = self.alpha_start + t * (self.alpha_end - self.alpha_start)
        self.step += 1

        # 2) 바로 Entmax 호출 — Softmax·보간 제거
        return entmax_bisect(x, alpha=alpha, dim=self.dim)

# ================
# Per-Channel Symmetric Quantization Utilities
# ================
def elemax_quantize(tensor: torch.Tensor):
    """
    Per-channel symmetric quantization.
    tensor: (n_head, T, T)
    """
    # 1) 채널별 (n_head 기준) 최대 절댓값 계산
    maxabs = tensor.abs().amax(dim=(1, 2), keepdim=True)  # (n_head, 1, 1)
    # 2) maxabs를 이용하여 scale 계산 (대칭적 -128 ~ 127)
    scale = maxabs / 128.0  # (n_head, 1, 1)
    # 3) 양자화 수행
    quantized = torch.round(tensor / scale)  # [-128, 127] 범위로 매핑
    quantized = torch.clamp(quantized, -128, 127)  # int8 범위 제한
    quantized = quantized.to(torch.int8)  # int8 변환

    return quantized, scale  # (n_head, T, T), (n_head, 1, 1)

def elemax_dequantize(quantized: torch.Tensor, scale: torch.Tensor):
    """
    Per-channel symmetric dequantization.
    tensor: (n_head, T, T)
    scale: (n_head, 1, 1)
    """
    return quantized.float() * scale  # 원래 float 값으로 변환


# ================
# Per-Channel QAT Function with STE
# ================
class EleMaxQuanFunction(torch.autograd.Function):
    """Per-Channel Symmetric Quantization with STE (Straight-Through Estimator)"""
    @staticmethod
    def forward(ctx, beta):
        # Per-channel 양자화 수행
        beta_quantized, scale_beta = elemax_quantize(beta)
        beta_dequantized = elemax_dequantize(beta_quantized, scale_beta)

        # 스케일을 ctx에 저장 (Backprop을 위해)
        ctx.save_for_backward(scale_beta)

        return beta_dequantized

    @staticmethod
    def backward(ctx, grad_beta):
        return grad_beta  # STE: Gradients 그대로 전달

# Per-Channel QAT 적용 함수
_elemax_quan = EleMaxQuanFunction.apply


# ================
# Modified EleMax with Per-Channel Quantization
# ================
class EleMaxQuan(nn.Module):
    def __init__(self, config, dim=-1):
        super().__init__()
        self.dim = dim
        self.n_head = config.n_head
        self.relu = nn.ReLU()
        
        # Learnable beta factor (Per-Channel)
        self.beta_factor = nn.Parameter(torch.ones(self.n_head, 1024, 1))

    def forward(self, x):
        if self.training:
            # QAT 적용 (훈련 중)
            fake_beta = _elemax_quan(self.beta_factor)
            result = fake_beta * self.relu(x)
        else:
            # 실제 추론 시 Per-Channel 양자화 수행
            beta_quantized, scale_beta = elemax_quantize(self.beta_factor)
            beta_dequantized = elemax_dequantize(beta_quantized, scale_beta)
            result = beta_dequantized * self.relu(x)

        return result


class Sparsemax(nn.Module):
    """Sparsemax implementation (Martins & Astudillo, 2016)"""
    def __init__(self, config, dim=-1):
        super().__init__()
        self.dim = dim
        self.softmax_io_logging = config.softmax_io_logging
        self.softmax_io_log_interval = config.softmax_io_log_interval
        self.iter_num = 0
        
        if self.softmax_io_logging:
            self.inputs = []
            self.outputs = []

    def forward(self, x):
        # Store original shape
        original_size = x.size()
        
        # Reshape input for processing
        x = x.transpose(0, self.dim)
        x = x.reshape(x.size(0), -1)
        x = x.transpose(0, 1)
        
        # Translate input by max for numerical stability
        x = x - torch.max(x, dim=1, keepdim=True)[0].expand_as(x)
        
        # Sort input in descending order
        zs = torch.sort(input=x, dim=1, descending=True)[0]
        number_of_logits = x.size(1)
        range = torch.arange(start=1, end=number_of_logits + 1, step=1, 
                           device=x.device, dtype=x.dtype).view(1, -1)
        range = range.expand_as(zs)
        
        # Determine sparsity of projection
        bound = 1 + range * zs
        cumulative_sum_zs = torch.cumsum(zs, dim=1)
        is_gt = torch.gt(bound, cumulative_sum_zs).type(x.type())
        k = torch.max(is_gt * range, dim=1, keepdim=True)[0]
        
        # Compute threshold function
        zs_sparse = is_gt * zs
        
        # Compute taus
        taus = (torch.sum(zs_sparse, dim=1, keepdim=True) - 1) / k
        taus = taus.expand_as(x)
        
        # Compute sparsemax
        output = torch.max(torch.zeros_like(x), x - taus)
        
        # Reshape back to original shape
        output = output.transpose(0, 1)
        output = output.reshape(original_size)
        output = output.transpose(0, self.dim)
        
        if self.training and self.softmax_io_logging and self.iter_num % self.softmax_io_log_interval == 0:
            self.inputs = x
            self.outputs = output
            
        if self.training:
            self.iter_num += 1
            
        return output

softmax_dictionary = {
    # Note: we use the built in library for regular softmax
    "consmax": ConSmax,
    "consmax_v2": ConSmaxV2,
    "consmax_quan": ConSmaxQuan,
    "saturatingconsmax": SaturatingConSmax,
    "vpolymax": VPolymax,
    "polymax": Polymax,
    "exppolymax": ExpPolymax,
    "softermax": Softermax,
    "strongermax": Strongermax,
    "sigsoftmax": SigSoftmax,
    "relumax": ReLUMax,
    "relu2max": ReLU2Max,
    "sigmoidmax": SigmoidMax,
    "softplus": Softplus,
    "squareplus": Squareplus,
    "linearmax" : LinearMax,
    "consmax_v3" : ConSmaxV3,
    "elemax" : EleMax,
    "elemax_quan" : EleMaxQuan,
    "headmax" : HeadMax,
    "relumax_paper" : ReLUMaxPaper,
    "entmax15" : Entmax15,
    "sparsemax": Sparsemax,
}
