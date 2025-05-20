import torch
from entmax import entmax15  # HuggingFace 구현
import numpy as np

# --- (A) HuggingFace entmax15 래퍼
def hf_entmax15(x):
    # entmax15 함수는 이미 softmax와 유사한 API: entmax15(x, dim=-1)
    return entmax15(x, dim=-1)

# --- (B) 사용자 구현 Entmax15 (forward 안에 τ·support_size 프린트 추가)
class MyEntmax15(torch.nn.Module):
    def __init__(self, dim=-1, eps=1e-8):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x):
        max_val, _ = x.max(dim=self.dim, keepdim=True)
        z = (x - max_val) / 2

        # 정렬
        z_srt, idx = torch.sort(z, dim=self.dim, descending=True)
        size = z.size(self.dim)
        rho = torch.arange(1, size + 1, device=x.device, dtype=x.dtype)
        if self.dim != -1 and self.dim != x.dim() - 1:
            shape = [1]*x.dim(); shape[self.dim]=size
            rho = rho.view(shape)

        cumsum = z_srt.cumsum(dim=self.dim)
        mean = cumsum / rho
        cumsum_sq = (z_srt**2).cumsum(dim=self.dim)
        mean_sq = cumsum_sq / rho

        ss = rho * (mean_sq - mean**2)
        delta = (1 - ss) / rho
        delta_nz = torch.clamp(delta, min=0)
        tau = mean - torch.sqrt(delta_nz)

        # support mask
        support_mask = (z_srt >= tau).float()
        support_size = support_mask.sum(dim=self.dim)

        # τ* 추출
        safe_idx = (support_size.clamp(min=1,max=size)-1).long()
        safe_idx = safe_idx.view(-1, *([1]*(z_srt.dim()-2)), 1).expand_as(z_srt)
        tau_star = torch.gather(tau, self.dim, safe_idx)

        out = torch.clamp(z - tau_star, min=0)**2
        norm = out.sum(dim=self.dim, keepdim=True) + self.eps
        out = out / norm

        # 디버깅 정보 출력
        print(f"=== Debug Entmax15 ===\n"
              f"Input z (scaled): {z.tolist()}\n"
              f"Sorted z:           {z_srt.tolist()}\n"
              f"Tau:                {tau.tolist()}\n"
              f"Support size:       {support_size.tolist()}\n"
              f"Output dist:        {out.tolist()}\n"
              f"======================")

        return out

# --- (C) 비교 실행
if __name__ == "__main__":
    # 테스트용 임의 입력
    x = torch.tensor([1.2, 0.5, -0.3, 2.4, 1.0])

    print(">> Hugging Face Entmax15")
    y_hf = hf_entmax15(x)
    print(y_hf.tolist())

    print("\n>> My Entmax15")
    model = MyEntmax15(dim=0)
    y_my = model(x)
    print(y_my.tolist())

    # 두 결과의 L1 차이
    diff = torch.abs(y_hf - y_my).sum().item()
    print(f"\nL1 difference: {diff:.6f}")
