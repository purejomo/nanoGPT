import torch
import matplotlib.pyplot as plt

def plot_beta_factor_distribution(ckpt_path, save_path=None):
    """
    체크포인트에서 특정 beta_factor의 weight 값 분포를 히스토그램으로 출력하는 함수.

    Args:
        ckpt_path (str): 체크포인트 파일 경로.
        save_path (str, optional): 그래프를 저장할 파일 경로. None이면 화면에 표시.
    """
    # 체크포인트 로드
    checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
    
    # beta_factor 키 리스트
    beta_factor_keys = [
        "transformer.h.0.attn.softmax_layer_attn.beta_factor",
    ]
    
    # beta_factor 값 수집
    beta_factors = {}
    for key in beta_factor_keys:
        if key in checkpoint["model"]:
            beta_factors[key] = checkpoint["model"][key].numpy().flatten()
    
    if not beta_factors:
        print("No specified 'beta_factor' keys found in the checkpoint.")
        return
    
    # 히스토그램 플롯
    plt.figure(figsize=(8, 6))
    for key, values in beta_factors.items():
        plt.hist(values, bins=50, alpha=0.5, label=key, edgecolor='black')
    
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title("Distribution of beta_factor Weights")
    plt.legend()
    plt.grid(True)
    
    # 저장 또는 표시
    if save_path:
        plt.savefig(save_path)
        print(f"Histogram saved to {save_path}")
    else:
        plt.show()

# 사용 예시
ckpt_path = "/home/ghlee/nanoGPT/params_nano/elemax-wikitext103/ckpt.pt"  # 파일 경로
save_path = "beta_factor_distribution.png"  # 저장할 파일 경로
plot_beta_factor_distribution(ckpt_path, save_path)
