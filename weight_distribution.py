import torch
import matplotlib.pyplot as plt
import os
import re

def plot_beta_factor_weights(ckpt_path, save_dir=None):
    """
    체크포인트에서 모든 layer의 beta_factor weight 값을 플롯하는 함수.
    
    Args:
        ckpt_path (str): 체크포인트 파일 경로.
        save_dir (str, optional): 그래프를 저장할 디렉토리 경로. None이면 화면에 표시.
    """
    # 체크포인트 로드
    checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
    
    # beta_factor 키를 모두 찾기
    beta_factor_pattern = r"transformer\.h\.(\d+)\.attn\.softmax_layer_attn\.beta_factor"
    beta_factors = {}
    
    for key in checkpoint["model"].keys():
        match = re.match(beta_factor_pattern, key)
        if match:
            layer_num = int(match.group(1))
            beta_factors[key] = {
                'layer': layer_num,
                'values': checkpoint["model"][key].numpy().flatten()
            }
    
    if not beta_factors:
        print("No 'beta_factor' keys found in the checkpoint.")
        return
    
    # 저장 디렉토리 생성
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # 레이어별 개별 플롯 생성
    for key, data in beta_factors.items():
        plt.figure(figsize=(10, 6))
        values = data['values']
        plt.scatter(range(len(values)), values, s=10, alpha=0.6)
        
        plt.xlabel("Parameter Index")
        plt.ylabel("Beta Factor Value")
        plt.title(f"Layer {data['layer']} Beta Factor Weights Distribution")
        plt.grid(True)
        
        if save_dir:
            file_path = os.path.join(save_dir, f"beta_factor_layer_{data['layer']}.png")
            plt.savefig(file_path)
            plt.close()
            print(f"Plot saved to {file_path}")
        else:
            plt.show()
    
    # 모든 레이어 함께 시각화
    if len(beta_factors) > 1:
        plt.figure(figsize=(12, 8))
        
        for key, data in beta_factors.items():
            plt.plot(data['values'], label=f"Layer {data['layer']}")
        
        plt.xlabel("Parameter Index")
        plt.ylabel("Beta Factor Value")
        plt.title("Beta Factor Weights Distribution Across Layers")
        plt.legend()
        plt.grid(True)
        
        if save_dir:
            file_path = os.path.join(save_dir, "beta_factor_all_layers.png")
            plt.savefig(file_path)
            plt.close()
            print(f"All layers plot saved to {file_path}")
        else:
            plt.show()
    
    # 히트맵으로 모든 레이어 시각화
    if len(beta_factors) > 1:
        layers = sorted(list(set(data['layer'] for data in beta_factors.values())))
        max_param_count = max(len(data['values']) for data in beta_factors.values())
        
        plt.figure(figsize=(12, 8))
        heatmap_data = []
        
        for layer in layers:
            layer_key = [k for k, v in beta_factors.items() if v['layer'] == layer][0]
            values = beta_factors[layer_key]['values']
            # Pad with NaN if needed
            padded_values = list(values) + [float('nan')] * (max_param_count - len(values))
            heatmap_data.append(padded_values)
        
        plt.imshow(heatmap_data, aspect='auto', cmap='viridis')
        plt.colorbar(label='Beta Factor Value')
        plt.xlabel('Parameter Index')
        plt.ylabel('Layer')
        plt.yticks(range(len(layers)), layers)
        plt.title('Beta Factor Values Across All Layers')
        
        if save_dir:
            file_path = os.path.join(save_dir, "beta_factor_all_layers_heatmap.png")
            plt.savefig(file_path)
            plt.close()
            print(f"Heatmap saved to {file_path}")
        else:
            plt.show()

# 사용 예시
ckpt_path = "/home/ghlee/nanoGPT/params_small/elemax-witkitext103-re/ckpt.pt"  # 파일 경로
save_dir = "elemax_beta_factors"  # 저장할 디렉토리 경로
plot_beta_factor_weights(ckpt_path, save_dir)
