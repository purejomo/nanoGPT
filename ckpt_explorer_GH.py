import torch

def explore_ckpt(ckpt_path):
    """
    PyTorch 체크포인트 파일의 키와 구조를 탐색하는 함수.

    Args:
        ckpt_path (str): 체크포인트 파일 경로.
    """
    # 체크포인트 로드
    checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))

    # model 키 내부 탐색
    if "model" in checkpoint:
        print("\nKeys in 'model':")
        for key in checkpoint["model"].keys():
            print(f" - {key}")
    else:
        print("'model' key not found in the checkpoint.")

# 예시 사용
ckpt_path = "/home/ghlee/nanoGPT/params_nano/elemax-wikitext103/ckpt.pt"  # 파일 경로
explore_ckpt(ckpt_path)
