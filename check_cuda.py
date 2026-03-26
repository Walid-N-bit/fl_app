try:
    import torch

    print(f"{torch.__version__ = }")
    print(f"{torch.cuda.is_available() = }")
except Exception as e:
    print(f"Ignored error: {e}")
