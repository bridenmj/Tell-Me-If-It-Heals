import torch

CLIP = 0.25
D_UPDATE_THRESHOLD = 2.0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"