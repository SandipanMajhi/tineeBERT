import torch

history = torch.load("Checkpoints/history_v1.pt")
print(history["loss"][-20:])