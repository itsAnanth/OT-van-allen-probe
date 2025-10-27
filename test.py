import torch

data = torch.load('checkpoints/54kev_19.pth', map_location=torch.device('cpu'), weights_only=True)
print(data)