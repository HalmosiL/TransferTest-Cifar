from models import getModel
import torch

for i in range(10):
    torch.save(getModel().state_dict(), f'./Models/Models_{i}.pt')