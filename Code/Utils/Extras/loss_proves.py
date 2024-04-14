import torch
import torch.nn as nn

# Define the shape
shape = (5, 1, 3, 4)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generate random integer tensor
random_tensor = torch.randint(low=0, high=25, size=shape).to(DEVICE)

random_tensor = random_tensor.float()

print(random_tensor)

p_y = torch.histc(random_tensor.view(-1), bins=256, min=0, max=255) / (random_tensor.numel() * 5)

print(p_y)
