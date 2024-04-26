import torch

# Generate a tensor with random integer values
tar= torch.randint(low=0, high=100, size=(2, 10, 2), dtype=torch.int)
src = torch.randint(low=0, high=100, size=(2, 10, 2), dtype=torch.int)

diff = torch.abs(tar-src)

reshaped_tensor = tar.view(tar.size(0), -1)
print(reshaped_tensor)

# Find the maximum value for each image in the batch
max_values_per_image, _ = torch.max(reshaped_tensor, dim=1)
print(max_values_per_image)

# Now you can proceed with the rest of your computation
count = torch.sum(diff < max_values_per_image.unsqueeze(1).repeat(1, 10, 2) * 0.01).item()

pcs = count / tar.numel()

print(pcs)
