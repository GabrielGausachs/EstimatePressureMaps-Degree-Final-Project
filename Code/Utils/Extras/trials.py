import torch
from matplotlib import pyplot as plt

"""
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
count = torch.sum(diff < max_values_per_image.unsqueeze(1).unsqueeze(2).expand(2,10,2) * 0.1).item()

pcs = count / tar.numel()

print(pcs)
"""
params = [31055183,7762465,1942289,486409]
values = [1.585,1.661,1.81,1.831]
acc = [0.9048,0.8996,0.8955,0.8876]

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(params, acc, marker='o', linestyle='-', color='b')

# Add labels and title
plt.xlabel('Number of Parameters')
plt.ylabel('PerCS')
plt.title('Model PerCS vs. Number of Parameters')

# Show the plot
plt.show()