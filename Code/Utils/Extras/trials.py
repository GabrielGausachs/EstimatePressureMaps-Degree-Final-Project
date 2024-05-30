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



params = [31055183,7762465,1942289,486409]
values = [1.553,1.60,1.71,1.798]
acc = [0.9048,0.8996,0.8955,0.8876]

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(params, values, marker='o', linestyle='-', color='b')

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

# Add labels and title
plt.xlabel('Number of Parameters', fontsize=18)
plt.ylabel('MSELoss', fontsize=18)
plt.title('Model MSELoss vs. Number of Parameters', fontsize=20)

# Show the plot
plt.show()

"""
import wandb
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# Initialize the wandb API
api = wandb.Api()

# Replace with your actual project details
entity = "1604373"  # Your username or team name
project = "TFG"  # Your project name

run_id = "2w4l4ccz"  # The specific run ID
# Access the run
run = api.run(f"{entity}/{project}/{run_id}")
# Fetch the logged data
history = run.history(keys=["Train SSIMMetric()","Val SSIMMetric()", "Train MSELoss()", "Val MSELoss()"])
# Convert the history to a pandas DataFrame
history_df = pd.DataFrame(history)
# Display the first few rows to check the data
print(history_df.head())

run_id = "wa0fkkul"  # The specific run ID
# Access the run
run_2 = api.run(f"{entity}/{project}/{run_id}")
# Fetch the logged data
history_2 = run_2.history(keys=["Train SSIMMetric()","Val SSIMMetric()", "Train MSELoss()", "Val MSELoss()"])
# Convert the history to a pandas DataFrame
history_df_2 = pd.DataFrame(history_2)
print(history_df_2.head())


run_id = "faz6krvy"  # The specific run ID
# Access the run
run_3 = api.run(f"{entity}/{project}/{run_id}")
# Fetch the logged data
history_3 = run_3.history(keys=["Train SSIMMetric()","Val SSIMMetric()", "Train MSELoss()", "Val MSELoss()"])
# Convert the history to a pandas DataFrame
history_df_3 = pd.DataFrame(history_3)
print(history_df_3.head())


# Plotting
plt.figure(figsize=(10, 6))



plt.plot(history_df['_step'], history_df['Train SSIMMetric()'], label='Model-MSE train', color='blue')
plt.plot(history_df['_step'],history_df['Val SSIMMetric()'], label='Model-MSE validation', color='blue', linestyle='--')
plt.plot(history_df_2['_step'],history_df_2['Train SSIMMetric()'], label='Model-HV train', color='red')
plt.plot(history_df_2['_step'],history_df_2['Val SSIMMetric()'], label='Model-HV validation', color='red', linestyle='--')
plt.plot(history_df_3['_step'],history_df_3['Train SSIMMetric()'], label='Model-SSIM train', color='green')
plt.plot(history_df_3['_step'],history_df_3['Val SSIMMetric()'], label='Model-SSIM validation', color='green', linestyle='--')
# Adding labels and title

plt.ylim(0.4, 1)
plt.xlabel('Epochs', fontsize=18)
plt.ylabel('SSIm', fontsize=18)
plt.title('Training and Validation SSIM for these 3 models', fontsize=20)

# Adding legend
plt.legend(fontsize=16)

#plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
#plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

# Display the plot
plt.show()

