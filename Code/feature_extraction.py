import wandb
import datetime
import torch
from matplotlib import pyplot as plt
import os
import math
import copy
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
from Utils.logger import initialize_logger, get_logger
from Utils.dataset import CustomDataset
from torchvision.transforms.functional import crop
from glob import glob
import matplotlib.image as mpimg
import numpy as np
import pandas as pd

from Utils import (
    logger,
    dataloader,
    train,
    savemodel,
    losses,
    val,
    metrics
    )

from Models import (
    UNet,
    UNet_phy
    )


from Utils.config import (
    WANDB,
    EXECUTION_NAME,
    MODEL_NAME,
    OPTIMIZER,
    CRITERION,
    LEARNING_RATE,
    EPOCHS,
    DEVICE,
    DO_TRAIN,
    EXPERTYPE,
    USE_PHYSICAL_DATA,
    BATCH_SIZE_TEST,
    BATCH_SIZE_TRAIN,
    PLOSS,
    WEIGHTSLOSSES
    )

# Models
models = {"UNet": UNet.UNET, "UNet_phy": UNet_phy.UNET_phy}

# Optimizers
optimizers = {
    "Adam": torch.optim.Adam
    }

# Criterion
criterion = {
    "MSELoss": torch.nn.MSELoss(),
    "HVLoss": losses.HVLoss(),
    "PLoss": losses.PhyLoss(),
    "SSIMLoss": losses.SSIMLoss()
    }

# Extra loss to combine
extra_loss = {True: losses.PhyLoss(),
            False: None}

# Metrics
metrics = [
    torch.nn.MSELoss(),
    metrics.PerCS(),
    metrics.MSEeff(),
    metrics.SSIMMetric()
    ]

def to_float32_and_scale(tensor,global_min,global_max):
    tensor = tensor.float()
    tensor = (tensor - global_min) / (global_max - global_min)
    return tensor

def crop_array(array):
    
    return crop(array, 20, 28, 85, 36)

logger.initialize_logger()
logger = logger.get_logger()

path_data = os.path.join(os.path.dirname(os.path.dirname((os.getcwd()))),'DadesUAB/Data')
    
transform = {
            'input': transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(crop_array),
                transforms.Resize((192, 84))]),
            'output': transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(1),
                transforms.Lambda(lambda x: to_float32_and_scale(x,0,905)),
                transforms.Resize((192, 84))])
}


images_tensor = {}
images_tensor['input']=[]
images_tensor['output']=[]
folders = os.listdir(path_data)
folders_sorted = sorted(folders)
print(folders_sorted)


for folder in folders_sorted:
    directory = os.path.join(path_data,folder)
    #print(directory)
    pattern = os.path.join(directory, '*IR.png')
    files_ir = glob(pattern)
    files_ir = sorted(files_ir)
    #print(len(files_ir))
    #print(files_ir)
    pattern = os.path.join(directory, '*Pressio.csv')
    files_pm = glob(pattern)
    files_pm = sorted(files_pm)
    #print(len(files_pm))
    #print(files_pm)

    for ir,pm in zip(files_ir,files_pm):
        #print(ir)
        #print(pm)
        ir_array = mpimg.imread(ir)
        array = np.rot90(ir_array, k=1, axes=(1,0))
        array_2 = np.copy(array)
        tensor_final = transform['input'](array_2)
        images_tensor['input'].append(tensor_final)
        ir_array = tensor_final.squeeze().numpy()
        pm = pd.read_csv(pm)
        pm_array = pm.to_numpy()
        pm_array = np.rot90(pm_array, k=1, axes=(1,0))
        pm_array_2 = np.copy(pm_array)
        pm_tensor = transform['output'](pm_array_2)
        images_tensor['output'].append(pm_tensor)
        final_array = pm_tensor.squeeze().numpy()

class CustomDataset(Dataset):
    def __init__(self, data_dict):
        self.inputs = data_dict['input']
        self.outputs = data_dict['output']
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        input_tensor = self.inputs[idx]
        output_tensor = self.outputs[idx]
        return input_tensor,output_tensor


indices = list(range(len(images_tensor['input'])))

np.random.shuffle(indices)

# Calculate the split index
num_samples = len(images_tensor['input'])
split_idx = int(0.8 * num_samples)

# Split the indices
train_indices = indices[:split_idx]
test_indices = indices[split_idx:]
print(train_indices)
print(test_indices)

train_data = {'input': [images_tensor['input'][i] for i in train_indices], 'output': [images_tensor['output'][i] for i in train_indices]}
test_data = {'input': [images_tensor['input'][i] for i in test_indices], 'output': [images_tensor['output'][i] for i in test_indices]}

custom_dataset_train = CustomDataset(train_data)
custom_dataset_test = CustomDataset(test_data)


train_loader = DataLoader(custom_dataset_train, batch_size=32, shuffle=False, num_workers=0, drop_last=True)
train_dataset_info = {
            'Number of samples': len(train_loader.dataset),
            'Batch size': train_loader.batch_size,
            'Number of batches': len(train_loader)
        }
print(train_dataset_info)
val_loader = DataLoader(custom_dataset_test, batch_size=32, shuffle=False, num_workers=0, drop_last=True)


from Models import (
    UNet,
    UNet_phy
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_NAME = "UNet"
name_model = "AAA - MSELoss - 512_20240529200423.pth"

models = {"UNet": UNet.UNET, "UNet_phy": UNet_phy.UNET_phy}

features = [64,128,256,512]

model = models[MODEL_NAME](1, 1,features).to(DEVICE)

# Load the model
model_file = os.path.join(os.path.join((os.getcwd()),'Models/SavedModels'),name_model)

try:
    model.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))
    model.to(DEVICE)
except Exception as e:
    raise RuntimeError(f"An error occurred while reading the file: {e}")
else:
    print('Model Loaded')

model.freeze_except_final()
for name, param in model.named_parameters():
    print(f"{name}: requires_grad={param.requires_grad}")

optimizer = optimizers[OPTIMIZER](model.parameters(), lr=LEARNING_RATE)
# Create a criterion object
criterion = criterion[CRITERION]

# Iterate over training and test
for epoch in range(EPOCHS):
    epoch_loss_train, epoch_metric_train = train.train(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                metrics=metrics,
                epoch=epoch,
                epochs=EPOCHS,
                ploss=extra_loss[PLOSS],
                weightsloss=WEIGHTSLOSSES,
            )
    epoch_loss_val, epoch_metric_val = val.val(
                model=model,
                loader=val_loader,
                metrics=metrics,
                criterion=criterion,
                epoch=epoch,
                epochs=EPOCHS,
                ploss=extra_loss[PLOSS],
                weightsloss=WEIGHTSLOSSES,
            )
    print('Train loss:', epoch_loss_train)
    print('Train metrci:', epoch_metric_train)
    print('Val loss:', epoch_loss_val)
    print('Val metric:', epoch_metric_val)



"""
for input_images,target in test_loader:
    input_img = input_images.to(DEVICE)
    output_img = model(input_img)
    input_images = input_img.squeeze().cpu().numpy()
    output_images = output_img.squeeze().detach().cpu().numpy()
    target_img = target.squeeze().cpu().numpy()

    #kk = max(np.max(target_img),np.max(output_images))

    #target_img[0,0] = kk
    #output_images[0,0] = kk
    print('output:',np.max(output_images),np.min(output_images))
    print('target:',np.max(target_img),np.min(target_img))
    
        
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(input_images)
    axes[0].axis('off')
    
    axes[1].imshow(target_img)
    axes[1].axis('off')
    
    
    axes[2].imshow(output_images)
    axes[2].axis('off')
    
    fig.suptitle(f'Input and Output Image in UAB', fontsize=12)
    plt.show()
"""

