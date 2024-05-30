import os

import torch

# -----------------------------------------
# Paths
# -----------------------------------------

EXECUTION_NAME = "AAAPhysical - SSIMLoss"
MODEL_NAME = "UNet_phy" #UNet #UNet_phy
OPTIMIZER = "Adam"
CRITERION = "SSIMLoss" #UVLoss #HVLoss #MSELoss #SSIMLoss
PLOSS = False
WEIGHTSLOSSES = [1,1]
EXPERTYPE = 'Not fisical data, all scaled 0-1.'
WANDB = True
LAST_RUN_WANDB = ""



# -----------------------------------------

# Main steps
DO_TRAIN = True
USE_PHYSICAL_DATA = True
PARTITION = 1 # (0 - Random, 1- Patients)

# Paths -----------------------------------

PATH_DATASET = 'Server'
LOCAL_SLP_DATASET_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),'SLP/danaLab')
SERVER_SLP_DATASET_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))),'mnt/DADES2/SLP/SLP/danaLab')
MODELS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Models/SavedModels")
LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Logs")
IMG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),"img")

# Parameters ------------------------------
BATCH_SIZE_TRAIN = 32
BATCH_SIZE_TEST = 32
NUM_WORKERS = 2
EPOCHS = 100
LEARNING_RATE = 0.0002
LAMBDA_VALUE = 10

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#DEVICE = torch.device("cpu")
