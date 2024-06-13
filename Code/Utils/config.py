import os

import torch

# -----------------------------------------
# Training configuration
# -----------------------------------------

EXECUTION_NAME = "FE_100epochs_dadesUAB"
MODEL_NAME = "UNet" #UNet #UNet_phy
OPTIMIZER = "Adam"
CRITERION = "MSELoss" #UVLoss #HVLoss #MSELoss #SSIMLoss
EXPERTYPE = 'Not fisical data, all scaled 0-1.'
WANDB = False
LAST_RUN_WANDB = ""


# -----------------------------------------
# Main steps
# -----------------------------------------

DO_TRAIN = True
USE_PHYSICAL_DATA = False
PARTITION = 1 # (0 - Random, 1- Patients)

# -----------------------------------------
# Paths 
# -----------------------------------------

PATH_DATASET = 'Server'
LOCAL_SLP_DATASET_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),'SLP/danaLab')
SERVER_SLP_DATASET_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))),'mnt/DADES2/SLP/SLP/danaLab')
MODELS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Models/SavedModels")
LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Logs")
IMG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),"img")

# -----------------------------------------
# Parameters 
# -----------------------------------------

BATCH_SIZE_TRAIN = 32
BATCH_SIZE_TEST = 32
NUM_WORKERS = 2
EPOCHS = 100
LEARNING_RATE = 0.02
LAMBDA_VALUE = 10

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#DEVICE = torch.device("cpu")
