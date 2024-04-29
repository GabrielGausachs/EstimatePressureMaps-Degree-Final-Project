import os

import torch

# -----------------------------------------
# Paths
# -----------------------------------------

EXECUTION_NAME = "UNet_no_phy_MSELoss"
MODEL_NAME = "UNet"
OPTIMIZER = "Adam"
CRITERION = "MSELoss" #UVLoss #HVLoss #MSELoss
METRIC = "PerCS" #PerCS #MSELoss
EXPERTYPE = 'np-input-norm-patients'
WANDB = True
LAST_RUN_WANDB = ""

# -----------------------------------------

# Main steps
DO_TRAIN = True
USE_PHYSICAL_DATA = False
EVALUATION = False
PARTITION = 0 # (0 - Random, 1- Patients)

# Paths -----------------------------------

PATH_DATASET = 'Server'
LOCAL_SLP_DATASET_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),'SLP/danaLab')
SERVER_SLP_DATASET_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))),'mnt/DADES2/SLP/SLP/danaLab')
MODELS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Models/SavedModels")
LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Logs")
IMG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),"img")
LAST_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Models/SavedModels/UNet_20240416113035.pth")


# Parameters ------------------------------
BATCH_SIZE_TRAIN = 32
BATCH_SIZE_TEST = 32
NUM_WORKERS = 2
EPOCHS = 50
LEARNING_RATE = 0.02
LAMBDA_VALUE = 10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#DEVICE = torch.device("cpu")
