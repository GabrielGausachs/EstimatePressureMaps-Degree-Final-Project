import os

import torch

# -----------------------------------------
# Paths
# -----------------------------------------

EXECUTION_NAME = "UNet_no_phy_PWRS"
MODEL_NAME = "UNet"
OPTIMIZER = "Adam"
CRITERION = "HVLoss" #PWRSWtL #HVLoss #MSELoss
EXPERTYPE = 'np-input-norm-patients'
WANDB = False
LAST_RUN_WANDB = ""

# -----------------------------------------

# Main steps
DO_TRAIN = False
DO_TEST = False
SHOW_IMAGES = False
SHOW_HISTOGRAM = False
USE_PHYSICAL_DATA = False
EVALUATION = True
PARTITION = 1 # (0 - Random, 1- Patients)

# Paths -----------------------------------

PATH_DATASET = 'Server'
LOCAL_SLP_DATASET_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),'SLP/danaLab')
SERVER_SLP_DATASET_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))),'mnt/DADES2/SLP/SLP/danaLab')
MODELS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Models/SavedModels")
LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Logs")
IMG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),"img")
LAST_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Models/SavedModels/UNet_20240416113035.pth")


# Parameters ------------------------------
BATCH_SIZE_TRAIN = 64
BATCH_SIZE_TEST = 64
NUM_WORKERS = 2
EPOCHS = 5
LEARNING_RATE = 0.002
WEIGHT_DECAY = 1e-5
MOMENTUM = 0.9
LAMBDA_VALUE = 10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#DEVICE = torch.device("cpu")
