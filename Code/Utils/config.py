import os

import torch

# -----------------------------------------
# Paths
# -----------------------------------------

EXECUTION_NAME = "UNetPDataMSE%PLoss_Type2"
MODEL_NAME = "UNet_phy" #UNet #UNet_phy
MAX_FEATURE = 1024
OPTIMIZER = "Adam"
CRITERION = "MSELoss" #UVLoss #HVLoss #MSELoss
PLOSS = True
WEIGHTSLOSSES = [1,1]
METRIC = "PerCS" #PerCS #MSELoss
EXPERTYPE = 'PdataWithOutGender-NoNormalization'
WANDB = True
LAST_RUN_WANDB = ""

# -----------------------------------------

# Main steps
DO_TRAIN = True
USE_PHYSICAL_DATA = True
EVALUATION = False
PARTITION = 1 # (0 - Random, 1- Patients)

# Paths -----------------------------------

PATH_DATASET = 'Server'
LOCAL_SLP_DATASET_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),'SLP/danaLab')
SERVER_SLP_DATASET_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))),'mnt/DADES2/SLP/SLP/danaLab')
MODELS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Models/SavedModels")
LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Logs")
IMG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),"img")
LAST_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Models/SavedModels/UNet_20240429150032.pth")


# Parameters ------------------------------
BATCH_SIZE_TRAIN = 32
BATCH_SIZE_TEST = 32
NUM_WORKERS = 2
EPOCHS = 100
LEARNING_RATE = 0.05
LAMBDA_VALUE = 10

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#DEVICE = torch.device("cpu")
