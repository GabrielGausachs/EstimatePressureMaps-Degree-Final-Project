import os

import torch

# -----------------------------------------
# Paths
# -----------------------------------------

EXECUTION_NAME = "UNet-first_mv"
MODEL_NAME = "UNet"
OPTIMIZER = "Adam"
CRITERION = "PWRSWtL"
EXPERTYPE = 'only-np-no-normalization-lambda-10'
WANDB = True
LAST_RUN_WANDB = ""

# -----------------------------------------

# Main steps
DO_TRAIN = True
DO_TEST = False
SHOW_IMAGES = False
SHOW_HISTOGRAM = False
USE_PHYSICAL_DATA = False
EVALUATION = False

# Paths -----------------------------------

PATH_DATASET = 'Server'
LOCAL_SLP_DATASET_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),'SLP/danaLab')
SERVER_SLP_DATASET_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))),'mnt/DADES2/SLP/SLP/danaLab')
MODELS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Models/SavedModels")
LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Logs")
IMG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),"img")
LAST_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Models/SavedModels/UNet_20240325185958.pth")


# Parameters ------------------------------
BATCH_SIZE_TRAIN = 128
BATCH_SIZE_TEST = 128
NUM_WORKERS = 2
EPOCHS = 15
LEARNING_RATE = 0.0003
WEIGHT_DECAY = 1e-5
MOMENTUM = 0.9

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#DEVICE = torch.device("cpu")