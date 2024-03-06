import os

import torch

# -----------------------------------------
# Paths
# -----------------------------------------

EXECUTION_NAME = "Prova"
MODEL_NAME = "UNet"
OPTIMIZER = "Adam"
CRITERION = "MSELoss"
DATASET = "Local-SLP"  # "Server-Cropped", "Server-Annotated", "Local-Cropped", "Local-Annotated"
WANDB = True
THRESHOLD_MSE = 0.5

# -----------------------------------------

# Main steps
DO_TRAIN = False
DO_TEST = True
SHOW_IMAGES = False
USE_PHYSICAL_DATA = True

# Paths -----------------------------------

LOCAL_SLP_DATASET_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),'SLP/danaLab')
SERVER_SLP_DATASET_PATH = ""
MODELS_PATH = os.path.join(os.path.dirname(__file__), "Models")
LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Logs")

# Parameters ------------------------------
BATCH_SIZE_TRAIN = 128
BATCH_SIZE_TEST = 128
NUM_WORKERS = 2
EPOCHS = 40
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
MOMENTUM = 0.9
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")