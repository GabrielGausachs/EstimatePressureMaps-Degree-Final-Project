import os

import torch

# -----------------------------------------
# Paths
# -----------------------------------------

EXECUTION_NAME = "Prova"
MODEL_NAME = "SimpleUNet"
OPTIMIZER = "Adam"
CRITERION = "MSELoss"
DATASET = "Local-SLP"
WANDB = False
THRESHOLD_MSE = 0.5

# -----------------------------------------

# Main steps
DO_TRAIN = True
DO_TEST = True
IS_RANDOM = True
SHOW_IMAGES = False
SHOW_HISTOGRAM = False
USE_PHYSICAL_DATA = False

# Paths -----------------------------------

LOCAL_SLP_DATASET_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),'SLP/danaLab')
SERVER_SLP_DATASET_PATH = ""
MODELS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Models")
LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Logs")
IMG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),"img")

# Parameters ------------------------------
BATCH_SIZE_TRAIN = 128
BATCH_SIZE_TEST = 128
NUM_WORKERS = 2
EPOCHS = 2
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
MOMENTUM = 0.9

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#DEVICE = torch.device("cpu")