import os

import torch

# -----------------------------------------
# Paths
# -----------------------------------------

LOG_PATH = os.path.join(os.path.dirname(__file__), "Logs")
EXECUTION_NAME = "ConvAE_All_Imatges"
MODEL_NAME = "ConvAE"
OPTIMIZER = "Adam"
CRITERION = "MSELoss"
DATASET = "Server-Cropped"  # "Server-Cropped", "Server-Annotated", "Local-Cropped", "Local-Annotated"
WANDB = True
THRESHOLD_MSE = 0.5

# -----------------------------------------

# Main steps
DO_TRAIN = False
DO_TEST = True
DO_DETECTION = False

# Paths -----------------------------------
SERVER_CROPED_DATASET_PATH = "/fhome/mapsiv/QuironHelico/CroppedPatches"
SERVER_ANNOTATED_DATASET_PATH = "/fhome/mapsiv/QuironHelico/AnnotatedPatches"
LOCAL_CROPPED_DATASET_PATH = os.path.join(os.path.dirname(__file__), "../1. Dataset/CroppedPatches")
LOCAL_ANNOTATED_DATASET_PATH = os.path.join(os.path.dirname(__file__), "../1. Dataset/AnnotatedPatches")
MODELS_PATH = os.path.join(os.path.dirname(__file__), "models")
OUTPUTS_PATH = os.path.join(os.path.dirname(__file__), "outputs")

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


# Dimensions ------------------------------
THRESHOLD_P = 0.027755102040816326

SERGI_TOKEN = "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"
