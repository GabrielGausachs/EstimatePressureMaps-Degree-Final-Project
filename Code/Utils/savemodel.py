from Utils.logger import initialize_logger,get_logger
import torch
import datetime

from Utils.config import (
    MODEL_NAME,
    MODELS_PATH,
)


logger = get_logger()

def save_model(model):
    # Save the model pth and the arquitecture
    logger.info("Saving the model and its architecture")
    model.to("cpu")
    torch.save(
        model.state_dict(), f"{MODELS_PATH}/{MODEL_NAME}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.pth"
    )

    file_path = f"{MODELS_PATH}/{MODEL_NAME}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}_arch.txt"
    with open(file_path, "w") as f:
        f.write(str(model))
    logger.info(f"Model saved in {MODELS_PATH}")
    logger.info("-" * 50)