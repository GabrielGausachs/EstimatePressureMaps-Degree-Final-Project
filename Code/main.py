from Utils import logger

import wandb
import datetime

from config import (
    WANDB,
    EXECUTION_NAME,
    MODEL_NAME,
    OPTIMIZER,
    CRITERION,
    LEARNING_RATE,
    EPOCHS,
    THRESHOLD_MSE
    )


if __name__ == "__main__":
    # Initialize logger
    logger.initialize_logger()
    logger = logger.get_logger()

    # Initialize wandb
    if WANDB:
        wandb.login()
        wandb.init(
            project="TFG",
            entity='1604373',
            name=f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}_{EXECUTION_NAME}",
            config={
                    "model_name": MODEL_NAME,
                    "optimizer": OPTIMIZER,
                    "criterion": CRITERION,
                    "learning_rate": LEARNING_RATE,
                    "epochs": EPOCHS,
                    "threshold_mse": THRESHOLD_MSE,
            },
        )