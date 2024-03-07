from Utils import (
    logger, 
    dataloader,
    train
)

import wandb
import datetime
import torch

from Utils.config import (
    WANDB,
    EXECUTION_NAME,
    MODEL_NAME,
    OPTIMIZER,
    CRITERION,
    LEARNING_RATE,
    EPOCHS,
    THRESHOLD_MSE,
    DEVICE
    )

from Models import (
    UNet,
)

# Models
models = {"UNet": UNet.UNet}

# Optimizers
optimizers = {
    "Adam": torch.optim.Adam,
}

# Criterion
criterion = {
    "MSELoss": torch.nn.MSELoss,
}


if __name__ == "__main__":
    print('helloworld')
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
            save_code=False
        )

    train_loader, val_loader = dataloader.CustomDataloader().prepare_dataloaders(False)

    # Create a model
    model = models[MODEL_NAME](3,3).to(DEVICE)

    # Create an optimizer object
    optimizer = optimizers[OPTIMIZER](model.parameters(), lr=LEARNING_RATE)

    # Create a criterion object
    criterion = criterion[CRITERION]()

    # Iterate over training and test
    for epoch in range(EPOCHS):
        logger.info(f"--- Epoch: {epoch} ---")
        train(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            epoch=epoch,
            epochs=EPOCHS,
        )



    if WANDB:
        wandb.finish()