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
    DEVICE,
    DO_TRAIN,
    MODELS_PATH
    )

from Models import (
    UNet,
    SimpleUNet
)

# Models
models = {"UNet": UNet.UNet, "SimpleUNet": SimpleUNet.UNet}

# Optimizers
optimizers = {
    "Adam": torch.optim.Adam,
}

# Criterion
criterion = {
    "MSELoss": torch.nn.MSELoss,
}


if __name__ == "__main__":
    # Initialize logger
    logger.initialize_logger()
    logger = logger.get_logger()

    train_loader, val_loader = dataloader.CustomDataloader().prepare_dataloaders(True)

    if DO_TRAIN:
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


        # Create a model
        model = models[MODEL_NAME](1,1).to(DEVICE)

        # Create an optimizer object
        optimizer = optimizers[OPTIMIZER](model.parameters(), lr=LEARNING_RATE)

        # Create a criterion object
        criterion = criterion[CRITERION]()

        logger.info("-" * 50)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Starting training with model {MODEL_NAME} that has {num_params} parameters")
        
        # Iterate over training and test
        for epoch in range(EPOCHS):
            logger.info(f"--- Epoch: {epoch} ---")
            train.train(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                epoch=epoch,
                epochs=EPOCHS,
            )

        if WANDB:
            wandb.finish()

        # Save the model pth and the arquitecture
        logger.info("Saving the model and its architecture")
        model.to("cpu")
        torch.save(
            model.state_dict(), f"{MODELS_PATH}/{MODEL_NAME}_{datetime.now().strftime('%Y%m%d%H%M%S')}.pth"
        )

        file_path = f"{MODELS_PATH}/{MODEL_NAME}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}_arch.txt"
        with open(file_path, "w") as f:
            f.write(str(model))
        logger.info(f"Model saved in {MODELS_PATH}")
        logger.info("-" * 50)