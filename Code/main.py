from Utils import (
    logger, 
    dataloader,
    train,
    savemodel,
    evaluation,
    losses,
    test,
)

import wandb
import datetime
import torch
from matplotlib import pyplot as plt 
import cv2
import os

from Utils.config import (
    WANDB,
    EXECUTION_NAME,
    MODEL_NAME,
    OPTIMIZER,
    CRITERION,
    LEARNING_RATE,
    EPOCHS,
    DEVICE,
    DO_TRAIN,
    DO_TEST,
    EVALUATION,
    LAST_RUN_WANDB,
    )

from Models import (
    UNet,
    Simple_net
)

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Models
models = {"UNet": UNet.UNet, "Simple_net": Simple_net.Simple_net}

# Optimizers
optimizers = {
    "Adam": torch.optim.Adam,
}

# Criterion
criterion = {
    "MSELoss": torch.nn.MSELoss(reduction = 'mean'),
    "PWRSWtL": losses.PWRSWtL(1.0)
}


if __name__ == "__main__":
    # Initialize logger
    logger.initialize_logger()
    logger = logger.get_logger()

    train_loader, val_loader = dataloader.CustomDataloader().prepare_dataloaders()

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
                },
                save_code=False
            )
            logger.info("-" * 50)
            logger.info("Wandb correctly initialized")


        # Create a model
        model = models[MODEL_NAME](1,1).to(DEVICE)

        # Create an optimizer object
        optimizer = optimizers[OPTIMIZER](model.parameters(), lr=LEARNING_RATE)

        # Create a criterion object
        criterion = criterion[CRITERION]

        logger.info("-" * 50)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Starting training with model {MODEL_NAME} that has {num_params} parameters")
        
        # Iterate over training and test
        for epoch in range(EPOCHS):
            logger.info(f"--- Epoch: {epoch} ---")
            epoch_loss = train.train(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                epoch=epoch,
                epochs=EPOCHS,
            )

            if WANDB:
                #wandb.log({"epoch": epoch, "train_loss": epoch_loss})
                wandb.log({'train_loss': epoch_loss}, step=epoch)

        # Save the model pth and the arquitecture
        savemodel.save_model(model)

    logger.info("-" * 50)

    if DO_TEST:
        if DO_TRAIN:

            logger.info("Starting testing")

            for epoch in range(EPOCHS):
                logger.info(f"--- Epoch: {epoch} ---")
                epoch_loss = test.test(
                    model=model,
                    loader=val_loader,
                    criterion=criterion,
                    epoch=epoch,
                    epochs=EPOCHS,
                )

                if WANDB:
                    #wandb.log({"epoch": epoch, "train_loss": epoch_loss})
                    wandb.log({'test_loss': epoch_loss}, step=epoch)
                
        else:

            logger.info("Starting testing of a past model")
            model = models[MODEL_NAME](1,1).to(DEVICE)

            if WANDB:
                wandb.login()
                run = wandb.init(
                    project="TFG",
                    entity='1604373',
                    name=LAST_RUN_WANDB,
                    save_code=False,
                    resume='allow'  # Resume existing run if found
                )

            for epoch in range(EPOCHS):
                logger.info(f"--- Epoch: {epoch} ---")
                test.test(
                    model=model,
                    loader=val_loader,
                    criterion=criterion,
                    epoch=epoch,
                    epochs=EPOCHS,
                )
            
                if WANDB:
                    wandb.log({'test_loss': epoch_loss}, step=epoch)

        logger.info("Testing Completed!")
        logger.info("-" * 50)

    if EVALUATION:
        if DO_TRAIN:
            # The train has just been done and we want to evaluate
            logger.info("The train is done and is starting the evaluation")
            evaluation.evaluation(model,criterion,val_loader)
        else:
            # The train is not done and we want to evaluate another model
            logger.info("Starting evaluation of a past model")
            model = models[MODEL_NAME](1,1).to(DEVICE)
            evaluation.evaluation(model,criterion[CRITERION](),val_loader)
        logger.info("Evaluation Completed!")
        logger.info("-" * 50)
