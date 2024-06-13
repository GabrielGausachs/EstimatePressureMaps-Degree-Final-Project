import wandb
import datetime
import torch
from matplotlib import pyplot as plt
import os
import math
import copy
import torch.optim.lr_scheduler as lr_scheduler

from Utils import (
    logger,
    dataloader,
    train,
    savemodel,
    losses,
    val,
    metrics
    )

from Models import (
    UNet,
    UNet_phy
    )


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
    EXPERTYPE,
    USE_PHYSICAL_DATA,
    BATCH_SIZE_TEST,
    BATCH_SIZE_TRAIN
    )

# Models
models = {"UNet": UNet.UNET, "UNet_phy": UNet_phy.UNET_phy}

# Optimizers
optimizers = {
    "Adam": torch.optim.Adam
    }

# Criterion
criterion = {
    "MSELoss": torch.nn.MSELoss(),
    "HVLoss": losses.HVLoss(),
    "PLoss": losses.PhyLoss(),
    "SSIMLoss": losses.SSIMLoss()
    }

# Metrics
metrics = [
    torch.nn.MSELoss(),
    metrics.PerCS(),
    metrics.MSEeff(),
    metrics.SSIMMetric()
    ]


if __name__ == "__main__":

    # Initialize logger
    logger.initialize_logger()
    logger = logger.get_logger()

    train_loader, val_loader = dataloader.CustomDataloader().prepare_dataloaders()
    
    features= [32,64,128,256]

    if DO_TRAIN:
        # Initialize wandb
        if WANDB:
            wandb.login()
            wandb.init(
                project="TFG",
                entity='1604373',
                name=f"{EXECUTION_NAME}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}",
                config={
                        "model_name": MODEL_NAME,
                        "optimizer": OPTIMIZER,
                        "criterion": CRITERION,
                        "learning_rate": LEARNING_RATE,
                        "epochs": EPOCHS,
                        "experiment_type": EXPERTYPE,
                        "batch_train_size": BATCH_SIZE_TRAIN,
                        "batch_test_size": BATCH_SIZE_TEST,
            "features":features,
			"when": datetime.datetime.now().strftime('%Y%m%d%H%M%S'),
                },
                save_code=False,
            )
            logger.info("-" * 50)
            logger.info("Wandb correctly initialized")

        # Create a model

        if USE_PHYSICAL_DATA:
            model = models[MODEL_NAME](1, 9, 1,features).to(DEVICE)
        else:
            model = models[MODEL_NAME](1, 1, features).to(DEVICE)

        # Create an optimizer object
        optimizer = optimizers[OPTIMIZER](model.parameters(), lr=LEARNING_RATE)
        #scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=70)

        # Create a criterion object
        criterion = criterion[CRITERION]

        logger.info("-" * 50)
        num_params = sum(p.numel()
                         for p in model.parameters() if p.requires_grad)
        logger.info(
            f"Starting training with model {MODEL_NAME} that has {num_params} parameters")
        logger.info(f"Learning rate: {LEARNING_RATE}")

	    # Initialize Variables for EarlyStopping
        best_loss = math.inf
        best_model_weights = None
        patience = 10

        # Iterate over training and test
        for epoch in range(EPOCHS):
            logger.info(f"--- Epoch: {epoch} ---")
            epoch_loss_train, epoch_metric_train = train.train(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                metrics=metrics,
                epoch=epoch,
                epochs=EPOCHS
            )
            epoch_loss_val, epoch_metric_val = val.val(
                model=model,
                loader=val_loader,
                metrics=metrics,
                criterion=criterion,
                epoch=epoch,
                epochs=EPOCHS,
            )
            logger.info(f"Learning rate: {optimizer.param_groups[0]['lr']}")
            #scheduler.step()

            if WANDB:
                # wandb.log({"epoch": epoch, "train_loss": epoch_loss})
                wandb.log({'train_loss': epoch_loss_train}, step=epoch)
                wandb.log({'val_loss': epoch_loss_val}, step=epoch)

                for i, metric in enumerate(metrics):
                    wandb.log(
                        {f'Val {metric}': epoch_metric_val[i]}, step=epoch)
                    wandb.log(
                        {f'Train {metric}': epoch_metric_train[i]}, step=epoch)

            print(f"--- Epoch: {epoch} finished ---")

            if CRITERION == 'SSIMLoss':
                min_improvement = 0.001
            else:
                min_improvement = 0.000002

            if epoch_loss_val < best_loss:
                improvement = best_loss - epoch_loss_val
                if improvement >= min_improvement:
                    best_loss = epoch_loss_val
                    best_model_weights = copy.deepcopy(
                        model.state_dict())  # Deep copy here
                    patience = 10  # Reset patience counter
                else:
                    patience -= 1
                    if patience == 0:
                        logger.info("--- Early stopping activated ---")
                        logger.info(f"Best loss: {best_loss}")
                        break
            else:
                patience -= 1
                if patience == 0:
                    logger.info("--- Early stopping activated ---")
                    logger.info(f"Best loss: {best_loss}")
                    break

        # Save the model pth and the arquitecture
	    # Load the best model weights
        model.load_state_dict(best_model_weights)
        savemodel.save_model(model)

    logger.info("-" * 50)
