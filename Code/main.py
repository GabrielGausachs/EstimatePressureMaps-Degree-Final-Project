from Utils import (
    logger,
    dataloader,
    train,
    savemodel,
    evaluation,
    losses,
    val,
    metrics,
)

import wandb
import datetime
import torch
from matplotlib import pyplot as plt
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
    EVALUATION,
    EXPERTYPE,
    LAMBDA_VALUE,
    USE_PHYSICAL_DATA,
    BATCH_SIZE_TEST,
    BATCH_SIZE_TRAIN,
    MAX_FEATURE,
    PLOSS,
    WEIGHTSLOSSES,
)

from Models import (
    UNet,
    Simple_net,
    UNet_phy
)

# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Models
models = {"Simple_net": Simple_net.Simple_net,
          "UNet": UNet.UNET, "UNet_phy": UNet_phy.UNET_phy}

# Optimizers
optimizers = {
    "Adam": torch.optim.Adam,
}

# Criterion
criterion = {
    "MSELoss": torch.nn.MSELoss(),
    "UVLoss": losses.UVLoss(LAMBDA_VALUE),
    "HVLoss": losses.HVLoss()
}

ploss = {True: losses.PhyLoss(),
         False: None}

# Metrics
metrics = [
    torch.nn.MSELoss(),
    metrics.PerCS()
]


def force_cudnn_initialization():
    s = 32
    dev = torch.device('cuda')
    torch.nn.functional.conv2d(torch.zeros(
        s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))


if __name__ == "__main__":

    force_cudnn_initialization()

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
			"when": datetime.datetime.now().strftime('%Y%m%d%H%M%S'),
			"Max feature in model": MAX_FEATURE
                },
                save_code=False,
            )
            logger.info("-" * 50)
            logger.info("Wandb correctly initialized")

        # Create a model
        if USE_PHYSICAL_DATA:
            model = models[MODEL_NAME](1, 11, 1).to(DEVICE)
        else:
            model = models[MODEL_NAME](1, 1).to(DEVICE)

        # Create an optimizer object
        optimizer = optimizers[OPTIMIZER](model.parameters(), lr=LEARNING_RATE)

        # Create a criterion object
        criterion = criterion[CRITERION]

        logger.info("-" * 50)
        num_params = sum(p.numel()
                         for p in model.parameters() if p.requires_grad)
        logger.info(
            f"Starting training with model {MODEL_NAME} that has {num_params} parameters")
        logger.info(f"Learning rate: {LEARNING_RATE}")
        logger.info(f"Lambda value: {LAMBDA_VALUE}")

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
                epochs=EPOCHS,
                ploss=ploss[PLOSS],
                weightsloss=WEIGHTSLOSSES,
            )
            epoch_loss_val, epoch_metric_val = val.val(
                model=model,
                loader=val_loader,
                metrics=metrics,
                criterion=criterion,
                epoch=epoch,
                epochs=EPOCHS,
                ploss=ploss[PLOSS],
                weightsloss=WEIGHTSLOSSES,
            )

            if WANDB:
                #wandb.log({"epoch": epoch, "train_loss": epoch_loss})
                wandb.log({'train_loss': epoch_loss_train}, step=epoch)
                wandb.log({'val_loss': epoch_loss_val}, step=epoch)

                for i, metric in enumerate(metrics):
                    wandb.log(
                        {f'Val {metric}': epoch_metric_val[i]}, step=epoch)
                    wandb.log(
                        {f'Train {metric}': epoch_metric_train[i]}, step=epoch)

            print(f"--- Epoch: {epoch} finished ---")

        # Save the model pth and the arquitecture
        savemodel.save_model(model)

    logger.info("-" * 50)

    

    if EVALUATION:
        if DO_TRAIN:
            # The train has just been done and we want to evaluate
            logger.info("The train is done and is starting the evaluation")
            evaluation.evaluation(model, criterion[CRITERION], val_loader)
        else:
            # The train is not done and we want to evaluate another model
            logger.info("Starting evaluation of a past model")
            model = models[MODEL_NAME](1, 1).to(DEVICE)
            evaluation.evaluation(model, criterion[CRITERION], val_loader)
        logger.info("Evaluation Completed!")
        logger.info("-" * 50)
