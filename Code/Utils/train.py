
from Utils.logger import initialize_logger,get_logger
import torch
import wandb
import gc

from Utils.config import (
    DEVICE,
    WANDB,
)

logger = get_logger()

def train(model, loader, optimizer, criterion, epoch=0, epochs=0):
    total_loss = 0
    model.train()

    logger.info(f"Epoch: {epoch}/{epochs}, Starting training...")

    # Logger info
    logger.info(f"Loader length: {len(loader)}")
    logger.info(f"Loader batch size: {loader.batch_size}")
    logger.info(f"Loader drop last: {loader.drop_last}")
    logger.info(f"Loader num workers: {loader.num_workers}")

    torch.cuda.empty_cache()  # Clean CUDA Cache if used GPU
    gc.collect()  # Collect trash to free memory not used
    logger.info("Memory cleaned!")

    for batch_idx, (input_images, target_images) in enumerate(loader, 1):
        logger.info(f"Epoch: {epoch}/{epochs}, Processing batch {batch_idx}/{len(loader)}...")

        input_images = input_images.to(DEVICE)
        target_images = target_images.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(input_images)
        print(outputs.shape)
        print(target_images.shape)
        train_loss = criterion(outputs, target_images)
        train_loss.backward()
        optimizer.step()

        total_loss += train_loss.item()

        # Free memory in each iteration
        del input_images
        del target_images
        del train_loss
        torch.cuda.empty_cache() # Clean CUDA Cache if used GPU
        gc.collect()  # Collect trash to free memory not used

    epoch_loss = total_loss / len(loader)
    #result.add_loss("train", epoch_loss)

    logger.info(f"Epoch: {epoch}/{epochs}, Train loss = {epoch_loss:.6f}")
    if WANDB:
        #wandb.log({"epoch": epoch, "train_loss": epoch_loss})
        wandb.log({'train_loss': epoch_loss}, step=epoch)

    torch.cuda.empty_cache()  # Clean CUDA Cache if used GPU
    gc.collect()  # Collect trash to free memory not used
    logger.info("Train finished! Memory cleaned!")
    logger.info("-" * 50)
