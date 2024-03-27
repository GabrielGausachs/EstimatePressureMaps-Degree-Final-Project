from Utils.logger import initialize_logger,get_logger
import torch
import wandb
import gc

from Utils.config import (
    DEVICE,
    WANDB,
    LAST_MODEL_PATH,
    IMG_PATH,
    DO_TRAIN,
)

logger = get_logger()

def test(model, loader, criterion, epoch=0, epochs=0):
    if not DO_TRAIN:
        model.load_state_dict(torch.load(LAST_MODEL_PATH, map_location=torch.device('cpu')))
        model.to(DEVICE)

    total_loss = 0
    model.eval()

    logger.info(f"Epoch: {epoch}/{epochs}, Starting testing...")

    # Logger info
    logger.info(f"Loader length: {len(loader)}")
    logger.info(f"Loader batch size: {loader.batch_size}")
    logger.info(f"Loader drop last: {loader.drop_last}")
    logger.info(f"Loader num workers: {loader.num_workers}")
    logger.info(f"Criterion: {criterion}")

    torch.cuda.empty_cache()  # Clean CUDA Cache if used GPU
    gc.collect()  # Collect trash to free memory not used
    logger.info("Memory cleaned!")

    with torch.no_grad():
        for batch_idx, (input_images, target_images) in enumerate(loader, 1):
            logger.info(f"Epoch: {epoch}/{epochs}, Processing batch {batch_idx}/{len(loader)}...")

            input_images = input_images.to(DEVICE)
            target_images = target_images.to(DEVICE)

            outputs = model(input_images)
            train_loss = criterion(outputs, target_images)

            total_loss += train_loss.item()

            # Free memory in each iteration
            del input_images
            del target_images
            del train_loss
            torch.cuda.empty_cache() # Clean CUDA Cache if used GPU
            gc.collect()  # Collect trash to free memory not used

    epoch_loss = total_loss / len(loader)
    #result.add_loss("train", epoch_loss)

    logger.info(f"Epoch: {epoch}/{epochs}, Test loss = {epoch_loss:.6f}")

    torch.cuda.empty_cache()  # Clean CUDA Cache if used GPU
    gc.collect()  # Collect trash to free memory not used
    logger.info("Test finished! Memory cleaned!")
    logger.info("-" * 50)

    return epoch_loss