
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

    # logger.info loader info
    logger.info(f"Loader length: {len(loader)}")
    logger.info(f"Loader batch size: {loader.batch_size}")
    logger.info(f"Loader drop last: {loader.drop_last}")
    logger.info(f"Loader num workers: {loader.num_workers}")

    torch.cuda.empty_cache()  # Limpiar la caché de CUDA si se utiliza GPU
    gc.collect()  # Recolectar la basura para liberar memoria no utilizada
    logger.info("Memory cleaned!")

    for batch_idx, batch_features in enumerate(loader, 1):
        logger.info(f"Epoch: {epoch}/{epochs}, Processing batch {batch_idx}/{len(loader)}...")

        batch_features = batch_features.to(DEVICE)

        assert tuple(batch_features.shape[1:]) == model.input_size, "Input size mismatch"

        optimizer.zero_grad()
        outputs = model(batch_features.to(DEVICE))
        train_loss = criterion(outputs, batch_features.to(DEVICE))
        train_loss.backward()
        optimizer.step()

        total_loss += train_loss.item()

        # Liberar memoria en cada iteración
        del batch_features
        del outputs
        del train_loss
        torch.cuda.empty_cache()  # Limpiar la caché de CUDA si se utiliza GPU
        gc.collect()  # Recolectar la basura para liberar memoria no utilizada

    epoch_loss = total_loss / len(loader)
    #result.add_loss("train", epoch_loss)

    logger.info(f"Epoch: {epoch}/{epochs}, Train loss = {epoch_loss:.6f}")
    if WANDB:
        wandb.log({"epoch": epoch, "train_loss": epoch_loss})

    torch.cuda.empty_cache()  # Limpiar la caché de CUDA si se utiliza GPU
    gc.collect()  # Recolectar la basura para liberar memoria no utilizada
    logger.info("Train finished! Memory cleaned!")
