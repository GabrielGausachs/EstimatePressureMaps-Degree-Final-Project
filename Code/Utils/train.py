
from Utils.logger import initialize_logger, get_logger
import torch
import wandb
import gc

from Utils.config import (
    DEVICE,
    USE_PHYSICAL_DATA,
)

logger = get_logger()


def train(model, loader, optimizer, criterion, metrics, epoch=0, epochs=0):
    total_loss = 0
    total_metric = [0, 0, 0, 0]
    model.train()

    logger.info(f"Epoch: {epoch}/{epochs}, Starting training...")

    # Logger info
    logger.info(f"Loader length: {len(loader)}")
    logger.info(f"Loader batch size: {loader.batch_size}")
    logger.info(f"Loader drop last: {loader.drop_last}")
    logger.info(f"Loader num workers: {loader.num_workers}")
    logger.info(f"Criterion: {criterion}")
    logger.info(f"Physical loss: {ploss}")

    torch.cuda.empty_cache()  # Clean CUDA Cache if used GPU
    gc.collect()  # Collect trash to free memory not used
    logger.info("Memory cleaned!")

    if not USE_PHYSICAL_DATA:

        for batch_idx, (input_images, target_images) in enumerate(loader, 1):
            input_images = input_images.to(DEVICE)
            target_images = target_images.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(input_images)
            train_loss = criterion(outputs, target_images)

            for i, metric in enumerate(metrics):

                metric_loss = metric(outputs, target_images)

                total_metric[i] += metric_loss

            train_loss.backward()
            optimizer.step()

            total_loss += train_loss.item()
            # print('loss:',train_loss.item())

            # Free memory in each iteration
            del input_images
            del target_images
            del train_loss
            torch.cuda.empty_cache()  # Clean CUDA Cache if used GPU
            gc.collect()  # Collect trash to free memory not used
    
    else:

        for batch_idx, (input_images, target_images, tensor_data) in enumerate(loader, 1):
            input_images = input_images.to(DEVICE)
            target_images = target_images.to(DEVICE)
            tensor_data = tensor_data.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(input_images, tensor_data)
            train_loss = criterion(outputs, target_images)

            for i, metric in enumerate(metrics):

                metric_loss = metric(outputs, target_images)

                total_metric[i] += metric_loss

            train_loss.backward()
            optimizer.step()

            total_loss += train_loss.item()

            # Free memory in each iteration
            del input_images
            del target_images
            del tensor_data
            del train_loss
            torch.cuda.empty_cache()  # Clean CUDA Cache if used GPU
            gc.collect()  # Collect trash to free memory not used


    epoch_loss = total_loss / len(loader)
    print(epoch_loss)

    epoch_metric = [total_metric[0] /
                    len(loader), total_metric[1] / len(loader),
                    total_metric[2] / len(loader), total_metric[3] / len(loader)]

    logger.info(f"Epoch: {epoch}/{epochs}, Train loss = {epoch_loss:.6f}")

    for i, metric in enumerate(metrics):
        logger.info(
            f"Epoch: {epoch}/{epochs}, Train {metric} = {epoch_metric[i]:.6f}")

    torch.cuda.empty_cache()  # Clean CUDA Cache if used GPU
    gc.collect()  # Collect trash to free memory not used
    logger.info("Train finished! Memory cleaned!")
    logger.info("-" * 50)

    return epoch_loss, epoch_metric