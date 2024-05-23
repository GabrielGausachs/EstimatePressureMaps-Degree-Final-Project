from Utils.logger import initialize_logger, get_logger
import torch
import wandb
import gc

from Utils.config import (
    DEVICE,
    USE_PHYSICAL_DATA,
)

logger = get_logger()


def val(model, loader, metrics, criterion, epoch=0, epochs=0,ploss=None,weightsloss=[0,0]):
    total_metric = [0, 0, 0]
    total_loss = 0
    model.eval()

    logger.info(f"Epoch: {epoch}/{epochs}, Starting validation...")

    # Logger info
    logger.info(f"Loader length: {len(loader)}")
    logger.info(f"Loader batch size: {loader.batch_size}")
    logger.info(f"Loader drop last: {loader.drop_last}")
    logger.info(f"Loader num workers: {loader.num_workers}")
    logger.info(f"Metrics: {metrics}")
    logger.info(f"Physical loss: {ploss}")

    torch.cuda.empty_cache()  # Clean CUDA Cache if used GPU
    gc.collect()  # Collect trash to free memory not used
    logger.info("Memory cleaned!")

    with torch.no_grad():

        if not USE_PHYSICAL_DATA:
            for batch_idx, (input_images, target_images) in enumerate(loader, 1):
                input_images = input_images.to(DEVICE)
                target_images = target_images.to(DEVICE)

                outputs = model(input_images)
                for i, metric in enumerate(metrics):

                    val_metric = metric(outputs, target_images)

                    total_metric[i] += val_metric

                val_loss = criterion(outputs, target_images)

                if ploss is not None:
                    loss_physical = ploss(outputs,target_images)
                    val_loss = val_loss * weightsloss[0] + loss_physical * weightsloss[1]
                    
                total_loss += val_loss.item()

                # Free memory in each iteration
                del input_images
                del target_images
                del val_loss
                torch.cuda.empty_cache()  # Clean CUDA Cache if used GPU
                gc.collect()  # Collect trash to free memory not used

        else:
            for batch_idx, (input_images, target_images, tensor_data) in enumerate(loader, 1):
                input_images = input_images.to(DEVICE)
                target_images = target_images.to(DEVICE)
                tensor_data = tensor_data.to(DEVICE)

                outputs = model(input_images,tensor_data)
                for i, metric in enumerate(metrics):

                    val_metric = metric(outputs, target_images)

                    total_metric[i] += val_metric

                val_loss = criterion(outputs, target_images)

                if ploss is not None:
                    loss_physical = ploss(outputs,target_images)
                    val_loss = val_loss * weightsloss[0] + loss_physical * weightsloss[1]
                    
                total_loss += val_loss.item()

                # Free memory in each iteration
                del input_images
                del target_images
                del tensor_data
                del val_loss
                torch.cuda.empty_cache()  # Clean CUDA Cache if used GPU
                gc.collect()  # Collect trash to free memory not used

    epoch_metric = [total_metric[0] /
                    len(loader), total_metric[1] / len(loader),
                    total_metric[2] / len(loader)]

    epoch_loss = total_loss / len(loader)

    logger.info(f"Epoch: {epoch}/{epochs}, Val loss = {epoch_loss:.6f}")

    for i, metric in enumerate(metrics):
        logger.info(
            f"Epoch: {epoch}/{epochs}, Val {metric} = {epoch_metric[i]:.6f}")

    torch.cuda.empty_cache()  # Clean CUDA Cache if used GPU
    gc.collect()  # Collect trash to free memory not used
    logger.info("Validation finished! Memory cleaned!")
    logger.info("-" * 50)

    return epoch_loss, epoch_metric