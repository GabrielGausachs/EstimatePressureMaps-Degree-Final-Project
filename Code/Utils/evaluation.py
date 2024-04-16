from Utils.logger import initialize_logger,get_logger
import torch
from matplotlib import pyplot as plt 
import os

from Utils.config import (
    DEVICE,
    LAST_MODEL_PATH,
    IMG_PATH,
    DO_TRAIN,
)

logger = get_logger()



def evaluation(model,criterion,val_loader):
    if not DO_TRAIN:
        model.load_state_dict(torch.load(LAST_MODEL_PATH, map_location=torch.device('cpu')))
        model.to(DEVICE)

    model.eval()

    with torch.no_grad():

        for input_img, target_img in val_loader:
            break

        input_img = input_img.to(DEVICE)
        target_img = target_img.to(DEVICE)

        output_img = model(input_img)
        loss = criterion(output_img[0], target_img[0])

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        axes[0].imshow(target_img[0].squeeze().cpu().numpy())
        axes[0].set_title('Target Image')

        axes[1].imshow(output_img[0].squeeze().cpu().numpy())
        axes[1].set_title('Output Image')

        fig.suptitle('Comparison of Target and Output Images', fontsize=12)
        fig.text(0.5, 0.01, f'Loss: {loss.item():.4f}', ha='center')
        plt.savefig(os.path.join(IMG_PATH,'Comparing_output_HV_2.png'))
        plt.show()
