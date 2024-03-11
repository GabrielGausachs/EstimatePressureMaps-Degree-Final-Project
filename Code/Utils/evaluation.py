from Utils.logger import initialize_logger,get_logger
import torch
from matplotlib import pyplot as plt 
import os

from Utils.config import (
    MODEL_NAME,
    DEVICE,
    EVALUATION,
    LAST_MODEL_PATH,
    IMG_PATH,
    DO_TRAIN,
)

logger = get_logger()



def evaluation(model,val_loader):
    if not DO_TRAIN:
        model.load_state_dict(torch.load(LAST_MODEL_PATH, map_location=torch.device('cpu')))
        model.to(DEVICE)

    model.eval()

    with torch.no_grad():

        for input_img, target_img in val_loader:
            break

        print(input_img.shape) 

        output_img = model(input_img)
        print(output_img.shape)

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        axes[0].imshow(target_img[0].permute(1, 2, 0))
        axes[0].set_title('Target Image')

        axes[1].imshow(output_img[0].permute(1, 2, 0))
        axes[1].set_title('Output Image')


        plt.savefig(os.path.join(IMG_PATH,'Comparing_output.png'))
        plt.show()