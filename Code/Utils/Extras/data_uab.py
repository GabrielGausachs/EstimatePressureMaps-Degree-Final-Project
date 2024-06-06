import pandas as pd
import os
from glob import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from torchvision.transforms.functional import crop
import torch
import torchvision.transforms as transforms

def crop_array(array):
    
    return crop(array, 20, 28, 85, 36)

path_data = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))),'Data')

print(path_data)

transform = {
            'input': transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(crop_array),
                transforms.Resize((192, 84))])
}

max_ir = -np.inf
min_ir = np.inf
images = []
for folder in os.listdir(path_data):
    directory = os.path.join(path_data,folder)
    pattern = os.path.join(directory, '*IR.png')
    files_ir = glob(pattern)
    print(len(files_ir))
    pattern = os.path.join(directory, '*Pressio.csv')
    files_pm = glob(pattern)
    print(len(files_pm))

    for ir,pm in zip(files_ir,files_pm):
        ir_array = mpimg.imread(ir)
        array = np.rot90(ir_array, k=1, axes=(1,0))
        array_2 = np.copy(array)
        tensor_final = transform['input'](array_2)
        ir_array = tensor_final.squeeze().numpy()

        plt.imshow(ir_array)
        plt.axis('off')
        #    plt.savefig(os.path.join(IMG_PATH, f"Final_Comparing_output_model-{cover}-{title}.png"))
        plt.show()
        #pm = pd.read_csv(pm)
        #pm_array = pm.to_numpy()
        #pm_array = np.rot90(pm_array, k=1, axes=(1,0))
        #print('ir shape',ir_array.shape)
        #print('pm shape',pm_array.shape)
        #images.append(img)
            

#pattern = os.path.join(directory, '*IR.png')
#files = glob(pattern)

