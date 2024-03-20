from PIL import Image
from torch.utils.data import Dataset
from Utils.logger import initialize_logger,get_logger
import cv2
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

logger = get_logger()
class CustomDataset(Dataset): 
    # Dataset for the random option
    # Includes:
    # - IR arrays
    # - PR arrays
    # Physical data (if provided)
    def __init__(self, ir_paths, pm_paths, p_data, transform=None):

        self.ir_paths = ir_paths
        self.pm_paths = pm_paths
        
        if p_data is not None:
            self.p_data = p_data.iloc[:, 1:]
        else:
            self.p_data = None
    
        self.transform = transform

    def __len__(self):
        return len(self.ir_paths)

    def __getitem__(self, index):

        input_path = self.ir_paths[index]
        output_path = self.pm_paths[index]

        input_array = self.load_array(input_path)
        output_array = self.load_array(output_path)
        input_array = input_array.astype(np.float32)
        output_array = output_array.astype(np.float32)
        input_image = Image.fromarray(input_array)
        output_image = Image.fromarray(output_array)

        if self.transform:
            #x = 28
            #y = 7
            #ancho = 71
            #altura = 142
            #imagen_recortada = input_image[y:y+altura, x:x+ancho]
            #imagen_escala = cv2.resize(imagen_recortada, (84, 192))
            input_image = self.transform(input_image)
            output_image = self.transform(output_image)

        if self.p_data != None:
            p_vector = self.p_data[index]
            return input_array, p_vector, output_array
        else:
            return input_image, output_image

    def load_array(self, path):
        # Load the array
        array = np.load(path)
        return array