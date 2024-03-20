from PIL import Image
from torch.utils.data import Dataset
from Utils.logger import initialize_logger,get_logger
import cv2
import torchvision.transforms as transforms

logger = get_logger()


class CustomDataset(Dataset): 
    # Dataset for the random option
    # Includes:
    # - IR images
    # - PR images
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

        input_image = self.load_image(input_path)
        output_image = self.load_image(output_path)

        if self.transform:
            x = 28
            y = 7
            ancho = 71
            altura = 142
            imagen_recortada = input_image[y:y+altura, x:x+ancho]
            imagen_escala = cv2.resize(imagen_recortada, (84, 192))
            input_img = self.transform(imagen_escala)
            transform_output_image = transforms.Compose([
            transforms.ToTensor()])
            output_image = transform_output_image(output_image)

        if self.p_data != None:
            p_vector = self.p_data[index]
            return input_img, p_vector, output_image
        else:
            return input_img, output_image

    def load_image(self, path):
        # Load the image
        image = cv2.imread(path)
        return image
    

class CustomDataset2(Dataset):
    # Dataset for the no-random option
    # Includes:
    # - IR images
    # - PR images
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

        input_image = self.load_image(input_path)
        output_image = self.load_image(output_path)

        if self.transform:
            x = 28
            y = 7
            ancho = 71
            altura = 142

            # Recortar la región de interés (ROI)
            imagen_recortada = input_image[y:y+altura, x:x+ancho]
            imagen_escala = cv2.resize(imagen_recortada, (84, 192))
            input_img = self.transform(imagen_escala)
            transform_output_image = transforms.Compose([
            transforms.ToTensor()])
            output_image = transform_output_image(output_image)

        if self.p_data != None:
            p_vector = self.p_data[index]
            return input_img, p_vector, output_image
        else:
            return input_img, output_image

    def load_image(self, path):
        # Load the image
        image = cv2.imread(path)
        return image