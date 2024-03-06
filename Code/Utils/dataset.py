from PIL import Image
from torch.utils.data import Dataset
from logger import initialize_logger,get_logger

logger = get_logger()


class CustomDataset(Dataset):
    def __init__(self, dic_ir, dic_pm, transform=None):

        self.ir_paths = dic_ir.values()
        self.pm_paths = dic_pm.values()
        self.transform = transform

    def __len__(self):
        return len(self.ir_paths)

    def __getitem__(self, index):

        input_path = self.ir_paths[index]
        output_path = self.pm_paths[index]

        input_image = self.load_image(input_path)
        output_image = self.load_image(output_path)

        if self.transform:
            input_image = self.transform(input_image)
            output_image = self.transform(output_image)

        return input_image, output_image

    def load_image(self, path):
        # Load the image
        image = Image.open(path)
        return image
    

class CustomDataset2(Dataset):
    def __init__(self, ir_paths, pm_paths, transform=None):

        self.ir_paths = ir_paths
        self.pm_paths = pm_paths
        self.transform = transform

    def __len__(self):
        return len(self.ir_paths)

    def __getitem__(self, index):

        input_path = self.ir_paths[index]
        output_path = self.pm_paths[index]

        input_image = self.load_image(input_path)
        output_image = self.load_image(output_path)

        if self.transform:
            input_image = self.transform(input_image)
            output_image = self.transform(output_image)

        return input_image, output_image

    def load_image(self, path):
        # Load the image
        image = Image.open(path)
        return image