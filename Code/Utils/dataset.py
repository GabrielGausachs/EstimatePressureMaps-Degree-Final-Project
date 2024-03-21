
from torch.utils.data import Dataset
from Utils.logger import initialize_logger,get_logger
import numpy as np


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

        if self.transform:
            input_array = self.transform['input'](input_array)
            output_array = self.transform['output'](output_array)

        if self.p_data != None:
            p_vector = self.p_data[index]
            return input_array, p_vector, output_array
        else:
            return input_array, output_array

    def load_array(self, path):
        # Load the array
        array = np.load(path)
        return array