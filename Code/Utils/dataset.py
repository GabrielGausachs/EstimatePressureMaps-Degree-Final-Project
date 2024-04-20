from torch.utils.data import Dataset
from Utils.logger import initialize_logger, get_logger
import numpy as np
import torch
from scipy import signal
from Utils.config import (
    USE_PHYSICAL_DATA,
)

logger = get_logger()


class CustomDataset(Dataset):
    # Dataset for the random option
    # Includes:
    # - IR arrays
    # - PR arrays
    # Physical data
    def __init__(self, ir_paths, pm_paths, p_data, transform=None):

        self.ir_paths = ir_paths
        self.pm_paths = pm_paths
        self.p_data = p_data

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

            parts = str(output_path.split("/")[-4])
            number = int(parts)
            p_vector = self.p_data.iloc[number-1]
            weight = p_vector[1]
            tensor_data = torch.tensor(p_vector.values)

            # Applying median filter
            median_array = signal.medfilt2d(output_array)
            max_array = np.maximum(output_array, median_array)

            area_m = 1.03226 / 10000
            ideal_pressure = weight * 9.81 / (area_m * 1000)

            output_array = (max_array / np.sum(max_array)) * ideal_pressure
            output_array = self.transform['output'](output_array)

            if USE_PHYSICAL_DATA:
                return input_array, output_array, tensor_data
            else:
                return input_array, output_array

    def load_array(self, path):
        # Load the array
        array = np.load(path)
        return array
