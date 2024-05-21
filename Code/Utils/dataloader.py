import os
import random
import numpy as np
import cv2
import json
import pandas as pd
from matplotlib import pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from Utils.logger import initialize_logger, get_logger
from Utils.dataset import CustomDataset
from torchvision.transforms.functional import crop
from scipy import signal
import datetime
from sklearn.preprocessing import StandardScaler

from Utils.config import (
    BATCH_SIZE_TEST,
    BATCH_SIZE_TRAIN,
    LOCAL_SLP_DATASET_PATH,
    SERVER_SLP_DATASET_PATH,
    NUM_WORKERS,
    USE_PHYSICAL_DATA,
    IMG_PATH,
    PATH_DATASET,
    PARTITION,
)

# initialize_logger()

logger = get_logger()


class CustomDataloader:
    def __init__(self):

        self.test_batch_size = BATCH_SIZE_TEST
        self.train_batch_size = BATCH_SIZE_TRAIN
        self.num_workers = NUM_WORKERS
        self.path_arrays = PATH_DATASET

    def prepare_dataloaders(self):
        """Prepare dataloaders for training and testing"""

        logger.info("-" * 50)
        logger.info(f'Reading the data from {self.path_arrays}...')

        # Get the data

        # 2 dictionaries for IR numpys & PM numpys
        # Each one has a diccionary for each patient
        # Each patient diccionary has a diccionary where:
        # - keys: the different category (cover1,cover2,uncover)
        # - values: paths to the images/numpys

        dic_ir_numpy = {}
        dic_pm_numpy = {}

        if self.path_arrays == 'Local':
            path_arrays = LOCAL_SLP_DATASET_PATH
        else:
            path_arrays = SERVER_SLP_DATASET_PATH

        for patient in os.listdir(path_arrays):
            patient_path = os.path.join(path_arrays, patient)
            if os.path.isdir(patient_path):

                dic_ir_numpy[patient] = {}
                dic_pm_numpy[patient] = {}

                dics = [dic_ir_numpy, dic_pm_numpy]

                ir_np_path = os.path.join(patient_path, 'IRraw')
                pm_np_path = os.path.join(patient_path, 'PMarray')

                dir_paths = [ir_np_path, pm_np_path]

                for path, dic in zip(dir_paths, dics):
                    if os.path.exists(path):
                        for category in os.listdir(path):
                            category_path = os.path.join(path, category)
                            if os.path.isdir(category_path):
                                dic[patient][category] = []
                                for file in os.listdir(category_path):
                                    if file.endswith(('.png', '.npy')):
                                        dic[patient][category].append(
                                            os.path.join(category_path, file))
                                    else:
                                        print(patient)
                                        print(category)
                    else:
                        raise FileNotFoundError('Path not found')

        logger.info(f'Number of pacients: {len(dic_ir_numpy)}')
        logger.info(
            f'Number of categories in a patient: {len(dic_ir_numpy["00001"])}')

        p_data = pd.read_csv(os.path.join(path_arrays, 'physiqueData.csv'))
        #p_data['gender'] = p_data['gender'].str.strip()
        #p_data = pd.get_dummies(p_data, columns=['gender'])
        p_data = p_data.drop('sub_idx', axis=1)
        p_data = p_data.drop('gender',axis = 1)
        #weights = p_data.iloc[:,1]

        #scaler = StandardScaler()

        # Fit the scaler to the data and transform the data
        #p_data_scaled = scaler.fit_transform(p_data)
        #p_data = pd.DataFrame(p_data_scaled, columns=p_data.columns)

        #p_data['gender_male'] = p_data['gender_male'].astype(int)
        #p_data['gender_female'] = p_data['gender_female'].astype(int)

        logger.info(f'Size of the physical dataset: {p_data.shape}')

        # Create dataset
        logger.info("-" * 50)
        logger.info('Creating dataset...')

        train_arrays = {'ir': [], 'pm': []}
        val_arrays = {'ir': [], 'pm': []}
        test_arrays = {'pm': [], 'ir': []}

        # Patient Partition
        if PARTITION == 1:

            logger.info('Partition --> Patients')

            for key in dic_pm_numpy.keys():
                for category in dic_pm_numpy[key].keys():
                    indexes = list(range(len(dic_pm_numpy[key][category])))
                    random.shuffle(indexes)
                    train_split_index = int(0.75 * len(indexes))
                    val_split_index = int(0.95 * len(indexes))

                    train_arrays['pm'].extend(
                        dic_pm_numpy[key][category][:train_split_index])
                    train_arrays['ir'].extend(
                        dic_ir_numpy[key][category][:train_split_index])

                    val_arrays['pm'].extend(
                        dic_pm_numpy[key][category][train_split_index:val_split_index])
                    val_arrays['ir'].extend(
                        dic_ir_numpy[key][category][train_split_index:val_split_index])

                    test_arrays['pm'].extend(
                        dic_pm_numpy[key][category][val_split_index:])
                    test_arrays['ir'].extend(
                        dic_ir_numpy[key][category][val_split_index:])

        with open(f"Models/TestJson/test_paths_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.json", "w+") as outfile:
            json.dump(test_arrays, outfile)

        # Data transformation if needed
        transform = {
            'input': transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(crop_array),
                transforms.Resize((192, 84)),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]),
            'output': transforms.Compose([transforms.ToTensor()])}

        train_dataset = CustomDataset(
            train_arrays['ir'], train_arrays['pm'], p_data, transform=transform)

        val_dataset = CustomDataset(
            val_arrays['ir'], val_arrays['pm'], p_data, transform=transform)

        logger.info(f"Train size: {len(train_dataset)}")
        logger.info(f"Val size: {len(val_dataset)}")

        logger.info("-" * 50)
        logger.info('Creating dataloaders...')

        # Create  dataloaders
        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE_TRAIN, shuffle=True, num_workers=0, drop_last=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=BATCH_SIZE_TEST, shuffle=False, num_workers=0, drop_last=True
        )

        train_dataset_info = {
            'Number of samples': len(train_loader.dataset),
            'Batch size': train_loader.batch_size,
            'Number of batches': len(train_loader)
        }

        val_dataset_info = {
            'Number of samples': len(val_loader.dataset),
            'Batch size': val_loader.batch_size,
            'Number of batches': len(val_loader)
        }

        logger.info(f"Train loader info: {train_dataset_info}")
        logger.info(
            f"Array input size of the train loader: {next(iter(train_loader))[0].shape}")
        logger.info(
            f"Array output size of the train loader: {next(iter(train_loader))[1].shape}")
        if USE_PHYSICAL_DATA:
            logger.info(
                f"Size of the data of train loader:{next(iter(train_loader))[2].shape}")
        logger.info(f"Val loader info: {val_dataset_info}")
        logger.info(
            f"Array input size of the val loader: {next(iter(val_loader))[0].shape}")
        logger.info(
            f"Array output size of the val loader: {next(iter(val_loader))[1].shape}")
        if USE_PHYSICAL_DATA:
            logger.info(
                f"Size of the data of the val loader:{next(iter(val_loader))[2].shape}")

        # Function to check how are the arrays that we pass to the model
        check_transform(val_loader,self.path_arrays)

        return train_loader, val_loader


# ----------------------------------- EXTRA FUNCTIONS -----------------------------------

def crop_array(array):
    return crop(array, 7, 29, 140, 66)


def check_transform(val_loader, path_arrays):
    for i in range(1):
        batch = next(iter(val_loader))

        input_img = batch[0][0].squeeze().cpu().numpy()
        target_img = batch[1][0].squeeze().cpu().numpy()

        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(input_img)
        plt.title('Input image')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(target_img)
        plt.title('Target image')
        plt.axis('off')

        #plt.savefig(os.path.join(
        #    IMG_PATH, f'compare_transforms_{path_arrays}.png'))

        plt.show()
