import os
import random

import numpy as np
import cv2
import pandas as pd
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import (
    DataLoader,
    random_split,
)

from sklearn.preprocessing import OneHotEncoder
from Utils.logger import initialize_logger,get_logger
from Utils.dataset import CustomDataset, CustomDataset2

from Utils.config import (
    BATCH_SIZE_TEST,
    BATCH_SIZE_TRAIN,
    DATASET,
    DEVICE,
    LOCAL_SLP_DATASET_PATH,
    SERVER_SLP_DATASET_PATH,
    NUM_WORKERS,
    SHOW_IMAGES,
    USE_PHYSICAL_DATA,
)

#initialize_logger()

logger = get_logger()

class CustomDataloader:
    def __init__(self):

        self.test_batch_size = BATCH_SIZE_TEST
        self.train_batch_size = BATCH_SIZE_TRAIN
        self.num_workers = NUM_WORKERS
        self.local_slp = LOCAL_SLP_DATASET_PATH
        self.server_slp = SERVER_SLP_DATASET_PATH

    def prepare_dataloaders(self, is_random):
        """Prepare dataloaders for training and testing"""

        # Data transformation if needed
        transform = transforms.Compose([transforms.Resize((120,160)),
                                        transforms.ToTensor()])

        logger.info("-" * 50)
        logger.info('Read the data')

        # Get the data

        # 4 dictionaries for IR Images, IR numpys, PM Imatges & PM numpys
        # Each one has a diccionary for each patient
        # Each patient diccionary has a diccionary where:
        # - keys: the different category (cover1,cover2,uncover)
        # - values: paths to the images/numpys 

        dic_ir_img = {}
        dic_ir_numpy = {}
        dic_pm_img = {}
        dic_pm_numpy = {}

        for patient in os.listdir(self.local_slp):
            patient_path = os.path.join(self.local_slp,patient)
            if os.path.isdir(patient_path):

                dic_ir_numpy[patient] = {}
                dic_ir_img[patient] = {}
                dic_pm_img[patient] = {}
                dic_pm_numpy[patient] = {}

                dics = [dic_ir_img,dic_ir_numpy,dic_pm_img,dic_pm_numpy]

                ir_path = os.path.join(patient_path,'IR')
                ir_np_path = os.path.join(patient_path,'IRraw')
                pm_path = os.path.join(patient_path,'PM')
                pm_np_path = os.path.join(patient_path,'PMarray')

                dir_paths = [ir_path,ir_np_path,pm_path,pm_np_path]

                for path,dic in zip(dir_paths,dics):
                    if os.path.exists(path):
                        for category in os.listdir(path):
                            category_path = os.path.join(path,category)
                            if os.path.isdir(category_path):
                                dic[patient][category]=[]
                                for file in os.listdir(category_path):
                                    if file.endswith(('.png','.npy')):
                                        dic[patient][category].append(os.path.join(category_path,file))
                                    else:
                                        print(patient)
                                        print(category)
                    else:
                        raise FileNotFoundError ('Path not found')
                    
        logger.info(f'Number of pacients: {len(dic_ir_numpy)}')
        logger.info(f'Number of categories in a patient: {len(dic_ir_img["00001"])}')
        
        if SHOW_IMAGES: # Show the IR, PM image and PM array of a uncover random patient

            random_patient = random.choice(list(dic_ir_img.keys()))

            patient_ir_img = dic_ir_img[random_patient]['cover1'][0]
            ir_img = cv2.imread(patient_ir_img)
            cv2.imshow("IR Image", ir_img)
            cv2.waitKey(0)  # Wait for a key press to close the window
            cv2.destroyAllWindows()
            logger.info(f'Shape of the IR Image: {ir_img.shape}')

            # PM IMAGE
            patient_pm_img = dic_pm_img[random_patient]['cover1'][0]
            pm_img = cv2.imread(patient_pm_img)
            cv2.imshow("PM Image", pm_img)
            cv2.waitKey(0)  # Wait for a key press to close the window
            cv2.destroyAllWindows()
            logger.info(f'Shape of the PM Image: {pm_img.shape}')

            # PM NUMPY to Image
            #patient_pm_np = dic_pm_numpy[random_patient]['cover1'][0]
            #img = Image.fromarray(np.load(patient_pm_np).astype('uint8'))
            #img.show()

        # Read the phyisical data
            
        p_data = None

        if USE_PHYSICAL_DATA:

            p_data = pd.read_csv(os.path.join(self.local_slp, 'physiqueData.csv'))
            p_data['gender'] = p_data['gender'].str.strip()

            p_data = pd.get_dummies(p_data, columns=['gender'])

            logger.info(f'Size of the physical dataset: {p_data.size}')
        
        # Create dataset
        logger.info("-" * 50)
        logger.info(f'Create dataset with random = {is_random}')
        
        if is_random:

            # Create Dataset (we pass IR images and PM images)
            all_ir_img = []
            all_pm_img = []

            for (patient_ir, category_ir), (patient_pm, category_pm) in zip(dic_ir_img.items(), dic_pm_img.items()):
                for (category_name_ir, images_ir), (category_name_pm, images_pm) in zip(category_ir.items(), category_pm.items()):
                    all_ir_img.extend(images_ir)
                    all_pm_img.extend(images_pm)

            dataset = CustomDataset(all_ir_img, all_pm_img, p_data, transform=transform)

            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size

            # Split data into train and validation
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        else: # We try to get train and val images of each category of each patient.
            
            train_images = {'ir': [], 'pm': []}
            val_images = {'ir': [], 'pm': []}

            for (patient_ir, category_ir), (patient_pm, category_pm) in zip(dic_ir_img.items(), dic_pm_img.items()):
                assert patient_ir == patient_pm

                for (category_name_ir, images_ir), (category_name_pm, images_pm) in zip(category_ir.items(), category_pm.items()):
                    assert category_name_ir == category_name_pm 

                    indexs = list(range(len(images_ir)))
                    random.shuffle(indexs)


                    for i in indexs[:int(len(indexs) * 0.8)]:
                        train_images['ir'].append(images_ir[i])
                        train_images['pm'].append(images_pm[i])
                    if p_data is not None:
                        train_dt = p_data.iloc[indexs[:int(len(indexs) * 0.8)]]
                    else:
                        train_dt = None

                    for i in indexs[int(len(indexs)*0.8):]:
                        val_images['ir'].append(images_ir[i])
                        val_images['pm'].append(images_pm[i])
                    if p_data is not None:
                        val_dt = p_data.iloc[indexs[int(len(indexs) * 0.8):]]
                    else:
                        val_dt = None

            train_dataset = CustomDataset2(train_images['ir'], train_images['pm'], train_dt, transform=transform)

            val_dataset = CustomDataset2(val_images['ir'], val_images['pm'], val_dt, transform=transform)
        
        logger.info(f"Train size: {len(train_dataset)}")
        logger.info(f"Val size: {len(val_dataset)}")
        logger.info("-" * 50)
        logger.info('Creating dataloaders')
        
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
        logger.info(f"Image size of the train loader: {next(iter(train_loader))[0].shape}")
        logger.info(f"Val loader info: {val_dataset_info}")
        logger.info(f"Image size of the val loader: {next(iter(val_loader))[0].shape}")


        return train_loader, val_loader








                

        

#print(os.listdir(LOCAL_SLP_DATASET_PATH))
#f = CustomDataloader().prepare_dataloaders(False)
