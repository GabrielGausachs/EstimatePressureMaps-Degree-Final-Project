import os
import random

import numpy as np
import cv2
import pandas as pd
from matplotlib import pyplot as plt 
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from Utils.logger import initialize_logger,get_logger
from Utils.dataset import CustomDataset
from torchvision.transforms.functional import crop

from Utils.config import (
    BATCH_SIZE_TEST,
    BATCH_SIZE_TRAIN,
    LOCAL_SLP_DATASET_PATH,
    SERVER_SLP_DATASET_PATH,
    NUM_WORKERS,
    SHOW_IMAGES,
    USE_PHYSICAL_DATA,
    SHOW_HISTOGRAM,
    IMG_PATH,
    PATH_DATASET,
)

#initialize_logger()

logger = get_logger()

class CustomDataloader:
    def __init__(self):

        self.test_batch_size = BATCH_SIZE_TEST
        self.train_batch_size = BATCH_SIZE_TRAIN
        self.num_workers = NUM_WORKERS
        self.path_arrays = PATH_DATASET

    def prepare_dataloaders(self):
        """Prepare dataloaders for training and testing"""

        # Data transformation if needed
        # Do I normalize?
        transform = {
            'input': transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Lambda(crop_array),  
                    transforms.Resize((192, 84)),
		    transforms.Normalize(mean=[0.5], std=[0.5]),
                    ]),
            'output': transforms.Compose([transforms.ToTensor()])}


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
            patient_path = os.path.join(path_arrays,patient)
            if os.path.isdir(patient_path):

                dic_ir_numpy[patient] = {}
                dic_pm_numpy[patient] = {}

                dics = [dic_ir_numpy,dic_pm_numpy]

                ir_np_path = os.path.join(patient_path,'IRraw')
                pm_np_path = os.path.join(patient_path,'PMarray')
                

                dir_paths = [ir_np_path,pm_np_path]

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
        dic_cali = {}         
        for patient in os.listdir(path_arrays):
            patient_path = os.path.join(path_arrays,patient)
            if os.path.isdir(patient_path):
                dic_cali[patient]={}
                cal_indiv = np.load(os.path.join(patient_path, 'PMcali.npy'))
                pm_np_path = os.path.join(patient_path, 'PMarray')
                for category in os.listdir(pm_np_path):
                    category_path = os.path.join(pm_np_path, category)
                    if os.path.isdir(category_path):
                        dic_cali[patient][category]=[]
                        for p, file in enumerate(os.listdir(category_path)):
                            values = cal_indiv[:, p]
                            if category == 'cover1':
                                dic_cali[patient][category].append(values[1])

                            elif category == 'cover2':
                                dic_cali[patient][category].append(values[2])

                            elif category == 'uncover':
                                dic_cali[patient][category].append(values[0])
                        
        logger.info(f'Number of pacients: {len(dic_ir_numpy)}')
        logger.info(f'Number of categories in a patient: {len(dic_ir_numpy["00001"])}')
        logger.info(f'Calibration diccionary num of pacients: {len(dic_cali)}')
        logger.info(f'Number of values in a category for a patient: {len(dic_cali["00002"]["cover1"])}')
        
        if SHOW_IMAGES: # Show the IR, PM image and PM array of a uncover random patient
            show_image(dic_ir_numpy,'IR')
            show_image(dic_pm_numpy,'PM')

        if SHOW_HISTOGRAM:
            show_histogram(dic_ir_numpy,dic_pm_numpy,['IR','PM'])

        p_data = None

        if USE_PHYSICAL_DATA:

            p_data = pd.read_csv(os.path.join(path_arrays, 'physiqueData.csv'))
            p_data['gender'] = p_data['gender'].str.strip()

            p_data = pd.get_dummies(p_data, columns=['gender'])

            logger.info(f'Size of the physical dataset: {p_data.size}')
        
        random_patient = random.choice(list(dic_ir_numpy.keys()))
        patient_np = dic_ir_numpy[random_patient]['cover1'][0]

        arr = np.load(patient_np)
        logger.info(f'Shape of the IR numpy array: {arr.shape}')
        logger.info(f'Max value of IR array: {arr.max()}')

        patient_np = dic_pm_numpy[random_patient]['cover1'][0]
        arr = np.load(patient_np)
        logger.info(f'Shape of the PM numpy array: {arr.shape}')
        logger.info(f'Max value of PM array: {arr.max()}')

        # Create dataset
        logger.info("-" * 50)
        logger.info('Creating dataset...')

        # We try to get train and val images of each category of each patient.
            
        train_arrays = {'ir': [], 'pm': [], 'cali': []}
        val_arrays = {'ir': [], 'pm': [], 'cali': []}

        for ((patient_ir, category_ir), (patient_pm, category_pm), (patient_cali, category_cali)) in zip(dic_ir_numpy.items(), dic_pm_numpy.items(), dic_cali.items()):
            assert patient_ir == patient_pm == patient_cali

            for ((category_name_ir, arrays_ir), (category_name_pm, arrays_pm), (category_name_cali, arrays_cali)) in zip(category_ir.items(), category_pm.items(), category_cali.items()):
                assert category_name_ir == category_name_pm == category_name_cali

                #jugar per aqui per confirmar q tot es faci bé

                indexs = list(range(len(arrays_ir)))
                random.shuffle(indexs)


                for i in indexs[:int(len(indexs) * 0.8)]:
                    train_arrays['ir'].append(arrays_ir[i])
                    train_arrays['pm'].append(arrays_pm[i])
                    train_arrays['cali'].append(arrays_cali[i])
                if p_data is not None:
                    train_dt = p_data.iloc[indexs[:int(len(indexs) * 0.8)]]
                else:
                    train_dt = None

                for i in indexs[int(len(indexs)*0.8):]:
                    val_arrays['ir'].append(arrays_ir[i])
                    val_arrays['pm'].append(arrays_pm[i])
                    val_arrays['cali'].append(arrays_cali[i])
                if p_data is not None:
                    val_dt = p_data.iloc[indexs[int(len(indexs) * 0.8):]]
                else:
                    val_dt = None

        train_dataset = CustomDataset(train_arrays['ir'], train_arrays['pm'],train_arrays['cali'], train_dt, transform=transform)

        val_dataset = CustomDataset(val_arrays['ir'], val_arrays['pm'], val_arrays['cali'], val_dt, transform=transform)
        
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
        logger.info(f"Array input size of the train loader: {next(iter(train_loader))[0].shape}")
        logger.info(f"Array output size of the train loader: {next(iter(train_loader))[1].shape}")
        logger.info(f"Val loader info: {val_dataset_info}")
        logger.info(f"Array input size of the val loader: {next(iter(val_loader))[0].shape}")
        logger.info(f"Array output size of the val loader: {next(iter(val_loader))[1].shape}")

        check_transform(val_loader,self.path_arrays)
            
        
        return train_loader, val_loader


# ----------------------------------- EXTRA FUNCTIONS -----------------------------------

def crop_array(array):
        return crop(array,7, 29, 140, 66)

def check_transform(val_loader,path_arrays):
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

        plt.savefig(os.path.join(IMG_PATH, f'compare_transforms_{path_arrays}.png'))
            
        plt.show()


def show_image(dic,module):
    random_patient = random.choice(list(dic.keys()))
    patient_img_np = dic[random_patient]['cover1'][0]

    img_array = np.load(patient_img_np)
    img = np.array(img_array, dtype=np.uint8)
    cv2.imshow(f"{module} Image", img)
    #cv2.imwrite(os.path.join(IMG_PATH,f"{module}_image_{random_patient}_np.jpg"), img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    logger.info(f'Shape of the {module} Image np: {img.shape}')


def show_histogram(dic_ir, dic_pm, modules):
    random_patient = random.choice(list(dic_ir.keys()))
    if 'IR' in modules:
        patient_ir_np = dic_ir[random_patient]['cover1'][0]
        
        array = np.array(patient_ir_np)

        fig, axes = plt.subplots(1, 1, figsize=(8, 6))

        axes.hist(array.flatten(), 64, [0, 256], color='gray', alpha=0.5)
        axes.set_title('Grayscale Array Histogram')

        plt.savefig(os.path.join(IMG_PATH, 'IR_histogram.png'))
        plt.show()

    if 'PM' in modules:
        patient_pm_np = dic_pm[random_patient]['cover1'][0]
        
        array = np.array(patient_pm_np)

        fig, axes = plt.subplots(1, 1, figsize=(8, 6))

        axes.hist(array.flatten(), 64, [0, 256], color='gray', alpha=0.5)
        axes.set_title('Grayscale Array Histogram')

        plt.savefig(os.path.join(IMG_PATH, 'PM_histogram.png'))
        plt.show()
