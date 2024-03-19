import os
import random

import numpy as np
import cv2
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt 
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
    LOCAL_SLP_DATASET_PATH,
    SERVER_SLP_DATASET_PATH,
    NUM_WORKERS,
    SHOW_IMAGES,
    USE_PHYSICAL_DATA,
    SHOW_HISTOGRAM,
    IMG_PATH,
    IS_RANDOM,
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

    def prepare_dataloaders(self):
        """Prepare dataloaders for training and testing"""

        # Data transformation if needed
        transform = transforms.Compose([transforms.Resize((192, 144)),
                                              transforms.CenterCrop((192,84)),
                                        transforms.ToTensor()])
        
        #transform = transforms.Compose([transforms.CenterCrop((160,84)),
        #                                transforms.Resize((192,84)),
        #                                transforms.ToTensor()])
        
        
        
        #transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])


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
            show_image(dic_ir_img,'IR',dic_ir_numpy)
            show_image(dic_pm_img,'PM',dic_pm_numpy)

        if SHOW_HISTOGRAM:
            show_histogram(dic_ir_img,dic_pm_img,['IR','PM'])

        p_data = None

        if USE_PHYSICAL_DATA:

            p_data = pd.read_csv(os.path.join(self.local_slp, 'physiqueData.csv'))
            p_data['gender'] = p_data['gender'].str.strip()

            p_data = pd.get_dummies(p_data, columns=['gender'])

            logger.info(f'Size of the physical dataset: {p_data.size}')
        
        # Check how many channels have the images
        num_single_channel_images = 0
        total_images=0
        
        for patient in dic_ir_img.values():
            for category, images_path in patient.items():
                for path in images_path:
                    input_image = cv2.imread(path)
                    num_channels = input_image.shape[2]  # Assuming input_image is in (H, W, C) format

                    if num_channels == 1:
                        num_single_channel_images += 1
                    total_images+=1

        logger.info(f"Number of images with only 1 channel: {num_single_channel_images} / {total_images}")
        
        random_patient = random.choice(list(dic_ir_numpy.keys()))
        patient_img = dic_ir_numpy[random_patient]['cover1'][0]

        """
        arreglo = np.load(patient_img)

        # Imprimir la forma del arreglo
        print("Forma del arreglo:", arreglo.shape)

        random_patient = random.choice(list(dic_pm_numpy.keys()))
        patient_img = dic_pm_numpy[random_patient]['cover1'][0]
        arreglo = np.load(patient_img)

        # Imprimir la forma del arreglo
        print("Forma del arreglo 2:", arreglo.shape)
        """

        # Create dataset
        logger.info("-" * 50)
        logger.info(f'Create dataset with random = {IS_RANDOM}')
        
        if IS_RANDOM:

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

        num_single_channel_images = 0

        for i in range(len(train_dataset)):
            input_image, _ = train_dataset[i]

            num_channels = input_image.shape[0]  # Assuming input_image is in (C, H, W) format
            
            if num_channels == 1:
                num_single_channel_images += 1
        
        logger.info(f"Number of images with only 1 channel: {num_single_channel_images} / {len(train_dataset)}")

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
        logger.info(f"Image input size of the train loader: {next(iter(train_loader))[0].shape}")
        logger.info(f"Image output size of the train loader: {next(iter(train_loader))[1].shape}")
        logger.info(f"Val loader info: {val_dataset_info}")
        logger.info(f"Image input size of the val loader: {next(iter(val_loader))[0].shape}")
        logger.info(f"Image output size of the val loader: {next(iter(val_loader))[1].shape}")

        """
        for i in range(5):
            batch = next(iter(train_loader))
            
            # Extrae una imagen de entrada y su correspondiente imagen objetivo del lote
            imagen_entrada = batch[0][0]  # Primera imagen del lote
            imagen_objetivo = batch[1][0]  # Primera imagen objetivo del lote

            # Convierte las imágenes a NumPy y transpónlas para que sean compatibles con matplotlib
            imagen_entrada_numpy = imagen_entrada.permute(1, 2, 0).numpy()
            imagen_objetivo_numpy = imagen_objetivo.permute(1, 2, 0).numpy()

            # Muestra las imágenes
            plt.figure(figsize=(10, 5))
            
            plt.subplot(1, 2, 1)
            plt.imshow(imagen_entrada_numpy)
            plt.title('Imagen de entrada')
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(imagen_objetivo_numpy)
            plt.title('Imagen objetivo')
            plt.axis('off')

            plt.show()
        """

        return train_loader, val_loader

# ----------------------------------- EXTRA FUNCTIONS -----------------------------------

def show_image(dic,module,dic_np):
    random_patient = random.choice(list(dic.keys()))

    patient_img = dic[random_patient]['cover1'][0]
    patient_img_np = dic_np[random_patient]['cover1'][0]
    img = cv2.imread(patient_img)
    cv2.imshow(f"{module} Image", img)

    img_array = np.load(patient_img_np)
    img = np.array(img_array, dtype=np.uint8)
    cv2.imshow(f"{module} Image", img)
    cv2.imwrite(os.path.join(IMG_PATH,f"{module}_image_{random_patient}_np.jpg"), img)
    cv2.waitKey(0)  # Wait for a key press to close the window
    cv2.destroyAllWindows()
    logger.info(f'Shape of the {module} Image: {img.shape}')

def show_histogram(dic_ir,dic_pm,modules):
    random_patient = random.choice(list(dic_ir.keys()))
    if 'IR' in modules:
        patient_ir_img = dic_ir[random_patient]['cover1'][0]
        img = cv2.imread(patient_ir_img) 
        b, g, r = cv2.split(img)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].hist(b.flatten(), 256, [0, 256], color='blue', alpha=0.5)
        axes[0].set_title('Blue Channel')

        axes[1].hist(g.flatten(), 256, [0, 256], color='green', alpha=0.5)
        axes[1].set_title('Green Channel')

        axes[2].hist(r.flatten(), 256, [0, 256], color='red', alpha=0.5)
        axes[2].set_title('Red Channel')

        plt.savefig(os.path.join(IMG_PATH,'IR_histogram.png'))
        plt.show()
    
    if 'PM' in modules:
        patient_pm_img = dic_pm[random_patient]['cover1'][0]
        img = cv2.imread(patient_pm_img) 
        b, g, r = cv2.split(img)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].hist(b.flatten(), 256, [0, 256], color='blue', alpha=0.5)
        axes[0].set_title('Blue Channel')

        axes[1].hist(g.flatten(), 256, [0, 256], color='green', alpha=0.5)
        axes[1].set_title('Green Channel')

        axes[2].hist(r.flatten(), 256, [0, 256], color='red', alpha=0.5)
        axes[2].set_title('Red Channel')

        plt.savefig(os.path.join(IMG_PATH,'PM_histogram.png'))
        plt.show()