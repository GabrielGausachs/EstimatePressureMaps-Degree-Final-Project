import os
import random

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data import (
    DataLoader,
    random_split,
)

from logger import initialize_logger

from config import (
    BATCH_SIZE_TEST,
    BATCH_SIZE_TRAIN,
    DATASET,
    DEVICE,
    LOCAL_SLP_DATASET_PATH,
    SERVER_SLP_DATASET_PATH,
    NUM_WORKERS,
)

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
        transform = transforms.Compose([transforms.ToTensor()])

        # Get the data

        # 4 dictionaries for IR Images, IR numpys, PM Imatges & PM numpys
        # Each one has a diccionary for each patient
        # Each patient diccionary has 3 diccionaries that correspon to each category (uncover,cover1,cover2)
        # Each category has the respective paths
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
                                files = [os.path.join(category_path,file) for file in os.listdir(category_path) if file.endswith(('.png','.npy'))]
                                dic[patient][category]=files
                            else:
                                print(patient)
                                print(category)
                    else:
                        raise FileNotFoundError ('Path not found')
                    
        print(len(dic_ir_numpy))
        print(len(dic_pm_img))
        print(len(dic_ir_img['00001']))
                

        

print(os.listdir(LOCAL_SLP_DATASET_PATH))
f = CustomDataloader().prepare_dataloaders()
# Per cada pacient, vaig a la carpeta IR, vaig a cada carpeta (cover,...) i extrec les imatges
# i guardo els paths en una llista. -- IRImages
# Per cada pacient, vaig a la carpeta IRRAW, vaig a cada carpeta, llegeixo cada arxiu .npy i
# ho guardo en una llista. -- IRNumpys