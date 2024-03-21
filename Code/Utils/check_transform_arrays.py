from skimage import transform
import numpy as np
import math
from math import cos, sin
from matplotlib import pyplot as plt 
import os
import cv2
import random

for i, patient in enumerate(os.listdir('C:/Users/Gabriel/OneDrive/Escritorio/4t any uni/tfg/SLP/danaLab')):
    if os.path.isdir(os.path.join('C:/Users/Gabriel/OneDrive/Escritorio/4t any uni/tfg/SLP/danaLab',patient)):
        print('Patient:', patient)
        patient_path = os.path.join('C:/Users/Gabriel/OneDrive/Escritorio/4t any uni/tfg/SLP/danaLab',patient)
        if os.path.isdir(patient_path):
            pm_np_path = os.path.join(patient_path,'PMarray')
            ir_np_path = os.path.join(patient_path,'IRraw')
            for category_pm,category_ir in zip(os.listdir(pm_np_path),os.listdir(ir_np_path)):
                category_pm_path = os.path.join(pm_np_path,category_pm)
                category_ir_path = os.path.join(ir_np_path,category_ir)
                filenames_ir = os.listdir(category_ir_path)
                filenames_pm = os.listdir(category_pm_path)

                random_index = random.randint(0, len(filenames_ir) - 1)
                file_ir = os.path.join(category_ir_path,filenames_ir[random_index])
                file_pm = os.path.join(category_pm_path,filenames_pm[random_index])
                print('ir array:',file_ir)
                print('pm array:',file_pm)
                array_ir = np.load(file_ir)
                array_pm = np.load(file_pm)

                #transform array_ir
                x = 29
                y = 7
                ancho = 66
                altura = 140

                array_ir = array_ir[y:y+altura, x:x+ancho]
                array_ir = cv2.resize(array_ir, (84, 192))

                fig, axs = plt.subplots(1, 2)

                # Plot the array from file_ir
                axs[0].imshow(array_ir, cmap='gray')
                axs[0].set_title('Array from IR')

                # Plot the array from file_pm
                axs[1].imshow(array_pm, cmap='gray')
                axs[1].set_title('Array from PM')

                # Show the plot
                plt.show()