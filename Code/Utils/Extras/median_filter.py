import numpy as np
from matplotlib import pyplot as plt 
import os
import cv2
import random
from scipy import signal

for i, patient in enumerate(os.listdir('C:/Users/Gabriel/OneDrive/Escritorio/4t any uni/tfg/SLP/danaLab')):
    if os.path.isdir(os.path.join('C:/Users/Gabriel/OneDrive/Escritorio/4t any uni/tfg/SLP/danaLab',patient)):
        print('Patient:', patient)
        patient_path = os.path.join('C:/Users/Gabriel/OneDrive/Escritorio/4t any uni/tfg/SLP/danaLab',patient)
        if os.path.isdir(patient_path):
            pm_np_path = os.path.join(patient_path,'PMarray')
            ir_np_path = os.path.join(patient_path,'IRraw')
            for category_pm,category_ir in zip(os.listdir(pm_np_path),os.listdir(ir_np_path)):
                category_pm_path = os.path.join(pm_np_path,category_pm)
                filenames_pm = os.listdir(category_pm_path)

                random_index = random.randint(0, len(filenames_pm) - 1)
                file_pm = os.path.join(category_pm_path,filenames_pm[random_index])
                print('pm array:',file_pm)
                array_pm = np.load(file_pm)

                median = signal.medfilt2d(array_pm)

                maximum = np.maximum(array_pm,median)

                fig, axs = plt.subplots(1, 3)

                # Plot the array from file_ir
                axs[0].imshow(array_pm)
                axs[0].set_title('Array from PM')

                # Plot the array from file_pm
                axs[1].imshow(median)
                axs[1].set_title('Median filter')

                axs[2].imshow(maximum)
                axs[2].set_title('Maximum')

                # Show the plot
                plt.show()
