import numpy as np
from matplotlib import pyplot as plt 
import os
import random
from scipy import signal
import pandas as pd
import cv2

import sys
sys.path.append(os.path.dirname(os.getcwd()))

from config import (
    LOCAL_SLP_DATASET_PATH,
    SERVER_SLP_DATASET_PATH
)

# File to apply the transforms in the PM values, show the arrays
# Also the histogram fo LWIR and PM arrays

path_data = os.path.join(SERVER_SLP_DATASET_PATH,'physiqueData.csv')
p_data = pd.read_csv(path_data)
mass = p_data['weight (kg)']

for i, patient in enumerate(os.listdir(SERVER_SLP_DATASET_PATH)[:1]):
    if os.path.isdir(os.path.join(SERVER_SLP_DATASET_PATH,patient)):
        print('Patient:', patient)
        patient_path = os.path.join(SERVER_SLP_DATASET_PATH,patient)
        if os.path.isdir(patient_path):
            pm_np_path = os.path.join(patient_path,'PMarray')
            ir_np_path = os.path.join(patient_path,'IRraw')
            for category_pm,category_ir in zip(os.listdir(pm_np_path),os.listdir(ir_np_path)):
                category_pm_path = os.path.join(pm_np_path,category_pm)
                category_ir_path = os.path.join(ir_np_path,category_ir)
                filenames_ir = os.listdir(category_ir_path)
                filenames_pm = os.listdir(category_pm_path)

                for file in filenames_pm:
                    file_pm = os.path.join(category_pm_path,file)
                    file_ir = os.path.join(category_ir_path,file)

                    # Read both LWIR and PM array
                    array_pm = np.load(file_pm)
                    array_ir = np.load(file_ir)

                    # Apply median filter
                    median = signal.medfilt2d(array_pm)

                    # Get the maximum values
                    maximum = np.maximum(array_pm,median)

                    # Normalize the values so the sum is the total corrected pressure
                    area_m = 1.03226 / 10000
                    ideal_pressure = mass[i] * 9.81 / (area_m * 1000)

                    output_array = (maximum / np.sum(maximum)) * ideal_pressure

                    output_array = (output_array - 0) / (102.36 - 0)


                    #transform array_ir
                    x = 29
                    y = 7
                    w = 66
                    h = 140

                    new_array = array_ir[y:y+h, x:x+w]
                    new_array = cv2.resize(new_array, (84, 192))

                    new_array = (new_array - 27195) / (31141 - 27195)

                    # Show histogram of LWIR and PM values
                    fig, axs = plt.subplots(1, 2, figsize=(14, 7))

                    axs[0].hist(new_array.flatten(), bins=100, color='royalblue')
                    axs[0].set_title("Histogram of LWIR array")
                    axs[1].hist(output_array.flatten(), bins=100, color='royalblue')
                    axs[1].set_title("Histogram of PM array")
                    plt.tight_layout()
                    #plt.savefig('Histogram_LWIR_PM.png')
                    plt.show()

                    # Show PM inicial
                    plt.imshow(array_pm)
                    plt.title('\nPM inicial')
                    #plt.savefig('PM_inicial.png')
                    plt.show()

                    # Show PM corrected
                    plt.imshow(output_array)
                    plt.title('\nPM corrected')
                    #plt.savefig('Median_filter_applied.png')
                    plt.show()