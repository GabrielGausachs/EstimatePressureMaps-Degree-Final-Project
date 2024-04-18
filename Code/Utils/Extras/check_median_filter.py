import numpy as np
from matplotlib import pyplot as plt 
import os
import cv2
import random
from scipy import signal
import pandas as pd

# File to apply the transforms in the PM values, show the arrays
# Also the histogram fo LWIR and PM arrays

path_data = os.path.join(
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))), 'SLP/danaLab/physiqueData.csv')

p_data = pd.read_csv(path_data)

mass = p_data['weight (kg)']

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

                random_index = random.randint(0, len(filenames_pm) - 1)
                file_ir = os.path.join(category_ir_path,filenames_ir[random_index])
                file_pm = os.path.join(category_pm_path,filenames_pm[random_index])

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

                # Get the min and max values
                maximum_value_output = np.max(output_array)
                maximum_value_array = np.max(array_pm)
                min_value_output = np.min(output_array)
                min_value_array = np.min(array_pm)
                
                # Show PM inicial
                plt.imshow(array_pm)
                plt.title(f'\nPM inicial\nMin and Max value: ({round(min_value_array,2)},{round(maximum_value_array,2)})')
                plt.savefig('PM_inicial.png')
                plt.show()

                # Show PM corrected
                plt.imshow(output_array)
                plt.title(f'\nPM corrected\nMin and Max value: ({round(min_value_output,2)},{round(maximum_value_output,2)})')
                plt.savefig('Median_filter_applied.png')
                plt.show()

                # Show histogram of LWIR and PM values
                fig, axs = plt.subplots(1, 2, figsize=(14, 7))

                axs[0].hist(array_ir.flatten(), bins=100, color='royalblue')
                axs[0].set_title("Histogram of LWIR array")
                axs[1].hist(output_array.flatten(), bins=100, color='royalblue')
                axs[1].set_title("Histogram of PM array")
                plt.tight_layout()
                plt.savefig('Histogram_LWIR_PM.png')
                plt.show()

                break
    break
