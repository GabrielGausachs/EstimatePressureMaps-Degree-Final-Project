import numpy as np
from matplotlib import pyplot as plt 
import os
import cv2
import random
from scipy import signal
import pandas as pd

p_data = pd.read_csv(os.path.join(
    'C:/Users/Gabriel/OneDrive/Escritorio/4t any uni/tfg/SLP/danaLab', 'physiqueData.csv'))

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
                print('pm array:',file_pm)
                array_pm = np.load(file_pm)
                array_ir = np.load(file_ir)

                median = signal.medfilt2d(array_pm)

                maximum = np.maximum(array_pm,median)

                print(mass[i])
                area_m = 1.03226 / 10000
                ideal_pressure = mass[i] * 9.81 / (area_m * 1000)

                output_array = (maximum / np.sum(maximum)) * ideal_pressure

                maximum_value_output = np.max(output_array)
                maximum_value_array = np.max(array_pm)
                min_value_output = np.min(output_array)
                min_value_array = np.min(array_pm)


                
                
                plt.imshow(array_pm)
                plt.title(f'\nPM inicial\nMin and Max value: ({round(min_value_array,2)},{round(maximum_value_array,2)})')
                plt.savefig('PM_inicial.png')
                plt.show()

                plt.imshow(output_array)
                plt.title(f'\nPM corrected\nMin and Max value: ({round(min_value_output,2)},{round(maximum_value_output,2)})')
                plt.savefig('Median_filter_applied.png')
                # Show the plot
                plt.show()

                break
    break
                
