import cv2
import numpy as np
import os
import pandas as pd


# File to check the obtained weight after applying the respected 
# calibration value in each of the PM arrays

# Uncover first row, cover1 second row, cover2 third row.

path_data = os.path.join(
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))), 'SLP/danaLab/physiqueData.csv')

p_data = pd.read_csv(path_data)

mass = p_data['weight (kg)']

dic = {}

for i, patient in enumerate(os.listdir('C:/Users/Gabriel/OneDrive/Escritorio/4t any uni/tfg/SLP/danaLab')):
    if os.path.isdir(os.path.join('C:/Users/Gabriel/OneDrive/Escritorio/4t any uni/tfg/SLP/danaLab', patient)):
        print(f'patient: {i}, mass: {mass[i]}')

        patient_path = os.path.join(
            'C:/Users/Gabriel/OneDrive/Escritorio/4t any uni/tfg/SLP/danaLab', patient)
        cal_indiv = np.load(os.path.join(patient_path, 'PMcali.npy'))
        if i==0:
            print(cal_indiv.shape)
        if os.path.isdir(patient_path):
            pm_np_path = os.path.join(patient_path, 'PMarray')
            n = 0
            sumatori = 0
            massa_total = 0
            for category in os.listdir(pm_np_path):
                category_path = os.path.join(pm_np_path, category)
                if os.path.isdir(category_path):
                    for p, file in enumerate(os.listdir(category_path)):
                        array = np.load(os.path.join(category_path, file))
                        values = cal_indiv[:, p]
                        if category == 'cover1':
                            pressure = array * values[1]

                        elif category == 'cover2':
                            pressure = array * values[2]

                        elif category == 'uncover':
                            pressure = array * values[0]

                        else:
                            print(category, 'is not a category')

                        area_m = 1.03226 / 10000
                        massa_cali = area_m * (np.sum(pressure)*1000) / 9.81
                        massa_array = area_m * (np.sum(array)*1000) / 9.81
                        #print('array weight:',massa_array)
                        dif = abs(mass[i]-massa_cali)
                        massa_total = massa_total+massa_cali
                        sumatori = sumatori+dif
                        n += 1
                
    
        dic[patient] = sumatori / n
        m = massa_total / n
        print('meaan diference:', dic[patient])
        print('mean weight:', m)
        print('difference between real weight and mean weight:',abs(mass[i]-m))