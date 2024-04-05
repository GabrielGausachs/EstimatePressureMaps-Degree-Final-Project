import numpy as np
import os
import pandas as pd
from matplotlib import pyplot as plt 

"""
for i, patient in enumerate(os.listdir('C:/Users/Gabriel/OneDrive/Escritorio/4t any uni/tfg/SLP/danaLab'))[:3]:
    if os.path.isdir(os.path.join('C:/Users/Gabriel/OneDrive/Escritorio/4t any uni/tfg/SLP/danaLab', patient)):
        print(f'patient: {i}')

        patient_path = os.path.join(
            'C:/Users/Gabriel/OneDrive/Escritorio/4t any uni/tfg/SLP/danaLab', patient)
        cal_indiv = np.load(os.path.join(patient_path, 'PMcali.npy'))
        if i==0:
            print(cal_indiv.shape)
        if os.path.isdir(patient_path):
            pm_np_path = os.path.join(patient_path, 'PMarray')
            for category in os.listdir(pm_np_path):
                category_path = os.path.join(pm_np_path, category)
                if os.path.isdir(category_path):
                    for p, file in enumerate(os.listdir(category_path)):
                        array = np.load(os.path.join(category_path, file))
"""

import numpy as np
from scipy.signal import convolve2d

array = np.load('C:/Users/Gabriel/OneDrive/Escritorio/4t any uni/tfg/SLP/danaLab/00001/PMarray/uncover/000001.npy')

print(array.shape)

kernel = [[1,1,1],  # define points to pick around the target
          [1,0,1],
          [1,1,1]]

diff_global = {}

for index in np.ndindex(array.shape):
    x = index[0]
    y = index[1]
    
    mask = np.zeros_like(array, dtype=bool)   
    mask[x,y] = True               # set target(s)
    # boolean indexing
    neighbours = array[convolve2d(mask, kernel, mode='same').astype(bool)]

    value = array[index]
    diff_t = 0
    for n in neighbours:
        diff_t += abs(value-n)
    diff_global[(x,y)]=diff_t/len(neighbours)

print(max(diff_global.items(), key=lambda x: x[1]))
    

    
