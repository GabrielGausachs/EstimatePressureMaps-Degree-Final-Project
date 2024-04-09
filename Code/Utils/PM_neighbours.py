import numpy as np
import os
import pandas as pd
from matplotlib import pyplot as plt 
import numpy as np
from scipy.signal import convolve2d
import random

kernel = [[1,1,1],
        [1,0,1],
        [1,1,1]]
#array = np.load('/mnt/DADES2/SLP/SLP/danaLab/00001/PMarray/uncover/000017.npy')
for i, patient in enumerate((os.listdir('C:/Users/Gabriel/OneDrive/Escritorio/4t any uni/tfg/SLP/danaLab')[:5])):
    if os.path.isdir(os.path.join('C:/Users/Gabriel/OneDrive/Escritorio/4t any uni/tfg/SLP/danaLab', patient)):
        print(f'patient: {i}')
        patient_path = os.path.join(
            'C:/Users/Gabriel/OneDrive/Escritorio/4t any uni/tfg/SLP/danaLab', patient)
        cal_indiv = np.load(os.path.join(patient_path, 'PMcali.npy'))
        if os.path.isdir(patient_path):
            pm_np_path = os.path.join(patient_path, 'PMarray')
            for category in os.listdir(pm_np_path):
                category_path = os.path.join(pm_np_path, category)
                if os.path.isdir(category_path):
                    num_files = len(os.listdir(category_path))
                    random_index = random.randint(0, num_files - 1)
                    print(random_index)
                    random_file = os.listdir(category_path)[random_index]
                    print('path of the file:', os.path.join(category_path,random_file))
                    pm = np.load(os.path.join(category_path, random_file))

                    values = cal_indiv[:, random_index]
                    if category == 'cover1':
                        pressure = pm * values[1]

                    elif category == 'cover2':
                        pressure = pm * values[2]

                    elif category == 'uncover':
                        pressure = pm * values[0]

                    else:
                        print(category, 'is not a category')
                    
                    #pm = pressure
                    print('shape:',pm.shape)
                    area_m = 1.03226 / 10000
                    massa = area_m * (np.sum(pressure)*1000) / 9.81
                    print('mass:',massa)
                    print('max value:',pm.max())
                    print('min value:',pm.min())

                    diff_global = {}

                    for index in np.ndindex(pm.shape):
                        x = index[0]
                        y = index[1]

                        mask = np.zeros_like(pm, dtype=bool)   
                        mask[x,y] = True               # set target(s)
                        # boolean indexing
                        neighbours = pm[convolve2d(mask, kernel, mode='same').astype(bool)]

                        value = pm[index]
                        diff_t = 0
                        for n in neighbours:
                            diff_t += abs(value-n)
                        diff_global[(x,y)]=diff_t/len(neighbours)

                    most_diff=max(diff_global.items(), key=lambda x: x[1])
                    print('Most difference:',most_diff)

                    plt.imshow(pm, cmap='gray')
                    plt.plot(most_diff[0][1], most_diff[0][0], 'ro', markersize = 5)  # Mark the point with maximum difference
                    plt.show()
                    print('-------------------------------------------')

"""
#array = np.load('/mnt/DADES2/SLP/SLP/danaLab/00001/PMarray/uncover/000017.npy')
array = np.load('C:/Users/Gabriel/OneDrive/Escritorio/4t any uni/tfg/SLP/danaLab/00001/PMarray/cover1/000039.npy')
diff_global = {}

kernel = [[1,1,1],
          [1,0,1],
          [1,1,1]]
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

p=max(diff_global.items(), key=lambda x: x[1])
print(p)

plt.imshow(array, cmap='gray')
plt.plot(p[0][1], p[0][0], 'ro', markersize = 5)  # Mark the point with maximum difference

plt.show()
"""