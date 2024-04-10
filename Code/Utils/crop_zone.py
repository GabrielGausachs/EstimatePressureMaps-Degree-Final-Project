import numpy as np
import os
import pandas as pd
from matplotlib import pyplot as plt 
import numpy as np
from scipy.signal import convolve2d
import random


pm = np.load('C:/Users/Gabriel/OneDrive/Escritorio/4t any uni/tfg/SLP/danaLab/00001/PMarray/uncover/000001.npy')

#x =43 y= 90
plt.imshow(pm)
plt.title('first image')
plt.show()

kernel = [[1,1,1],
        [1,0,1],
        [1,1,1]]
mask = np.zeros_like(pm, dtype=bool)   
mask[149,61] = True               # set target(s)
# boolean indexing
neighbours = pm[convolve2d(mask, kernel, mode='same').astype(bool)]

# Extract the value of the target pixel
target_value = pm[149, 61]

# Calculate the mean of the neighbor values
mean_neighbours = np.mean(neighbours)

# Calculate the coefficient
coefficient = target_value / mean_neighbours

print('Coefficient for pixel (43, 90):', coefficient)

for i, patient in enumerate((os.listdir('C:/Users/Gabriel/OneDrive/Escritorio/4t any uni/tfg/SLP/danaLab')[:10])):
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
                    random_file = os.listdir(category_path)[random_index]
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
                    
                    hist, bin_edges = np.histogram(pm, bins = range(50))  # You can adjust the number of bins as needed
                    plt.bar(bin_edges[:-1], hist, width = 1)
                    plt.xlim(min(bin_edges), max(bin_edges))
                    plt.show()
                    #plt.imshow(pm)
                    #plt.title('normal image')
                    #plt.show()
                    """
                    mask = np.zeros_like(pm, dtype=bool)   
                    mask[149,61] = True               # set target(s)
                    # boolean indexing
                    neighbours = pm[convolve2d(mask, kernel, mode='same').astype(bool)]

                   # Extract the value of the target pixel
                    target_value = pm[149, 61]

                    # Calculate the mean of the neighbor values
                    mean_neighbours = np.mean(neighbours)

                    # Calculate the coefficient
                    coefficient = target_value / mean_neighbours

                    print('Coefficient for pixel (43, 90):', coefficient)
                    cropped = pm[80:100,33:53]

                    #print(pm[90,43])
                    #plt.imshow(cropped)
                    #plt.title('cropped image')
                    #plt.show()
                    """