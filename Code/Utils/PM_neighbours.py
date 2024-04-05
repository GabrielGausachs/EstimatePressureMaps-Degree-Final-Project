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
"""
class ImageROISelector:
    def __init__(self, image):
        self.image = image
        self.fig, self.ax = plt.subplots()
        self.ax.imshow(self.image, cmap='gray')
        self.roi = None
        self.rect = None
        self.selected = False

        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)

    def on_press(self, event):
        if event.button == 1:
            self.selected = False
            self.roi = None
            if self.rect:
                self.rect.remove()
            self.rect = plt.Rectangle((event.xdata, event.ydata), 1, 1, edgecolor='r', facecolor='none')
            self.ax.add_patch(self.rect)
            self.fig.canvas.draw()

    def on_release(self, event):
        if event.button == 1:
            self.selected = True
            x0, y0 = self.rect.xy
            x1 = event.xdata
            y1 = event.ydata
            self.roi = (int(y0), int(y1), int(x0), int(x1))
            print("Selected ROI:", self.roi)
            self.rect.remove()
            self.fig.canvas.draw()

    def get_roi(self):
        return self.roi
"""
array = np.load('/mnt/DADES2/SLP/SLP/danaLab/00001/PMarray/uncover/000017.npy')

row_start, row_end = 25, 51
col_start, col_end = 50, 72

# Crop the array
array = array[row_start:row_end, col_start:col_end]
print(array.shape)

# Display the image and select a region of interest
#roi_selector = ImageROISelector(array)
#plt.show()

# Once you've selected the ROI, you can retrieve it using:
#selected_roi = roi_selector.get_roi()
#print("Selected ROI:", selected_roi)

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

p=max(diff_global.items(), key=lambda x: x[1])
print(p)

# Definimos el tamaño de la vecindad alrededor del índice central
tamano_vecindad = 1

# Creamos una máscara booleana para la vecindad
mask = np.zeros_like(array, dtype=bool)
mask[max(0, p[0][0] - tamano_vecindad):min(array.shape[0], p[0][0]+ tamano_vecindad + 1),
     max(0, p[0][1] - tamano_vecindad):min(array.shape[1], p[0][1] + tamano_vecindad + 1)] = True

# Obtenemos los índices de los elementos verdaderos en la máscara
indices = np.argwhere(mask)

# Iteramos sobre los índices y mostramos los valores correspondientes
for indice in indices:
    print(f"Índice: {tuple(indice)}, Valor: {array[tuple(indice)]}")

plt.imshow(array, cmap='gray')
plt.show()
