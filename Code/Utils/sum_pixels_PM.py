import cv2
import numpy as np

# 1. uncover primera fila, cover 1 segona fila, cover 2 tercera fila.
# Lee la imagen
img = cv2.imread('C:/Users/Gabriel/OneDrive/Escritorio/4t any uni/tfg/SLP/danaLab/00001/PM/cover1/image_000010.png')

array = np.load('C:/Users/Gabriel/OneDrive/Escritorio/4t any uni/tfg/SLP/danaLab/00001/PMarray/cover1/000010.npy')
cal_indiv = np.load('C:/Users/Gabriel/OneDrive/Escritorio/4t any uni/tfg/SLP/danaLab/00001/PMcali.npy')
print(cal_indiv.shape)
print(array.shape)
values = cal_indiv[:, 9]


pressure1 = img * values[0]
pressure2 = img * values[1]
pressure3 = array * values[2]

area_cm = 1.03226
area_m = 1.03226 / 10000
massa = area_m * (np.sum(pressure3)*1000) / 9.81
print('massa:', massa)
