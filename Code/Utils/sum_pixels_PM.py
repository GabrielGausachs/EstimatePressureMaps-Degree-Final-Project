

import cv2
import numpy as np

# Lee la imagen
imagen = cv2.imread('C:/Users/Gabriel/OneDrive/Escritorio/4t any uni/tfg/SLP/danaLab/00001/PM/uncover/image_000001.png',cv2.IMREAD_GRAYSCALE)

# Espaciado entre filas y columnas del sensor en centímetros
row_spacing_cm = 1.016  
col_spacing_cm = 1.016  

image_width_pixels = 84
image_height_pixels = 192



pixel_width_m = row_spacing_cm / 100
pixel_height_m =  col_spacing_cm/ 100

# Calcular el área de cada píxel en metros cuadrados
pixel_area_m2 = (pixel_width_m * pixel_height_m)  # Área de cada píxel en metros cuadrados
print(pixel_area_m2*10000)

print(np.sum(imagen)*pixel_area_m2/9.81)
print('max imagen',np.max(imagen))


array = np.load('C:/Users/Gabriel/OneDrive/Escritorio/4t any uni/tfg/SLP/danaLab/00001/PMarray/uncover/000001.npy')

sumatori = np.sum(array)
print(sumatori)

# Pressio = F / area --> Area = m * g / Pressio  --> m = Area * Pressio / g
print(np.sum(array)*pixel_area_m2/9.81)
print('max array',np.max(array))

