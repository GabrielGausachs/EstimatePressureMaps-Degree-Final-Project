from skimage import transform
import numpy as np
import math
from math import cos, sin

import cv2

imagen = cv2.imread('C:/Users/Gabriel/OneDrive/Escritorio/4t any uni/tfg/SLP/danaLab/00001/IR/uncover/image_000001.png')

# Coordenadas del rectángulo de recorte
x = 28
y = 7
ancho = 71
altura = 142

# Recortar la imagen
imagen_recortada = imagen[y:y+altura, x:x+ancho]

# Mostrar la imagen recortada
cv2.imshow("Imagen Recortada", imagen_recortada)
cv2.waitKey(0)  # Wait for a key press to close the window
cv2.destroyAllWindows()

imagen_redimensionada = cv2.resize(imagen_recortada, (84, 192))

cv2.imshow("Imagen redi", imagen_redimensionada)
cv2.waitKey(0)  # Wait for a key press to close the window
cv2.destroyAllWindows()
cv2.imwrite("imagen_redimensionada.jpg", imagen_redimensionada)