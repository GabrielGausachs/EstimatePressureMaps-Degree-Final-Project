from skimage import transform
import numpy as np
import math
from math import cos, sin
from matplotlib import pyplot as plt 

import cv2

imagen = cv2.imread('C:/Users/Gabriel/OneDrive/Escritorio/4t any uni/tfg/SLP/danaLab/00001/IR/uncover/image_000001.png')

# Definir las coordenadas del área de interés (ROI) para el recorte
x = 28
y = 7
ancho = 71
altura = 142

# Recortar la región de interés (ROI)
imagen_recortada = imagen[y:y+altura, x:x+ancho]
imagen_escala = cv2.resize(imagen_recortada, (84, 192))
# Mostrar la imagen redimensionada
#cv2.imshow("Imagen Redimensionada", imagen_recortada)
#cv2.waitKey(0)
#cv2.destroyAllWindows()


plt.imshow(cv2.cvtColor(imagen_escala, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()  
