import numpy as np
from matplotlib import pyplot as plt 
import os
import cv2
import random

# File to check the transforms applied in the LWIR images (Crop and resize)

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
                for file in filenames_ir:
                    file_ir = os.path.join(category_ir_path,file)

                    if '00001\IRraw\cover1' in file_ir and '000001' in file:
                        print('ir array:',file_ir)
                        array_ir = np.load(file_ir)

                        #transform array_ir
                        x = 29
                        y = 7
                        ancho = 66
                        altura = 140

                        new_array = array_ir[y:y+altura, x:x+ancho]
                        new_array = cv2.resize(new_array, (84, 192))

                        fig, axs = plt.subplots(1, 2)

                        # Plot before transform
                        axs[0].imshow(array_ir)
                        axs[0].set_title('Inicial LWIR')

                        # Plot after transform
                        axs[1].imshow(new_array)
                        axs[1].set_title('LWIR corrected')

                        # Show the plot
                        plt.show()
