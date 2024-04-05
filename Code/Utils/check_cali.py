import numpy as np
import os
import pandas as pd
from matplotlib import pyplot as plt 


for i, patient in enumerate(os.listdir('C:/Users/Gabriel/OneDrive/Escritorio/4t any uni/tfg/SLP/danaLab')):
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
                        #if n == 1:
                        #    img = np.array(array, dtype=np.uint8)
                        #    cv2.imshow(f"Image", img)
                            #cv2.imwrite(os.path.join(IMG_PATH,f"{module}_image_{random_patient}_np.jpg"), img)
                        #    cv2.waitKey(0)
                        #    cv2.destroyAllWindows()
                        values = cal_indiv[:, p]
                        if category == 'cover1':
                            pressure = array * values[1]

                        elif category == 'cover2':
                            pressure = array * values[2]

                        elif category == 'uncover':
                            pressure = array * values[0]
                        
                        fig, axs = plt.subplots(1, 2)

                        # Plot the array from file_ir
                        axs[0].imshow(array, cmap='gray')
                        axs[0].set_title('Array without cali')

                        # Plot the array from file_pm
                        axs[1].imshow(pressure, cmap='gray')
                        axs[1].set_title('Array after cali')

                        # Show the plot
                        plt.show()

                        break
                break
    break