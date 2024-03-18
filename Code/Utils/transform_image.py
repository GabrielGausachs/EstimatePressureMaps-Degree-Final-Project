from skimage import transform
import numpy as np
import math
from math import cos, sin

import cv2

def affineImg(img, scale=1,deg=0,  shf=(0,0)):
    '''
    scale, rotate and shift around center, same cropped image will be returned with skimage.transform.warp. use anti-clockwise
    :param img:  suppose to be 2D, or HxWxC format
    :param deg:
    :param shf:
    :param scale:
    :return:
    '''
    h,w = img.shape[:2] #
    c_x = (w+1)/2
    c_y = (h+1)/2
    rad = -math.radians(deg) #
    M_2Cs= np.array([
        [scale, 0, -scale * c_x],
        [0, scale, -scale * c_y],
        [0, 0,  1]
    ])
    M_rt = np.array([
        [cos(rad), -sin(rad), 0],
        [sin(rad), cos(rad), 0],
        [0, 0 ,     1]
    ])
    M_2O = np.array([
        [1, 0, c_x+shf[0]],
        [0, 1,  c_y+shf[1]],
        [0, 0 , 1]
                    ])
    # M= M_2O  * M_2Cs
    #M= np.linalg.multi_dot([M_2O, M_rt, M_2Cs]) # [2,2, no shift part?
    M= M_2O @ M_rt @ M_2Cs
    tsfm = transform.AffineTransform(np.linalg.inv(M))
    img_new = transform.warp(img, tsfm, preserve_range=True)
    return img_new


imagen = cv2.imread('C:/Users/Gabriel/OneDrive/Escritorio/4t any uni/tfg/SLP/danaLab/00003/PMarray/uncover/000001.npy')
img = affineImg(imagen)
cv2.imshow('new',img)
cv2.waitKey(0)  # Wait for a key press to close the window
cv2.destroyAllWindows()
