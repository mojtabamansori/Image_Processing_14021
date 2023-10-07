import numpy as np
from PIL import Image
from scipy.interpolate import interp2d
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy import interpolate
import sympy as syp
import pandas as pd
import cv2
import math


def upscale_nearest_neighbor(image, scale_factor):
    print('nn is run', end='')
    new_height = int(image.shape[0] * scale_factor)
    new_width = int(image.shape[1] * scale_factor)

    upscaled_image = np.zeros((new_height, new_width, image.shape[2]), dtype=np.uint8)

    h_ratio = image.shape[0] / new_height
    w_ratio = image.shape[1] / new_width

    for i in range(new_height):
        for j in range(new_width):
            orig_i = int(i * h_ratio)
            orig_j = int(j * w_ratio)
            upscaled_image[i, j, :] = image[orig_i, orig_j, :]
    print('\rnn is done\n')
    return upscaled_image
