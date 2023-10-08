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


def bilinear_interpolation(image, scale_factor):
    print('bilinear is run',end='')

    new_height = int(image.shape[0] * scale_factor)
    new_width = int(image.shape[1] * scale_factor)

    upscaled_image = np.zeros((new_height, new_width, image.shape[2]), dtype=np.uint8)

    h_ratio = 1 / scale_factor
    w_ratio = 1 / scale_factor

    for i in range(new_height):
        for j in range(new_width):
            orig_i = i * h_ratio
            orig_j = j * w_ratio

            i0, j0 = int(orig_i), int(orig_j)
            i1, j1 = min(i0 + 1, image.shape[0] - 1), min(j0 + 1, image.shape[1] - 1)

            weight_i0 = orig_i - i0
            weight_i1 = 1 - weight_i0
            weight_j0 = orig_j - j0
            weight_j1 = 1 - weight_j0

            interpolated_pixel = (
                    weight_i1 * (weight_j1 * image[i0, j0, :] + weight_j0 * image[i0, j1, :]) +
                    weight_i0 * (weight_j1 * image[i1, j0, :] + weight_j0 * image[i1, j1, :])
            )

            upscaled_image[i, j, :] = interpolated_pixel.astype(np.uint8)
    print('\rbilinear is done\n')
    return upscaled_image