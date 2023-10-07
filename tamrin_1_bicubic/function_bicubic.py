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




def bicubic_interpolation(image, new_shape):
    print('bicubic is run',end='')

    height, width, _ = image.shape
    new_height, new_width = new_shape
    output_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)  # Create a 3-channel output image.

    # Calculate the scaling factors.
    y_scale = float(height) / new_height
    x_scale = float(width) / new_width

    # Iterate through the new image pixels.
    for i in range(new_height):
        print(f'\rrun bicubic: [{i}/{new_height}]', end='')
        for j in range(new_width):
            # Map the new pixel coordinates to the original image coordinates.
            x_original = j * x_scale
            y_original = i * y_scale

            # Calculate the four nearest neighbor pixel positions.
            x1 = int(x_original)
            y1 = int(y_original)

            # Calculate the fractional parts.
            dx = x_original - x1
            dy = y_original - y1

            # Initialize interpolated pixel values for each channel.
            interpolated_pixel = [0, 0, 0]

            for c in range(3):  # Iterate over color channels (R, G, B).
                # Perform bicubic interpolation for each channel.
                channel_value = 0
                for m in range(-1, 3):
                    for n in range(-1, 3):
                        x_index = min(max(x1 + n, 0), width - 1)
                        y_index = min(max(y1 + m, 0), height - 1)
                        coefficient = bicubic_kernel(dx - n) * bicubic_kernel(dy - m)
                        channel_value += coefficient * image[y_index, x_index, c]

                interpolated_pixel[c] = np.clip(int(channel_value), 0, 255)

            output_image[i, j] = interpolated_pixel
    print('\rbicubic is done \n')
    return output_image


def bicubic_kernel(t):
    if abs(t) <= 1:
        result = (1.5 * abs(t) ** 3 - 2.5 * abs(t) ** 2 + 1)
        return result
    elif 1 < abs(t) <= 2:
        result = (-0.5 * abs(t) ** 3 + 2.5 * abs(t) ** 2 - 4 * abs(t) + 2)
        return result
    else:
        return 0