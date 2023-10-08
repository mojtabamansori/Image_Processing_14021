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
from function_bicubic import bicubic_interpolation
from function_nn import upscale_nearest_neighbor
from function_bilinear import bilinear_interpolation
from function_bicubic_rotation import rotate_bicubic

# data section
original_image = np.array(cv2.imread("test_images/cameraman.tif"))
upscale_factor = 1.005
angle = 135
new_x = int(int(original_image.shape[0]) * upscale_factor)
new_y = int(int(original_image.shape[1]) * upscale_factor)

# up_sample section
up_scaled_data_cubic = bicubic_interpolation(original_image, (new_x,new_y))
up_scaled_data_nn = upscale_nearest_neighbor(original_image, upscale_factor)
up_scaled_data_linear = bilinear_interpolation(original_image, upscale_factor)
rotate_image = rotate_bicubic(original_image, angle)

# save section
cv2.imwrite(f"original_image.jpg", original_image)
cv2.imwrite(f"up_scaled_image_bicubic_{new_x}.jpg", up_scaled_data_cubic)
cv2.imwrite(f"up_scaled_image_nn_{new_x}.jpg", up_scaled_data_nn)
cv2.imwrite(f"up_scaled_image_linear_{new_x}.jpg", up_scaled_data_linear)
cv2.imwrite(f"rotate_image_{angle}.jpg", rotate_image)


# plot section
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 10), sharex=True, sharey=True)
ax[0, 0].imshow(original_image, cmap='gray', vmin=0, vmax=255)
ax[0, 0].set_title('Original Image')
ax[0, 1].imshow(up_scaled_data_linear, cmap='gray', vmin=0, vmax=255)
ax[0, 1].set_title(f'Up_scaled Image, bi linear')
ax[1, 0].imshow(up_scaled_data_nn, cmap='gray', vmin=0, vmax=255)
ax[1, 0].set_title(f'Up_scaled Image, nn')
ax[1, 1].imshow(up_scaled_data_cubic, cmap='gray', vmin=0, vmax=255)
ax[1, 1].set_title(f'Up_scaled Image, Bi-cubic')
plt.show()

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10), sharex=True, sharey=True)
ax.imshow(rotate_image, cmap='gray', vmin=0, vmax=255)
ax.set_title(f'Rotated Image')  # عنوان شکل دوم
plt.show()
