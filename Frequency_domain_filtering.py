import cv2
import matplotlib.pyplot as plt
import tools
import numpy as np

path = "./img/"

man_img = cv2.imread(path + "cameraman.tif", cv2.IMREAD_GRAYSCALE)
man_noise_img = cv2.imread(path + "stripy_cameraman.png", cv2.IMREAD_GRAYSCALE)

original, _ = tools.fourier(man_img)
tools.plt_show("Original graph", man_img)
tools.plt_show("Spectrum of the original graph", original)
tools.plt_show("Noised graph", man_noise_img)


# 创建mask
rows, cols = man_noise_img.shape
mask = np.zeros((rows, cols, 2), np.uint8)
half_row, half_col = int(rows / 2), int(cols / 2)
mask[:, :] = 1
mask[half_row - 40:half_row - 15, half_col - 15:half_col - 11] = 0
mask[half_row + 15:half_row + 40, half_col + 11:half_col + 15] = 00

dft, shift = tools.fourier(man_noise_img)
tools.plt_show("Spectrum of the noised graph", 20 * np.log(cv2.magnitude(shift[:, :, 0], shift[:, :, 1])))
shift = shift * mask
result = tools.ifourier(shift)

tools.plt_show("Spectrum of the processed noised graph", 20 * np.log(cv2.magnitude(shift[:, :, 0], shift[:, :, 1])))
tools.plt_show("Results of frequency domain filtering", result)
