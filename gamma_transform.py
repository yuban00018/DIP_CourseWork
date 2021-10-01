import cv2
import time
import tools
import numpy as np

path = "./img/"

light_img = cv2.imread(path + "light.tif")
dark_img = cv2.imread(path + "dark.png")

time_start = time.time()
light_img_gamma = tools.gamma_transform(light_img, gamma=5)
dark_img_gamma = tools.gamma_transform(dark_img, gamma=0.4)
time_end = time.time()
print("常规gamma矫正所用时间：", time_end - time_start, "s")

time_start = time.time()
light_img_gamma = tools.gamma_transform_table(light_img, tools.create_gamma_table(5))
dark_img_gamma = tools.gamma_transform_table(dark_img, tools.create_gamma_table(0.4))
time_end = time.time()
print("查表(含建表时间）gamma矫正所用时间：", time_end - time_start, "s")

time_start = time.time()
light_img_gamma = cv2.LUT(light_img, np.round(tools.create_gamma_table(5)).astype(np.uint8))
dark_img_gamma = cv2.LUT(dark_img, np.round(tools.create_gamma_table(0.4)).astype(np.uint8))
time_end = time.time()
print("cv2库LUT矫正所用时间：", time_end - time_start, "s")

cv2.imshow("light origin", light_img)
cv2.imshow("dark origin", dark_img)
cv2.imshow("light gamma_transform", light_img_gamma)
cv2.imshow("dark gamma_transform", dark_img_gamma)
cv2.waitKey(0)
cv2.destroyAllWindows()
