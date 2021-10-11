import tools
import numpy as np
import cv2

path = "./img/"

star_img = cv2.imread(path + "star.tif",cv2.IMREAD_GRAYSCALE)


des = tools.mean_filter(star_img, kernel=tools.gen_gaussian_kernel(10, 10))
# des = tools.mean_filter(star_img, kernel=tools.gen_mean_kernel(10))
cv2.imshow("after gaussian", des)
ret, des = cv2.threshold(des, 90, 255, cv2.THRESH_BINARY)

cv2.imshow("origin", star_img)
cv2.imshow("final", des)
cv2.waitKey(0)
cv2.destroyAllWindows()