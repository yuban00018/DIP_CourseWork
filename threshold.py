import cv2
import tools
import numpy as np
path = "./img/"

rice_img = cv2.imread(path+"rice.tiff",cv2.IMREAD_GRAYSCALE)
finger_img = cv2.imread(path+"finger.tif",cv2.IMREAD_GRAYSCALE)
poly_img = cv2.imread(path+"poly.tif",cv2.IMREAD_GRAYSCALE)

rice_t = tools.basic_global_threshold(rice_img,5)
rice_mask = np.where(rice_img < rice_t)
rice_img[rice_mask] = 255
finger_t = tools.basic_global_threshold(finger_img,5)
finger_mask = np.where(finger_img > finger_t)
finger_img[finger_mask] = 255

cv2.imshow("rice",rice_img)
cv2.imshow("finger",finger_img)
cv2.imshow("poly",poly_img)
cv2.waitKey(0)
cv2.destroyAllWindows()