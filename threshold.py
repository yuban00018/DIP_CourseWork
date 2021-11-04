import cv2
import tools
import numpy as np
path = "./img/"

rice_img = cv2.imread(path+"rice.tiff",cv2.IMREAD_GRAYSCALE)
finger_img = cv2.imread(path+"finger.tif",cv2.IMREAD_GRAYSCALE)
poly_img = cv2.imread(path+"poly.tif",cv2.IMREAD_GRAYSCALE)

tools.show_hist(rice_img,"rice hist")
tools.show_hist(finger_img,"finger hist")
tools.show_hist(poly_img,"poly hist")

cv2.imshow("rice",rice_img)
cv2.imshow("finger",finger_img)
cv2.imshow("poly",poly_img)

rice_t = tools.basic_global_threshold(rice_img,5)
rice_mask = np.where(rice_img < rice_t)
rice_mask2 = np.where(rice_img >= rice_t)
rice_img[rice_mask] = 255
rice_img[rice_mask2] = 0

finger_t = tools.basic_global_threshold(finger_img,5)
finger_mask = np.where(finger_img >= finger_t)
finger_mask2 = np.where(finger_img < finger_t)
finger_img[finger_mask] = 255
finger_img[finger_mask2] = 0

poly_t = tools.basic_global_threshold(poly_img,5)
poly_mask = np.where(poly_img < poly_t)
poly_mask2 = np.where(poly_img >= poly_t)
poly_img[poly_mask] = 0
poly_img[poly_mask2] = 255

cv2.imshow("rice global",rice_img)
cv2.imshow("finger global",finger_img)
cv2.imshow("poly global",poly_img)


rice_img = cv2.imread(path+"rice.tiff",cv2.IMREAD_GRAYSCALE)
finger_img = cv2.imread(path+"finger.tif",cv2.IMREAD_GRAYSCALE)
poly_img = cv2.imread(path+"poly.tif",cv2.IMREAD_GRAYSCALE)

rice_k = tools.o_tsu(rice_img)
rice_mask = np.where(rice_img >= rice_k)
rice_mask2 = np.where(rice_img < rice_k)
rice_img[rice_mask] = 0
rice_img[rice_mask2] = 255

finger_k = tools.o_tsu(finger_img)
finger_mask = np.where(finger_img < rice_k)
finger_mask2 = np.where(finger_img >= rice_k)
finger_img[finger_mask] = 0
finger_img[finger_mask2] = 255

poly_k = tools.o_tsu(poly_img)
poly_mask = np.where(poly_img < poly_k)
poly_mask2 = np.where(poly_img >= poly_k)
poly_img[poly_mask] = 0
poly_img[poly_mask2] = 255

cv2.imshow("poly o tsu",poly_img)
cv2.imshow("rice o tsu",rice_img)
cv2.imshow("finger o tsu",finger_img)

cv2.waitKey(0)
cv2.destroyAllWindows()