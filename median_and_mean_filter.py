import cv2
import numpy as np

import tools

path = "./img/"

space_img = cv2.imread(path + "space.png",cv2.IMREAD_GRAYSCALE)
mona_img = cv2.imread(path + "mona.png",cv2.IMREAD_GRAYSCALE)

input = np.array([[1, 1, 1],
                  [1, 255, 1],
                  [1, 1, 1]])
space_mean_filter = tools.mean_filter(space_img)
mona_mean_filter = tools.mean_filter(mona_img)
space_mean_filter_cv2 = cv2.blur(space_img,(3,3))
mona_mean_filter_cv2 = cv2.blur(mona_img,(3,3))

space_median_filter = tools.median_filter(space_img)
mona_median_filter = tools.median_filter(mona_img)
space_median_filter_cv2 = cv2.medianBlur(space_img,3)
mona_median_filter_cv2 = cv2.medianBlur(mona_img,3)

print(space_median_filter)

cv2.imshow("mona_original", mona_img)
cv2.imshow("mona_mean_filter", mona_mean_filter)
cv2.imshow("mona_median_filter",mona_median_filter)
cv2.imshow("cv2 mona mean filter",mona_mean_filter_cv2)
cv2.imshow("cv2 mona median filter",mona_median_filter_cv2)


cv2.imshow("space_original",space_img)
cv2.imshow("space_mean_filter", space_mean_filter)
cv2.imshow("space_median_filter",space_median_filter)
cv2.imshow("cv2 space mean filter",space_mean_filter_cv2)
cv2.imshow("cv2 space median filter",space_median_filter_cv2)


cv2.waitKey(0)
cv2.destroyAllWindows()
