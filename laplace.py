import cv2
import numpy as np
import time
import tools

path = "./img/"

moon_img = cv2.imread(path + "blurry_moon.tif", cv2.IMREAD_GRAYSCALE)

start = time.time()
moon_laplace = tools.laplacian(moon_img)
end = time.time()
print(end-start)

start = time.time()
moon_laplace_cv2 = cv2.Laplacian(moon_img, -1)

for i in range(moon_laplace_cv2.shape[0]):
    for j in range(moon_laplace_cv2.shape[1]):
        num = moon_img[i, j].astype(np.float32) - moon_laplace_cv2[i, j].astype(np.float32)
        if num < 0:
            moon_laplace_cv2[i, j] = 0
        elif num > 255:
            moon_laplace_cv2[i, j] = 255
        else:
            moon_laplace_cv2[i, j] = num

end = time.time()
print(end-start)

cv2.imshow("original", moon_img)
cv2.imshow("laplace my", moon_laplace)
cv2.imshow("laplace cv2", moon_laplace_cv2)
cv2.waitKey(0)
cv2.destroyAllWindows()
