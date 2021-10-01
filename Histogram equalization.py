import itertools
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt

path = "./img/"


def cv2_equalize_hist(origin):
    img = origin.copy()
    img = cv2.equalizeHist(img)
    return img


def my_equalize_hist(origin):
    r_k = np.zeros(256)
    img = origin.copy()
    rows, cols = img.shape
    for x in range(rows):
        for y in range(cols):
            # 统计r
            r_k[img[x, y]] += 1
    # 一定要注意，取整要在最外面进行，否则小数点后面的数差距会非常大
    lut = np.array(list(itertools.accumulate(list(np.array(r_k * 255 / (rows * cols)))))).astype(np.uint8)
    for x in range(rows):
        for y in range(cols):
            img[x, y] = lut[img[x, y]]
    return img


def show_hist(img, name):
    plt.figure(name)
    plt.title(name)
    arr = img.flatten()
    # print(name, arr)
    plt.hist(arr, bins=256, facecolor='blue', alpha=0.75, density=True)
    plt.show()


school_img = cv2.imread(path + "school.png", cv2.IMREAD_GRAYSCALE)
baby_img = cv2.imread(path + "baby.png", cv2.IMREAD_GRAYSCALE)
hill_img = cv2.imread(path + "hill.jpg", cv2.IMREAD_GRAYSCALE)
show_hist(school_img, "school img")
show_hist(baby_img, "baby img")
show_hist(hill_img, "hill img")

time_start = time.time()
school_img_my = my_equalize_hist(school_img)
baby_img_my = my_equalize_hist(baby_img)
hill_img_my = my_equalize_hist(hill_img)
time_end = time.time()
show_hist(school_img_my, "school img my")
show_hist(baby_img_my, "baby img my")
show_hist(hill_img_my, "hill img my")
print("my equalize hist: ", time_end - time_start, "s")

time_start = time.time()
school_img_cv2 = cv2_equalize_hist(school_img)
baby_img_cv2 = cv2_equalize_hist(baby_img)
hill_img_cv2 = cv2_equalize_hist(hill_img)
time_end = time.time()
show_hist(school_img_cv2, "school img cv2")
show_hist(baby_img_cv2, "baby img cv2")
show_hist(hill_img_cv2, "hill img cv2")
print("cv2 equalize hist: ", time_end - time_start, "s")


cv2.imshow("school_origin", school_img)
cv2.imshow("baby_origin", baby_img)
cv2.imshow("hill_origin", hill_img)


cv2.imshow("school_my", school_img_my)
cv2.imshow("baby_my", baby_img_my)
cv2.imshow("hill_my", hill_img_my)


cv2.imshow("school_cv2", school_img_cv2)
cv2.imshow("baby_cv2", baby_img_cv2)
cv2.imshow("hill_cv2", hill_img_cv2)
cv2.waitKey(0)
cv2.destroyAllWindows()

