import time
import cv2
import tools
import matplotlib.pyplot as plt

path = "./img/"


def cv2_equalize_hist(origin):
    img = origin.copy()
    img = cv2.equalizeHist(img)
    return img


school_img = cv2.imread(path + "school.png", cv2.IMREAD_GRAYSCALE)
baby_img = cv2.imread(path + "baby.png", cv2.IMREAD_GRAYSCALE)
hill_img = cv2.imread(path + "hill.jpg", cv2.IMREAD_GRAYSCALE)
tools.show_hist(school_img, "school img")
tools.show_hist(baby_img, "baby img")
tools.show_hist(hill_img, "hill img")

time_start = time.time()
school_img_my = tools.equalize_hist(school_img)
baby_img_my = tools.equalize_hist(baby_img)
hill_img_my = tools.equalize_hist(hill_img)
time_end = time.time()
tools.show_hist(school_img_my, "school img my")
tools.show_hist(baby_img_my, "baby img my")
tools.show_hist(hill_img_my, "hill img my")
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

