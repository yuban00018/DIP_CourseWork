import cv2

import tools

path = "./img/"

img_wuKong = cv2.imread(path + "wukong.jpg")
img_board1 = cv2.imread(path + "board1.jpg")
img_jp = cv2.imread(path + "jp.jpg")

points_dst = tools.get_dst_points(img_board1, shrink=3)
img_dst = tools.replace_convex_area(img_board1, img_wuKong, points_dst, shrink=3)

points_dst = tools.get_dst_points(img_dst, shrink=3)
img_dst = tools.replace_convex_area(img_dst, img_jp, points_dst, shrink=3)

# 原图太大了，按比例resize一下
cv2.namedWindow("dst", 0)
cv2.resizeWindow("dst", int(img_dst.shape[1] / 2), int(img_dst.shape[0] / 2))
cv2.imshow("dst", img_dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
