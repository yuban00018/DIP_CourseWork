import cv2
import numpy as np
path = "./img/"

# 图片导入
cat_img = cv2.imread(path + "cat.jpg")
table_img = cv2.imread(path + "table.jpg")

# 切割图片，否则遮罩太大了
cat_img = cat_img[100:500, 100:450]
cat_img_copy = cat_img

# 遮罩在图片上的位移
y_shifting = 290
x_shifting = 460

rows, cols, channels = cat_img.shape
# 感兴趣区域，在这里是我们的桌子，通过手动偏移获得
roi = table_img[y_shifting:rows + y_shifting, x_shifting:cols + x_shifting]
# 获得猫的HSV图像，方便我们扣掉绿幕
cat_HSV = cv2.cvtColor(cat_img, cv2.COLOR_BGR2HSV)
# 把绿幕对应的位置在原图上全部变成白色
for i in range(cat_img.shape[0]):
    for j in range(cat_img.shape[1]):
        if 77 > cat_HSV[i, j, 0] > 40:
            cat_img[i, j] = [255, 255, 255]
# 生成灰度图像，然后用threshold形成一个遮罩
gray_cat = cv2.cvtColor(cat_img, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(gray_cat, 254, 255, cv2.THRESH_BINARY)
# 反向遮罩，用来扣取猫的图片
mask_inv = cv2.bitwise_not(mask)
# 图片和掩膜按位与
table_bg = cv2.bitwise_and(roi, roi, mask=mask)
cat_fg = cv2.bitwise_and(cat_img, cat_img, mask=mask_inv)
# 黑色像素按位与就变成另一张图片的像素，这张就是合成有猫的ROI区域
dst = cv2.add(table_bg, cat_fg)
# 图片的同一区域直接覆盖上去
table_img[y_shifting:rows + y_shifting, x_shifting:cols + x_shifting] = dst
# 显示结果
print(mask)
cv2.imshow("mask",mask)
cv2.imshow("table mask",table_bg)
cv2.imshow("cat mask",cat_fg)
cv2.imshow("ROI with cat",dst)
cv2.imshow("cat on the table", table_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
