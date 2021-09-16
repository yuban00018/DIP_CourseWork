# coding=utf-8
import cv2
import tools
import numpy as np


def concat_images_horizontal(image, words):
    for j in range(0, len(words)):
        temp = cv2.imread(path + str(words[j]) + ".jpg")
        # 计算高度差，因为是横向拼接需要高度一致
        padding_size = image.shape[0] - temp.shape[0]
        # 填充大小不一样的数组
        if padding_size > 0:
            temp = np.pad(temp, ((padding_size, 0), (0, 0), (0, 0)), 'constant',
                          constant_values=(
                              ([211, 223, 229], 0), (0, 0), (0, 0)))
        else:
            image = np.pad(image, ((-padding_size, 0), (0, 0), (0, 0)), 'constant',
                           constant_values=(
                               ([211, 223, 229], 0), (0, 0), (0, 0)))

        image = np.concatenate((image, temp), axis=1)
    return image


def concat_images_vertical(top, bottom):
    # 计算水平差
    padding_size1 = top.shape[1] - bottom.shape[1]
    # 填充大小不一样的数组
    if padding_size1 > 0:
        bottom = np.pad(bottom, ((0, 0), (int(padding_size1 / 2), padding_size1 - int(padding_size1 / 2)), (0, 0)),
                        'constant',
                        constant_values=(
                            (0, 0), ([211, 223, 229], [211, 223, 229]), (0, 0)))
    else:
        top = np.pad(top, ((0, 0), (int(-padding_size1 / 2), -padding_size1 + int(padding_size1 / 2)), (0, 0)),
                     'constant',
                     constant_values=(
                         (0, 0), ([211, 223, 229], [211, 223, 229]), (0, 0)))

    top = np.concatenate((top, bottom), axis=0)
    return top


path = "./img/"
# 读取文件并二值化
img = cv2.imread(path + "letter.jpg")
# 灰度图
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("gray scale", gray)
# 二值化
ret, thresh = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY)
cv2.imshow("thresh", thresh)

# MSER 最大稳定极值区域(文字区域定位)
mser = cv2.MSER_create()
regions, _ = mser.detectRegions(thresh)
# 凸包
# 凸包跟多边形逼近很像，只不过它是包围物体最外层的一个凸集，这个凸集是所有能包围这个物体的凸集的交集。
hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
boxes = []
for c in hulls:
    x, y, w, h = cv2.boundingRect(c)
    # 当得到对象轮廓后，可用boundingRect()得到包覆此轮廓的最小正矩形
    boxes.append([x, y, x + w, y + h, w * h])
    # w*h作为score，为了让NMS保留最大框

# 非极大值抑制，删除重叠框
boxes = tools.nms(np.array(boxes), 0.0001)

i = 0
for (x1, y1, x2, y2, _) in boxes:
    cv2.imwrite(path + str(i) + ".jpg", img[y1 - 2:y2 + 5, x1 - 5:x2 + 5])
    # 向img写入图片
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 1)
    # 给原图上框
    i += 1
cv2.imshow("boxes", img)

# 创建一个空格图片，颜色和背景一致
space = []
for i in range(75 * 60):
    # 填入背景颜色
    space.append([211, 223, 229])
space = np.reshape(space, (75, 60, 3)).astype("uint8")
cv2.imwrite(path + "36.jpg", space)

# 要打印的字符
letters = [4, 3, 8, 36, 27, 15, 12, 8, 36, 0, 16, 11]
numbers = [34, 18, 34, 25, 34, 21, 21, 25]

# 横向拼接
name = cv2.imread(path + str(letters[0]) + ".jpg")
del (letters[0])
name = concat_images_horizontal(name, letters)

my_id = cv2.imread(path + str(numbers[0]) + ".jpg")
del (numbers[0])
my_id = concat_images_horizontal(my_id, numbers)

# 竖向拼接
final = concat_images_vertical(my_id, space)
final = concat_images_vertical(final, name)

cv2.imshow("my name and my id", final)

cv2.waitKey(0)
cv2.destroyAllWindows()
