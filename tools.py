import numpy as np
import cv2
import itertools


def gen_mean_kernel(size):
    kernel = np.zeros([size,size])
    for i in range(size):
        for j in range(size):
            kernel[i,j]=1
    return kernel/(size**2)


def gen_gaussian_kernel(kernel_size=3, sigma=0):
    kernel = np.zeros([kernel_size, kernel_size])
    center = kernel_size // 2
    # 2c^2
    s = 2 * (sigma ** 2)
    sum_val = 0
    for i in range(kernel_size):
        for j in range(kernel_size):
            # 到中心的距离
            x = i - center
            y = j - center
            kernel[i, j] = np.exp(-(x ** 2 + y ** 2) / s)
            sum_val += kernel[i, j]
    return kernel / sum_val


def median_filter(origin, kernel_size=3, stride=1):
    img = origin.copy()
    output = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if i + kernel_size > output.shape[0] or j + kernel_size > output.shape[1]:
                continue
            area = img[i:i + kernel_size, j:j + kernel_size].reshape(kernel_size * kernel_size)
            sorted_pixel = np.sort(area)
            mid = kernel_size*kernel_size/2
            output[i+1, j+1] = sorted_pixel[int(mid)]
    return output


def mean_filter(origin, kernel=np.array(np.array(
    [[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9, dtype=np.float32), stride=1):
    img = origin.copy()
    output = np.zeros((img.shape[0], img.shape[1]),dtype=np.uint8)
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            if i + kernel.shape[0] > output.shape[0] or j + kernel.shape[1] > output.shape[1]:
                continue
            if i+1 <= img.shape[0] and j+1 <= img.shape[1]:
                output[i+1, j+1] = (img[i:i + kernel.shape[0], j:j + kernel.shape[1]] * kernel).sum().astype(int)
    return output


def equalize_hist(origin):
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


def create_gamma_table(gamma):
    table = np.zeros(256)
    for i in range(256):
        table[i] = pow(i / 255, gamma) * 255
    return table


def gamma_transform_table(origin_img, table):
    img = origin_img.copy()
    rows, cols, ch = img.shape
    for x in range(rows):
        for y in range(cols):
            img[x, y] = table[img[x, y]]
    return img


def gamma_transform(origin_img, gamma):
    img = origin_img.copy()
    rows, cols, ch = img.shape
    for x in range(rows):
        for y in range(cols):
            img[x, y] = 255 * (pow(img[x, y] / 255, gamma))
    return img


def replace_convex_area(img_bg, img_fg, dst_points, shrink=1):
    rows, cols, ch = img_fg.shape
    # ！！！重要提醒！！！
    # 横坐标是cols，纵坐标是rows，填反了你的图像在映射后长宽就是反的！
    origin_points = np.float32(
        [
            [0, 0], [cols, 0], [cols, rows], [0, rows]
        ]
    )
    # 凸多边形区域填充黑色
    cv2.fillConvexPoly(img_bg, dst_points.astype(int), 0)
    # 寻找单应性矩阵,通过四个点求八个解加上一个约束即为
    H, status = cv2.findHomography(origin_points, dst_points)
    # 生成和img_bg大小一致的单应性矩阵变换后图像
    mask = cv2.warpPerspective(img_fg, H, (img_bg.shape[1], img_bg.shape[0]))
    cv2.namedWindow("mask", 0)
    cv2.resizeWindow("mask", int(mask.shape[1] / shrink), int(mask.shape[0] / shrink))
    cv2.imshow("mask", mask)
    # 把图像的遮罩覆盖在黑色填充区域
    result = img_bg + mask
    return result


'''
event：鼠标事件名称，通过该值可以获取鼠标进行的何种事件操作；
x, y：鼠标进行事件操作一瞬间，所在的坐标位置；
flags：指的是与event相关的实践中包含FLAG的事件；
userdata：鼠标回调函数触发时传递进来的参数。
原文链接： https://xie.infoq.cn/article/05a10c2375b2f060ae063282d
'''


def mouse_handler(event, x, y, flags, data):
    # 鼠标事件的回调函数，遵循标准格式
    if event == cv2.EVENT_LBUTTONDOWN:
        # 在相应位置绘制红点
        cv2.circle(data['im'], (x, y), 3, (0, 0, 255), -1)
        cv2.namedWindow("Image", 0)
        # 覆盖原图
        cv2.imshow("Image", data['im'])
        # 只存四个点
        if len(data['points']) < 4:
            data['points'].append([x, y])


def get_dst_points(img, shrink=1):
    # 原图的拷贝，这样就不会导致最终图片上面有点了
    data = {'im': img.copy(), 'points': []}
    cv2.namedWindow("Image", 0)
    cv2.resizeWindow("Image", int(img.shape[1] / shrink), int(img.shape[0] / shrink))
    cv2.imshow('Image', img)
    # 顺时针标记点，与origin_points保持一致才能正常映射
    cv2.setMouseCallback("Image", mouse_handler, data)
    cv2.waitKey(0)
    cv2.destroyWindow("Image")
    # array转换成np.float32
    points = np.vstack(data['points']).astype(float)
    return points


# 非极大抑制，默认threshold为0.5
def nms(boxes, threshold=0.5):
    # 获取方框数组
    box_array = np.array(boxes, dtype=np.float)
    # 批量截取x,y轴坐标
    x1 = box_array[:, 0]
    y1 = box_array[:, 1]
    x2 = box_array[:, 2]
    y2 = box_array[:, 3]
    # 此处的scores代表方框大小，越大得分越高
    scores = box_array[:, 4]
    # 以scores为排序依据的下标数组
    order = scores.argsort()[::-1]
    # 注意这是一个面积数组，里面有所有方框的大小
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # 最终的返回结果
    keep = []
    # 循环的作用就是不断地剔除出iou>threshold的方框
    while order.size > 0:
        i = order[0]
        # 每次保留最大框
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        # 计算交界处面积，不相交则为0
        inter = np.maximum(0.0, xx2 - xx1 + 1) * np.maximum(0.0, yy2 - yy1 + 1)
        # 计算交并比，注意iou和inter都是数组，代表所有方框和当前方框的比较
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        # 保留不相交的方框
        index = np.where(iou <= threshold)[0]
        order = order[index + 1]

    return np.array(box_array[keep]).astype("int")
