import numpy as np
import cv2
import matplotlib.pyplot as plt
import itertools


def basic_global_threshold(img, delta):
    copy = img.copy()
    t = int(copy.sum() / copy.shape[0] / copy.shape[1])
    while True:
        g1 = np.where(copy > t)
        g2 = np.where(copy <= t)
        m1 = copy[g1].sum() / len(g1[0])
        m2 = copy[g2].sum() / len(g2[0])
        t1 = int(0.5 * (m1 + m2))
        if abs(t1 - t) < delta:
            t = t1
            break
        else:
            t = t1
    return t


def show_hist(img, name):
    plt.figure(name)
    plt.title(name)
    arr = img.flatten()
    plt.hist(arr, bins=256, facecolor='blue', alpha=0.75, density=True)
    plt.show()


def cvt_hsi2bgr(origin):
    output = np.zeros(origin.shape)
    for x in range(origin.shape[0]):
        for y in range(origin.shape[1]):
            h, s, i = origin[x][y]
            h, s, i = h * 360, s, i / 255
            b, g, r = 0, 0, 0
            # RG扇区
            if 0 <= h < 120:
                b = i * (1 - s)
                r = i * (1 + (s * np.cos(np.radians(h)) / np.cos(np.radians(60 - h))))
                g = 3 * i - (b + r)
            # GB扇区
            elif 120 <= h < 240:
                h -= 120
                r = i * (1 - s)
                g = i * (1 + (s * np.cos(np.radians(h)) / np.cos(np.radians(60 - h))))
                b = 3 * i - (r + g)
            # BR扇区
            elif 240 <= h < 360:
                h -= 240
                g = i * (1 - s)
                b = i * (1 + (s * np.cos(np.radians(h)) / np.cos(np.radians(60 - h))))
                r = 3 * i - (g + b)
            # 还原出来的值可能会超范围，需要额外判定
            b = b * 255
            g = g * 255
            r = r * 255
            if b > 255:
                b = 255
            if g > 255:
                g = 255
            if r > 255:
                r = 255
            if b < 0:
                b = 0
            if g < 0:
                g = 0
            if r < 0:
                r = 0
            output[x][y] = b, g, r
    return output.astype(np.uint8)


def cvt_bgr2hsi(origin):
    with np.errstate(divide='ignore', invalid='ignore'):
        bgr = np.int32(cv2.split(origin))

        blue = bgr[0]
        green = bgr[1]
        red = bgr[2]

        intensity = np.divide(blue + green + red, 3)

        minimum = np.minimum(np.minimum(red, green), blue)
        # 饱和度
        saturation = 1 - 3 * np.divide(minimum, red + green + blue)

        sqrt_result = np.sqrt(((red - green) * (red - green)) + ((red - blue) * (green - blue)))

        if (green >= blue).any():
            hue = np.arccos((1 / 2 * ((red - green) + (red - blue)) / sqrt_result))
        else:
            hue = 2 * np.pi - np.arccos((1 / 2 * ((red - green) + (red - blue)) / sqrt_result))

        hue = hue / (2 * np.pi)

        hsi = cv2.merge((hue, saturation, intensity))
        return hsi


def inverse_fourier(dft_shift):
    return cv2.idft(np.fft.ifftshift(dft_shift), flags=cv2.DFT_REAL_OUTPUT)


def plt_show(title, img, cmap="gray"):
    plt.imshow(img, cmap="gray")
    plt.title(title), plt.xticks([]), plt.yticks([])
    plt.show()


def fourier(img):
    # 离散傅里叶变换
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    # 离散傅里叶变换后如果想让直流分量在输出图像的中心，需要将结果沿两个方向平移
    dft_shift = np.fft.fftshift(dft)
    # 构建振幅的公式
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
    return magnitude_spectrum, dft_shift


def sobel(origin, x, y):
    img = origin.copy()
    output = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            if i + kernel_x.shape[0] > output.shape[0] or j + kernel_x.shape[1] > output.shape[1]:
                continue
            if i + 1 <= img.shape[0] and j + 1 <= img.shape[1]:
                num = (img[i:i + kernel_x.shape[0], j:j + kernel_x.shape[1]] * kernel_x * x +
                       img[i:i + kernel_x.shape[0], j:j + kernel_x.shape[1]] * kernel_y * y) \
                    .sum().astype(int)
                if num <= 0:
                    output[i + 1, j + 1] = 0
                elif num >= 255:
                    output[i + 1, j + 1] = 255
                else:
                    output[i + 1, j + 1] = num
    return output


def laplacian(origin, kernel=np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]), stride=1):
    img = origin.copy()
    output = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            if i + kernel.shape[0] > output.shape[0] or j + kernel.shape[1] > output.shape[1]:
                continue
            if i + 1 <= img.shape[0] and j + 1 <= img.shape[1]:
                num = (img[i:i + kernel.shape[0], j:j + kernel.shape[1]] * kernel).sum().astype(int)
                if num <= 0:
                    output[i + 1, j + 1] = 0
                elif num >= 255:
                    output[i + 1, j + 1] = 255
                else:
                    output[i + 1, j + 1] = num
    return output


def gen_mean_kernel(size):
    kernel = np.zeros([size, size])
    for i in range(size):
        for j in range(size):
            kernel[i, j] = 1
    return kernel / (size ** 2)


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
            mid = kernel_size * kernel_size / 2
            output[i + 1, j + 1] = sorted_pixel[int(mid)]
    return output


def mean_filter(origin, kernel=np.array(np.array(
    [[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9, dtype=np.float32), stride=1):
    img = origin.copy()
    output = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            if i + kernel.shape[0] > output.shape[0] or j + kernel.shape[1] > output.shape[1]:
                continue
            if i + 1 <= img.shape[0] and j + 1 <= img.shape[1]:
                output[i + 1, j + 1] = (img[i:i + kernel.shape[0], j:j + kernel.shape[1]] * kernel).sum().astype(int)
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
