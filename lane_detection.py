import cv2
import numpy as np

path = './video/'

video1 = cv2.VideoCapture(path + '1.mp4')
video2 = cv2.VideoCapture(path + '2.mp4')


def unique_lane(points, y_min, y_max):
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    # 曲线拟合，最小二乘法，返回斜率和截距
    fit = np.polyfit(y, x, 1)
    # 斜率和截距对应的函数
    fit_fn = np.poly1d(fit)
    # 计算这条直线在图像中最左侧的横坐标
    x_min = int(fit_fn(y_min))
    # 计算这条直线在图像中最右侧的横坐标
    x_max = int(fit_fn(y_max))
    return [(x_min, y_min), (x_max, y_max)]


def show_video(title, video):
    success, img = video.read()
    mask = np.zeros((img.shape[0], img.shape[1])).astype(np.uint8)
    trapezoidal = np.array([[img.shape[1] / 100 * 45, img.shape[0] / 1.65],
                            [img.shape[1] / 100 * 55, img.shape[0] / 1.65],
                            [img.shape[1] / 12 * 11, img.shape[0]],
                            [img.shape[1] / 12, img.shape[0]]], dtype=int)
    cv2.fillConvexPoly(mask, trapezoidal, 1)
    while True:
        success, frame = video.read()
        if not success:
            break
        # 灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 中值滤波滤去噪声
        after_blur = cv2.medianBlur(gray, 5)
        # Canny获得边界
        gray = cv2.Canny(after_blur, 50, 150)
        # 遮罩，获得ROI
        gray = cv2.bitwise_and(gray, gray, mask=mask)
        # 霍夫检测直线
        lines = cv2.HoughLinesP(gray, rho=1, theta=np.pi / 180, threshold=10, minLineLength=90, maxLineGap=150)
        points_left = []
        points_right = []
        # 直线归类，区分左右车道
        for line in lines:
            for x1, y1, x2, y2 in line:
                # 简单区分左右车道线，过中心线的就是右车道线
                if x1 < img.shape[1] / 2 and x2 < img.shape[1] / 2:
                    points_left.append([x1, y1])
                    points_left.append([x2, y2])
                else:
                    points_right.append([x1, y1])
                    points_right.append([x2, y2])
        # 在左侧多个车道线中找到唯一的线
        ([left_x1, left_y1], [left_x2, left_y2]) = unique_lane(points_left, int(img.shape[0] / 1.65), frame.shape[0])
        # 右侧车道线唯一线
        ([right_x1, right_y1], [right_x2, right_y2]) = unique_lane(points_right, int(img.shape[0] / 1.65),
                                                                   frame.shape[0])
        # 绘制车道线
        cv2.line(frame, (left_x1, left_y1), (left_x2, left_y2), (0, 0, 255), thickness=8)
        cv2.line(frame, (right_x1, right_y1), (right_x2, right_y2), (0, 0, 255), thickness=8)
        cv2.imshow(title, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


show_video("video1", video1)
show_video("video2", video2)
