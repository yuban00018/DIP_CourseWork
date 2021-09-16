import numpy as np

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
