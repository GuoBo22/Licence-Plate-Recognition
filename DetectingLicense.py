# -*- coding: utf-8 -*-
import cv2
import numpy as np


def Process(img):
    # 高斯平滑
    gaussian = cv2.GaussianBlur(img, (3, 3), 0, 0, cv2.BORDER_DEFAULT)
    cv2.imshow('test', gaussian)
    cv2.waitKey(0)
    # 中值滤波
    median = cv2.medianBlur(gaussian, 5)
    # Sobel算子
    # 梯度方向: x
    sobel = cv2.Sobel(median, cv2.CV_8U, 1, 0, ksize=3)
    # 利用Sobel方法可以进行sobel边缘检测

    ret, binary = cv2.threshold(sobel, 170, 255, cv2.THRESH_BINARY)
    # 灰度值小于175的点置0，灰度值大于175的点置255

    # 膨胀和腐蚀操作的核函数
    element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 7))
    # 第一个参数定义结构元素，如椭圆（MORPH_ELLIPSE）、交叉形结构（MORPH_CROSS）和矩形（MORPH_RECT）
    # 第二个参数就是指和函数的size，9×1

    # 膨胀一次，让轮廓突出
    dilation = cv2.dilate(binary, element2, iterations=1)
    # 腐蚀一次，去掉细小杂点
    erosion = cv2.erode(dilation, element1, iterations=1)
    # 再次膨胀，让轮廓更明显
    dilation2 = cv2.dilate(erosion, element2, iterations=3)
    cv2.imshow('test', dilation2)
    cv2.waitKey(0)

    return dilation2


def GetRegion(img):
    regions = []
    # 查找轮廓
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 筛选面积小的
    for contour in contours:
        # 计算该轮廓的面积
        area = cv2.contourArea(contour)
        # 面积小的都筛选掉
        if area < 4000:
            continue
        # 轮廓近似，作用很小
        epsilon = 1e-3 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        # epsilon，是从轮廓到近似轮廓的最大距离。是一个准确率参数，好的epsilon的选择可以得到正确的输出。True决定曲线是否闭合。
        # 找到最小的矩形，该矩形可能有方向
        rect = cv2.minAreaRect(approx)
        # box是四个点的坐标
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        # 计算高和宽
        height = abs(box[0][1] - box[2][1])
        width = abs(box[0][0] - box[2][0])
        # 车牌正常情况下长高比在2-5之间
        ratio = float(width) / float(height)
        if 4.5 > ratio > 2.8:
            regions.append(box)
    return regions


def detect(img):
    # 灰度化
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 预处理及形态学处理，得到可以查找矩形的图片
    prc = Process(gray)
    # 得到车牌轮廓
    regions = GetRegion(prc)
    print('[INFO]:Detect %d license plates' % len(regions))
    # 用绿线画出这些找到的轮廓
    for box in regions:
        cv2.drawContours(img, [box], 0, (0, 255, 0), 2)
        # 五个输入参数：原始图像，轮廓，轮廓的索引（当设置为-1时，绘制所有轮廓），画笔颜色，画笔大小
        # 一个返回值：返回绘制了轮廓的图像
        ys = [box[0, 1], box[1, 1], box[2, 1], box[3, 1]]
        xs = [box[0, 0], box[1, 0], box[2, 0], box[3, 0]]
        ys_sorted_index = np.argsort(ys)
        xs_sorted_index = np.argsort(xs)
        x1 = box[xs_sorted_index[0], 0]
        x2 = box[xs_sorted_index[3], 0]
        y1 = box[ys_sorted_index[0], 1]
        y2 = box[ys_sorted_index[3], 1]
        img_org2 = img.copy()
        img_plate = img_org2[y1:y2, x1:x2]

    return img_plate


if __name__ == '__main__':
    # 输入的参数为图片的路径
    img = cv2.imread(
        '../../../Documents/大三上/人工智能导论/LPR_Project/photos/0325-90_90-332&428_665&526-665&529_335&528_346&415_676&416-0_0_15_12_26_30_32-144-148.jpg')
    detect(img)
