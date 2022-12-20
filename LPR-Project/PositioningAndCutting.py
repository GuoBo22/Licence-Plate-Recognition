import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import DetectingLicense

from scipy import ndimage
from sympy import Limit

char_dict = {"京": 0, "沪": 1, "津": 2, "渝": 3, "冀": 4, "晋": 5, "蒙": 6, "辽": 7, "吉": 8, "黑": 9, "苏": 10,
             "浙": 11, "皖": 12, "闽": 13, "赣": 14, "鲁": 15, "豫": 16, "鄂": 17, "湘": 18, "粤": 19, "桂": 20,
             "琼": 21, "川": 22, "贵": 23, "云": 24, "藏": 25, "陕": 26, "甘": 27, "青": 28, "宁": 29, "新": 30,
             "0": 31, "1": 32, "2": 33, "3": 34, "4": 35, "5": 36, "6": 37, "7": 38, "8": 39, "9": 40,
             "A": 41, "B": 42, "C": 43, "D": 44, "E": 45, "F": 46, "G": 47, "H": 48, "J": 49, "K": 50,
             "L": 51, "M": 52, "N": 53, "P": 54, "Q": 55, "R": 56, "S": 57, "T": 58, "U": 59, "V": 60,
             "W": 61, "X": 62, "Y": 63, "Z": 64}


def new_label(old_label):
    provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙",
                 "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂",
                 "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学",
                 "O"]

    ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P',
           'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
           'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']

    car_code2 = ""
    for i, number in enumerate(old_label.split("_")):
        if i == 0:
            car_origin_number = provinces[int(number)]
        else:
            car_origin_number = ads[int(number)]
        car_code2 += str(car_origin_number)
    return car_code2


def WriteFileNameToTxt(dataset_path):
    """
    读取数据集内所有照片
    将照片的文件名按行写入txt文件
    """
    data = os.listdir(dataset_path)
    filename = "filename.txt"
    file = open(filename, 'w')
    for i in range(len(data)):
        s = str(data[i]).replace('[', '').replace(']', '')  # 去除[],这两行按数据不同，可以选择
        s = s.replace("'", '').replace(',', '') + '\n'  # 去除单引号，逗号，每行末尾追加换行符
        if '.DS_Store' in s:
            continue
        else:
            file.write(s)
    file.close()
    print("Successfully saved txt!")


def Limit(image, size):
    height, width, channel = image.shape
    # 设置权重
    weight = width / size
    # 计算输出图像的宽和高
    last_width = int(width / weight)
    last_height = int(height / weight)
    image = cv2.resize(image, (last_width, last_height))
    return image


def GetLicenseByPoints(dataset_path, txt_path, save_path):
    """
    基于CCPD数据集实现，通过图片文件名中的坐标信息提取车牌并存入save_path中
    """
    f = open(txt_path, encoding="utf-8")
    txt = []
    for line in f:
        txt.append(line.strip())
    for i in range(len(txt)):
        path_new = os.path.join(dataset_path, txt[i])
        img = cv2.imread(path_new)
        img_name = path_new
        iname = img_name.rsplit('/', 1)[-1].rsplit('.', 1)[0].split('-')
        old_label = iname[-3]
        old_label = new_label(old_label)

        # 图像变换
        point = [[int(eel) for eel in el.split('&')] for el in iname[3].split("_")]
        point = np.array(point, dtype=np.float32)
        pic = four_point_transform(img, point)
        # cv2.imshow('pic', pic)
        pic = Limit(pic, 600)
        imagename = save_path + "/" + str(old_label) + ".jpg"
        # cv2.imwrite(imagename, cropped)
        cv2.imencode('.jpg', pic)[1].tofile(imagename)


def GetLicenseByDetecting(dataset_path, txt_path, save_path):
    """
    通过识别的方式提取车牌，并存入save_path中
    """
    f = open(txt_path, encoding="utf-8")
    txt = []
    for line in f:
        txt.append(line.strip())
    for i in range(len(txt)):
        path_new = os.path.join(dataset_path, txt[i])
        img = cv2.imread(path_new)
        img_name = path_new
        iname = img_name.rsplit('/', 1)[-1].rsplit('.', 1)[0].split('-')
        old_label = iname[-3]
        old_label = new_label(old_label)
        licence = DetectingLicense.detect(img)
        licence = Limit(licence, 600)
        imagename = save_path + "/" + str(old_label) + ".jpg"
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        cv2.imencode('.jpg', licence)[1].tofile(imagename)


def order_points(pts):
    # 初始化坐标点
    rect = np.zeros((4, 2), dtype='float32')

    # 获取左上角和右下角坐标点
    s = pts.sum(axis=1)  # 每行像素值进行相加；若axis=0，每列像素值相加
    rect[0] = pts[np.argmin(s)]  # top_left,返回s首个最小值索引，eg.[1,0,2,0],返回值为1
    rect[2] = pts[np.argmax(s)]  # bottom_left,返回s首个最大值索引，eg.[1,0,2,0],返回值为2

    # 分别计算左上角和右下角的离散差值
    diff = np.diff(pts, axis=1)  # 第i+1列减第i列
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def four_point_transform(image, pts):
    """
    用于旋转图片倾斜角度
    :param image: 图片
    :param pts: points矩阵, 是四个顶点的坐标
    :return:
    """
    # 获取坐标点，并将它们分离开来
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # 计算新图片的宽度值，选取水平差值的最大值
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # 计算新图片的高度值，选取垂直差值的最大值
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # 构建新图片的4个坐标点,左上角为原点
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # 获取透视变换矩阵并应用它
    M = cv2.getPerspectiveTransform(rect, dst)
    # 进行透视变换
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # 返回变换后的结果
    return warped


def IsWhiteMore(binary):
    white = black = 0
    height, width = binary.shape
    # 遍历每个像素
    for i in range(height):
        for j in range(width):
            if binary[i, j] == 0:
                black += 1
            else:
                white += 1
    if white >= black:
        return True
    else:
        return False


# 统计白色像素点（分别统计每一行、每一列）
def White_Statistic(image):
    ptx = []  # 每行白色像素个数
    pty = []  # 每列白色像素个数
    height, width = image.shape
    # 逐行遍历
    for i in range(height):
        num = 0
        for j in range(width):
            if image[i][j] == 255:
                num = num + 1
        ptx.append(num)

    # 逐列遍历
    for i in range(width):
        num = 0
        for j in range(height):
            if image[j][i] == 255:
                num = num + 1
        pty.append(num)

    return ptx, pty


# 绘制直方图
def Draw_Hist(ptx, pty):
    # 依次得到各行、列
    rows, cols = len(ptx), len(pty)
    row = [i for i in range(rows)]
    col = [j for j in range(cols)]
    # 横向直方图
    plt.barh(row, ptx, color='black', height=1)
    #       纵    横
    plt.show()
    # 纵向直方图
    plt.bar(col, pty, color='black', width=1)
    #       横    纵
    plt.show()


def ManipulateImage(img):
    # 1、中值滤波
    mid = cv2.medianBlur(img, 5)
    # 2、灰度化
    gray = cv2.cvtColor(mid, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    blur = remove_upanddown_border(blur)
    # 3、二值化
    ret, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU)
    # 统一得到黑底白字
    if IsWhiteMore(binary):  # 白色部分多则为真，意味着背景是白色，需要黑底白字
        ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    # 4、膨胀（粘贴横向字符）
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 1))  # 横向连接字符
    dilate = cv2.dilate(binary, kernel)
    cv2.imshow('dilate', dilate)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # 5、统计各行各列白色像素个数（为了得到直方图横纵坐标）
    ptx, pty = White_Statistic(dilate)

    # 先根据直方图横向切割
    # 依次得到各行、列
    rows, cols = len(ptx), len(pty)
    row = [i for i in range(rows)]
    col = [j for j in range(cols)]

    h1, h2 = Cut_X(ptx, rows)
    cut_x = binary[h1:h2, :]
    cv2.imshow('cut_x', cut_x)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    ptx, pty = White_Statistic(cut_x)
    # Draw_Hist(ptx, pty)  # 绘制直方图
    Cut_Y(pty, cols, h1, h2, cut_x)


def find_waves(threshold, histogram):
    """ 根据设定的阈值和图片直方图，找出波峰，用于分隔字符 """
    up_point = -1  # 上升点
    is_peak = False
    if histogram[0] > threshold:
        up_point = 0
        is_peak = True
    wave_peaks = []
    for i, x in enumerate(histogram):
        if is_peak and x < threshold:
            if i - up_point > 2:
                is_peak = False
                wave_peaks.append((up_point, i))
        elif not is_peak and x >= threshold:
            is_peak = True
            up_point = i
    if is_peak and up_point != -1 and i - up_point > 4:
        wave_peaks.append((up_point, i))
    return wave_peaks


def remove_upanddown_border(img):
    """ 去除车牌上下无用的边缘部分，确定上下边界 """
    ret, plate_binary_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    row_histogram = np.sum(plate_binary_img, axis=1)  # 数组的每一行求和
    row_min = np.min(row_histogram)
    row_average = np.sum(row_histogram) / plate_binary_img.shape[0]
    row_threshold = (row_min + row_average) / 2
    wave_peaks = find_waves(row_threshold, row_histogram)
    # 挑选跨度最大的波峰
    wave_span = 0.0
    for wave_peak in wave_peaks:
        span = wave_peak[1] - wave_peak[0]
        if span > wave_span:
            wave_span = span
            selected_wave = wave_peak
    plate_binary_img = plate_binary_img[selected_wave[0]:selected_wave[1], :]
    # cv.imshow("plate_binary_img", plate_binary_img)
    return plate_binary_img


# 横向分割：上下边框
def Cut_X(ptx, rows):
    # 横向切割（分为上下两张图，分别找其波谷，确定顶和底）
    # 1、下半图波谷
    min, r = 600, 0
    for i in range(int(rows / 2)):
        if ptx[i] < min:
            min = ptx[i]
            r = i
    h1 = r  # 添加下行（作为顶）

    # 2、上半图波谷
    min, r = 600, 0
    for i in range(int(rows / 2), rows):
        if ptx[i] < min:
            min = ptx[i]
            r = i
    h2 = r  # 添加上行（作为底）
    return h1, h2


# 纵向分割：切割字符
def Cut_Y(pty, cols, h1, h2, binary):
    WIDTH = 90
    # 前谷 字符开始 字符结束
    w = 0
    w1 = 0
    w2 = 0
    begin = False  # 字符开始标记
    last = 14  # 上一次的值
    con = 0  # 计数
    # 纵向切割（正式切割字符）
    for j in range(40, int(cols)-20):
        if con >= 7:
            break
        # 1、前谷（前面的波谷）
        if pty[j] < 10 and begin == False:  # 前谷判断：像素数量<12
            last = pty[j]
            w = j
        # 2、字符开始（上升）
        elif last < 10 and 5 < pty[j]:
            last = pty[j]
            w1 = j
            begin = True
        # 3、字符结束
        elif pty[j] < 5 and begin == True:
            begin = False
            last = pty[j]
            w2 = j
            width = w2 - w1
            # 3-1、分割并显示（排除过小情况）
            if 40 < width < WIDTH + 3:  # 要排除掉干扰，又不能过滤掉字符”1“
                b_copy = binary.copy()
                b_copy = b_copy[h1:h2, w1:w2]
                cv2.imshow('binary%d-%d' % (count, con), b_copy)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                if not os.path.exists(f'car_characters/{count}'):
                    os.makedirs(f'car_characters/{count}')
                cv2.imwrite(f'car_characters/{count}/image{count}-{con}.jpg', b_copy)
                con += 1
            # 3-2、从多个贴合字符中提取单个字符
            elif width >= WIDTH + 3:
                # 统计贴合字符个数
                num = int(width / WIDTH + 0.5)  # 四舍五入
                for k in range(num):
                    # w1和w2坐标向后移（用w3、w4代替w1和w2）
                    w3 = w1 + k * WIDTH
                    w4 = w1 + (k + 1) * WIDTH
                    b_copy = binary.copy()
                    b_copy = b_copy[h1:h2, w3:w4]
                    # b_copy = cv2.resize(b_copy, (20, 20))
                    cv2.imshow('binary%d-%d' % (count, con), b_copy)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    if not os.path.exists(f'car_characters/{count}'):
                        os.makedirs(f'car_characters/{count}')
                    cv2.imwrite(f'car_characters/{count}/image{count}-{con}.jpg', b_copy)
                    con += 1

        # 4、分割尾部噪声（距离过远默认没有字符了）
        elif begin == False and (j - w2) > 30:

            break

    # 最后检查收尾情况
    if begin:
        w2 = 600
        b_copy = binary.copy()
        b_copy = b_copy[h1:h2, w1:w2]
        # b_copy = cv2.resize(b_copy, (20, 20))
        cv2.imshow('binary%d-%d' % (count, con), b_copy)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        if not os.path.exists(f'car_characters/{count}'):
            os.makedirs(f'car_characters/{count}')
        cv2.imwrite(f'car_characters/{count}/image{count}-{con}.jpg', b_copy)


def start(dataset_path, filename_path, save_path):
    global count
    count = 0
    WriteFileNameToTxt(dataset_path)
    # 图像路径

    # GetLicenseByPoints(dataset_path, filename_path, save_path)
    GetLicenseByDetecting(dataset_path, filename_path, save_path)

    images_list = os.listdir(save_path)
    if '.DS_Store' in images_list:
        images_list.remove('.DS_Store')  # 这是macOS下一个隐藏文件 自动生成的 要去掉 建议直接sudo删掉 不然txt文件也会报错呜呜
    for item in images_list:
        path = save_path + '/' + item
        print(path)
        imgae = cv2.imread(path)
        img = imgae.copy()
        ManipulateImage(img)
        count += 1
