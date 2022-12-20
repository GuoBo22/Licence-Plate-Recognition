import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pathlib
import cv2

def resize_keep_aspectratio(image_src, dst_size):
    src_h, src_w = image_src.shape[:2]
    # print(src_h, src_w)
    dst_h, dst_w = dst_size

    # 判断应该按哪个边做等比缩放
    h = dst_w * (float(src_h) / src_w)  # 按照ｗ做等比缩放
    w = dst_h * (float(src_w) / src_h)  # 按照h做等比缩放

    h = int(h)
    w = int(w)

    if h <= dst_h:
        image_dst = cv2.resize(image_src, (dst_w, int(h)))
    else:
        image_dst = cv2.resize(image_src, (int(w), dst_h))

    h_, w_ = image_dst.shape[:2]
    # print(h_, w_)

    top = int((dst_h - h_) / 2)
    down = int((dst_h - h_ + 1) / 2)
    left = int((dst_w - w_) / 2)
    right = int((dst_w - w_ + 1) / 2)

    value = [0, 0, 0]
    borderType = cv2.BORDER_CONSTANT
    # print(top, down, left, right)
    image_dst = cv2.copyMakeBorder(image_dst, top, down, left, right, borderType, None, value)

    return image_dst


def predict_chinese(pic_path):
    chinese = ['吉', '宁', '新', '青', '晋', '陕', '赣', '川', '沪', '苏', '豫', '贵', '黑', '辽', '云', '渝', '湘',
               '藏', '琼', '甘', '蒙', '闽', '桂', '皖', '粤', '浙', '津', '鲁', '鄂', '冀', '京']
    target_shape = (20, 20)
    img = cv2.imread(pic_path, cv2.IMREAD_GRAYSCALE)
    # img = remove_upanddown_border(img)
    # img = cv2.resize(img, target_shape)
    img = resize_keep_aspectratio(img, target_shape)
    plt.imshow(img, cmap='binary')
    plt.show()

    # im2arr = 255 - img
    im2arr = img.astype('float32') / 255
    im2arr = im2arr.reshape((1, 20, 20, 1))

    model = tf.keras.models.load_model('model/chinese-model.h5')
    y_pred = np.argmax(model.predict(im2arr))
    print(chinese[y_pred])


def predict_dight_and_letter(pic_path):
    dight_letter = ['R', 'U', '9', '0', '7', 'I', 'N', 'G', '6', 'Z', '1', '8', 'T', 'S', 'A', 'F', 'O', 'H', 'M', 'J',
                    'C', 'D', 'V', 'Q', '4', 'X', '3', 'E', 'B', 'K', 'L', '2', 'Y', '5', 'P', 'W']
    target_shape = (20, 20)
    img = cv2.imread(pic_path, cv2.IMREAD_GRAYSCALE)
    # img = remove_upanddown_border(img)
    # img = cv2.resize(img, target_shape)
    img = resize_keep_aspectratio(img, target_shape)
    plt.imshow(img, cmap='binary')
    plt.show()

    # im2arr = 255 - img
    im2arr = img.astype('float32') / 255
    im2arr = im2arr.reshape((1, 20, 20, 1))
    print(im2arr.shape)

    model = tf.keras.models.load_model('model/digit_letters-model.h5')
    y_pred = np.argmax(model.predict(im2arr))
    print(dight_letter[y_pred])


def predict_lpr(character_dir):
    chinese = ['吉', '宁', '新', '青', '晋', '陕', '赣', '川', '沪', '苏', '豫', '贵', '黑', '辽', '云', '渝', '湘',
               '藏', '琼', '甘', '蒙', '闽', '桂', '皖', '粤', '浙', '津', '鲁', '鄂', '冀', '京']
    dight_letter = ['R', 'U', '9', '0', '7', 'I', 'N', 'G', '6', 'Z', '1', '8', 'T', 'S', 'A', 'F', 'O', 'H', 'M', 'J',
                    'C', 'D', 'V', 'Q', '4', 'X', '3', 'E', 'B', 'K', 'L', '2', 'Y', '5', 'P', 'W']
    path = pathlib.Path(character_dir)
    model_chinese = tf.keras.models.load_model('model/chinese-model.h5')
    model_digit_letter = tf.keras.models.load_model('model/digit_letters-model.h5')
    target_shape = (20, 20)
    result = ['', '', '', '', '', '', '']

    for character in path.iterdir():
        img = cv2.imread(str(character), cv2.IMREAD_GRAYSCALE)
        img = resize_keep_aspectratio(img, target_shape)
        plt.imshow(img, cmap='binary')
        plt.show()
        im2arr = img.astype('float32') / 255
        im2arr = im2arr.reshape((1, 20, 20, 1))
        if character.name[-5] == '0':
            result[0] = chinese[np.argmax(model_chinese.predict(im2arr))]
        else:
            result[int(character.name[-5])] = dight_letter[np.argmax(model_digit_letter.predict(im2arr))]
    result_lpr = ''
    for ch in result:
        result_lpr += ch
    return result_lpr


if __name__ == '__main__':
    # predict_chinese('test/car_characters/image0-0.jpg')
    # predict_dight_and_letter('test/car_characters_2/image1-1.jpg')
    lpr = predict_lpr('test/car_characters_3/2')
    print(lpr)
