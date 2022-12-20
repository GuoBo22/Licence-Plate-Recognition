from tensorflow import keras
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import random
import pathlib
import cv2


def read_dataset():
    """
    读取图片集合存为数据集
    :return:
    """
    data_dir = pathlib.Path('car_datasets/chinese')
    image_count = len(list(data_dir.glob('*/*.jpg')))
    print('total image count:' + str(image_count))
    digit_class = []
    for item in data_dir.iterdir():
        digit_class.append(str(item).split('/')[-1])
    print(digit_class)

    train_images, train_labels, test_images, test_labels = [], [], [], []

    for digit_class_i in digit_class:
        digits = list(data_dir.glob(str(digit_class_i) + '/*.jpg'))
        train_number = int(len(digits) * 0.7)
        for i in range(len(digits)):
            img = cv2.imread(str(digits[i]), cv2.IMREAD_GRAYSCALE)
            if i <= train_number:
                train_images.append(img)
                train_labels.append(digit_class.index(digit_class_i))
            else:
                test_images.append(img)
                test_labels.append(digit_class.index(digit_class_i))

    train_images = np.array(train_images)
    train_labels = np.array(train_labels)
    test_images = np.array(test_images)
    test_labels = np.array(test_labels)

    # 数据集乱序
    index = [i for i in range(len(train_images))]
    random.shuffle(index)
    train_images = train_images[index]
    train_labels = train_labels[index]
    index = [i for i in range(len(test_images))]
    random.shuffle(index)
    test_images = test_images[index]
    test_labels = test_labels[index]

    print(train_images.shape)
    print(train_labels)
    print(test_images.shape)
    print(test_labels)

    return train_images, train_labels, test_images, test_labels


def one_hot(labels):
    """
    独热编码
    :param labels:
    :return:
    """
    onehot_labels = np.zeros(shape=[len(labels), 31])
    for i in range(len(labels)):
        index = labels[i]
        onehot_labels[i][index] = 1
    return onehot_labels


def cnn_net(input_shape):
    """
    构建cnn网络模型
    :param input_shape:
    :return:
    """
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(filters=32, kernel_size=3, strides=(1, 1),
                                  padding='same', activation=tf.nn.relu, input_shape=input_shape))
    model.add(keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    model.add(keras.layers.Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='same', activation=tf.nn.relu))
    model.add(keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    model.add(keras.layers.Dropout(0.236))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(units=128, activation=tf.nn.relu))
    model.add(keras.layers.Dropout(0.36))
    model.add(keras.layers.Dense(units=31, activation=tf.nn.softmax))
    print(model.summary())
    return model


def train_model(train_images, train_labels, test_images, test_labels):
    """
    训练模型
    :param train_images:
    :param train_labels:
    :param test_images:
    :param test_labels:
    :return:
    """
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    train_images = np.expand_dims(train_images, axis=3)
    test_images = np.expand_dims(test_images, axis=3)
    print(f'train_images: {train_images.shape}')
    print(f'test_images: {test_images.shape}')

    train_labels = one_hot(train_labels)
    test_labels = one_hot(test_labels)

    model = cnn_net(input_shape=(20, 20, 1))
    model.compile(optimizer=tf.optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x=train_images, y=train_labels, epochs=10)

    test_loss, test_acc = model.evaluate(x=test_images, y=test_labels)
    print('Test Accuracy %.2f' % test_acc)

    predictions = model.predict(test_images)
    y_pred, y_test = [], []
    TP, TN, FP, FN = np.zeros(31), np.zeros(31), np.zeros(31), np.zeros(31)
    for i in range(len(test_images)):
        target = np.argmax(predictions[i])
        y_pred.append(int(target))
        label = np.argmax(test_labels[i])
        y_test.append(int(label))
        if target == label:
            for j in range(31):
                if j == label:
                    TP[j] += 1
                else:
                    TN[j] += 1
        else:
            for j in range(31):
                if j == label:
                    FP[j] += 1
                else:
                    FN[j] += 1
    for i in range(31):
        print("------------" + str(i) + "------------")
        compute_matrics(TP[i], FP[i], TN[i], FN[i])

    model.save('model/chinese-model.h5')
    l = []
    for path in pathlib.Path('car_datasets/chinese').iterdir():
        l.append(str(path).split('/')[-1])
    plot_confusion_matrix(y_test, y_pred, np.array(l))


def compute_matrics(TP, FP, TN, FN):
    accuracy = (TP + TN) / (TP + FP + TN + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * (precision * recall) / (precision + recall)
    print("[accuracy] " + str(accuracy))
    print("[precision] " + str(precision))
    print("[recall] " + str(recall))
    print("[f1_score] " + str(f1_score))


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # print("Normalized confusion matrix")
    else:
        pass
        # print('Confusion matrix, without normalization')

    print(cm)

    plt.rcParams['font.sans-serif'] = ['Songti SC']
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots()

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    ax.set_ylim(len(classes) - 0.5, -0.5)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()

    plt.show()
    return ax


if __name__ == '__main__':
    train_images, train_labels, test_images, test_labels = read_dataset()
    train_model(train_images, train_labels, test_images, test_labels)
