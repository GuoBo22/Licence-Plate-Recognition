from VLPRS import modelNumber, modelChinese, PositioningAndCutting, predict
import pathlib

if __name__ == '__main__':
    # 读取汉字数据集
    train_images, train_labels, test_images, test_labels = modelChinese.read_dataset()
    # 训练汉字模型
    modelChinese.train_model(train_images, train_labels, test_images, test_labels)
    # 读取字符数据集
    train_images, train_labels, test_images, test_labels = modelNumber.read_dataset()
    # 训练字符模型
    modelNumber.train_model(train_images, train_labels, test_images, test_labels)

    dataset_path = 'photos'  # 数据集路径
    filename_path = 'filename.txt'  # 文件名txt文件路径
    save_path = 'images'  # 切割出的车牌图片存储的文件夹路径

    # 对目标图片进行车牌定位并切割字符
    PositioningAndCutting.start(dataset_path=dataset_path, filename_path=filename_path, save_path=save_path)

    for lpr in pathlib.Path('car_characters').iterdir():
        result = predict.predict_lpr(str(lpr))
        print(result)
