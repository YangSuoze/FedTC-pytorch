"""
生成训练集、测试集、验证集txt
"""

import os
import glob
import random


def get_training_txt():
    random.seed(2023)

    # data_path = "/home/FedTC/data/gatritis_data/"
    # WLI或LCI文件夹
    data_path = "/home/aaa/yangjie/COVID-19/COVID-19/COVID-19/"
    path = [
        "Pneumonia-Bacterial/*",
        "Pneumonia-Viral/*",
        "COVID-19/*",
        "Normal/*",
    ]

    data = [[] for _ in range(len(path))]
    for i in range(len(path)):
        data[i] += glob.glob(data_path + path[i])
        random.shuffle(data[i])
    # 划分训练集、测试集和验证集
    train_ratio, test_ratio, val_ratio = 0.7, 0.2, 0.1
    COVID_train, COVID_test, COVID_val = [], [], []
    for arr in data:
        COVID_train.append(arr[: int(len(arr) * train_ratio)])
        COVID_test.append(
            arr[
                int(len(arr) * train_ratio) : int(len(arr) * (train_ratio + test_ratio))
            ]
        )
        COVID_val.append(arr[int(len(arr) * (train_ratio + test_ratio)) :])
    # label dict
    COVID_train_dict, COVID_test_dict, COVID_val_dict = {}, {}, {}
    for index in range(len(COVID_train)):
        for item in COVID_train[index]:
            COVID_train_dict[item] = index
    for index in range(len(COVID_test)):
        for item in COVID_test[index]:
            COVID_test_dict[item] = index
    for index in range(len(COVID_val)):
        for item in COVID_val[index]:
            COVID_val_dict[item] = index

    # # 制作txt训练测试文件
    folder_path = "utils/data_Txt"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    else:
        print("文件夹已存在，无需创建。")
    with open("utils/data_Txt/COVID-19_train.txt", "w") as f:
        for key, val in COVID_train_dict.items():
            f.write(key + "," + str(val) + "\n")
    with open("utils/data_Txt/COVID-19_test.txt", "w") as f:
        for key, val in COVID_test_dict.items():
            f.write(key + "," + str(val) + "\n")
    with open("utils/data_Txt/COVID-19_val.txt", "w") as f:
        for key, val in COVID_val_dict.items():
            f.write(key + "," + str(val) + "\n")


if __name__ == "__main__":
    get_training_txt()
