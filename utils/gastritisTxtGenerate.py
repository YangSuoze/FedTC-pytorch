'''
生成训练集、测试集、验证集txt
'''
import os
import glob
import random


def get_training_txt():
    random.seed(2023)

    # data_path = "/home/FedTC/data/gatritis_data/"
    # WLI或LCI文件夹
    data_path = "/home/bigspace/huaxi_data/*I/"
    path = [
        "IM/贲门/*.TIF",
        "GA/贲门/*.TIF",
        "Normal/贲门/*.TIF",
        
        "IM/胃底/*.TIF",
        "GA/胃底/*.TIF",
        "Normal/胃底/*.TIF",
        
        "IM/胃窦/*.TIF",
        "GA/胃窦/*.TIF",
        "Normal/胃窦/*.TIF",
        
        "IM/胃角/*.TIF",
        "GA/胃角/*.TIF",
        "Normal/胃角/*.TIF",
        
        "IM/胃体/*.TIF",
        "GA/胃体/*.TIF",
        "Normal/胃体/*.TIF",
    ]
    data = [[] for _ in range(len(path))]
    for i in range(len(path)):
        data[i] += glob.glob(data_path + path[i])
        random.shuffle(data[i])
    d_IM, d_GA, d_Normal = [], [], []
    for i in range(0, len(path), 3):
        d_IM.append(data[i])
        d_GA.append(data[i + 1])
        d_Normal.append(data[i + 2])
    data_IM_problem = d_IM + d_Normal
    data_GA_problem = d_GA + d_Normal

    # 划分训练集、测试集和验证集
    train_ratio,test_ratio,val_ratio = 0.7,0.2,0.1
    IM_train, IM_test, IM_val = [], [], []
    for i in data_IM_problem:
        IM_train.append(i[: int(len(i) * train_ratio)])
        IM_test.append(i[int(len(i) * train_ratio) : int(len(i) * (train_ratio+test_ratio))])
        IM_val.append(i[int(len(i) * (train_ratio+test_ratio)) :])
    # label dict
    IM_train_dict, IM_test_dict, IM_val_dict = {}, {}, {}
    for index in range(len(IM_train)):
        for item in IM_train[index]:
            IM_train_dict[item] = index
    for index in range(len(IM_test)):
        for item in IM_test[index]:
            IM_test_dict[item] = index
    for index in range(len(IM_val)):
        for item in IM_val[index]:
            IM_val_dict[item] = index

    GA_train, GA_test, GA_val = [], [], []
    for i in data_GA_problem:
        GA_train.append(i[: int(len(i) * 0.7)])
        GA_test.append(i[int(len(i) * 0.7) : int(len(i) * 0.9)])
        GA_val.append(i[int(len(i) * 0.9) :])
    GA_train_dict, GA_test_dict, GA_val_dict = {}, {}, {}
    for index in range(len(GA_train)):
        for item in GA_train[index]:
            GA_train_dict[item] = index
    for index in range(len(GA_test)):
        for item in GA_test[index]:
            GA_test_dict[item] = index
    for index in range(len(GA_val)):
        for item in GA_val[index]:
            GA_val_dict[item] = index
    # # 制作txt训练测试文件
    folder_path = 'utils/data_Txt'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    else:
        print("文件夹已存在，无需创建。")
    with open("utils/data_Txt/IM_train.txt", "w") as f:
        for key, val in IM_train_dict.items():
            f.write(key + "," + str(val) + "\n")
    with open("utils/data_Txt/IM_test.txt", "w") as f:
        for key, val in IM_test_dict.items():
            f.write(key + "," + str(val) + "\n")
    with open("utils/data_Txt/IM_val.txt", "w") as f:
        for key, val in IM_val_dict.items():
            f.write(key + "," + str(val) + "\n")

    with open("utils/data_Txt/GA_train.txt", "w") as f:
        for key, val in GA_train_dict.items():
            f.write(key + "," + str(val) + "\n")
    with open("utils/data_Txt/GA_test.txt", "w") as f:
        for key, val in GA_test_dict.items():
            f.write(key + "," + str(val) + "\n")
    with open("utils/data_Txt/GA_val.txt", "w") as f:
        for key, val in GA_val_dict.items():
            f.write(key + "," + str(val) + "\n")
if __name__=='__main__':
    get_training_txt()