# 本代码只需要修改读取的数据集：0-肠化；1-萎缩；2-正常
from PIL import Image, ImageDraw
from torch.utils.data import Dataset, DataLoader
import glob
import random
from torchvision import transforms
import sys

sys.path.append("/home/aaa/yangjie/Fed_win/")
from configs.configs import hos_file_list, IM_pos_list, GA_pos_list, Normal_pos_list
from utils.options import args_parser

args = args_parser()
# 初始化随机种子
random.seed(20240323)


# 制作txt文件
def get_txt():
    hos_list = hos_file_list
    print(f"MultiHos:{args.dataset}")
    if "IM" in args.dataset:
        pos_list = IM_pos_list
    else:
        pos_list = GA_pos_list
    pos_list_Normal = Normal_pos_list
    # huaxi = '/home/bigspace/huaxi_data'
    pics_train = []
    pics_test = []
    pics_val = []
    for i, hos in enumerate(hos_list):
        res_train = []
        res_test = []
        res_val = []
        num_pos = []
        for j, pos in enumerate(pos_list):
            path = hos + pos
            cur = [i + "," + str(j) for i in glob.glob(path)]
            res_train += cur[: int(len(cur) * 0.9 * 0.8)]
            res_test += cur[
                int(len(cur) * 0.9 * 0.8) : int(len(cur) * 0.9 * 0.8)
                + int(len(cur) * 0.9 * 0.2)
            ]
            res_val += cur[int(len(cur) * 0.9 * 0.8) + int(len(cur) * 0.9 * 0.2) :]
            num_pos.append(len(cur))
        for j, pos in enumerate(pos_list_Normal):
            # TODO: real hos normal data is equal to 0
            # path = huaxi+pos # sample huaxi data
            path = hos + pos  # real three hospitals
            cur = [i + "," + str(j + 5) for i in glob.glob(path)]
            res_train += cur[: int(len(cur) * 0.9 * 0.8)]
            res_test += cur[
                int(len(cur) * 0.9 * 0.8) : int(len(cur) * 0.9 * 0.8)
                + int(len(cur) * 0.9 * 0.2)
            ]
            res_val += cur[int(len(cur) * 0.9 * 0.8) + int(len(cur) * 0.9 * 0.2) :]
        pics_train.append(res_train)
        pics_test.append(res_test)
        pics_val.append(res_val)
    return pics_train, pics_test, pics_val


# 使用PIL Image读取图片
def default_loader(path):
    try:
        image = Image.open(path)
        # 创建遮罩图像
        mask = Image.new("L", image.size, 0)
        draw = ImageDraw.Draw(mask)
        width, height = image.size
        center_x = width // 2
        center_y = height // 2
        radius = min(center_x, center_y) // 2
        draw.ellipse(
            (
                center_x - radius,
                center_y - radius,
                center_x + radius,
                center_y + radius,
            ),
            fill=255,
        )
        result = Image.new("RGB", image.size)
        result.paste(image, mask=mask)
        return result
    except:
        print("Cannot read image: {}".format(path))


# 标准化模型的输入输出，最终返回img张量和label
class customData(Dataset):  #
    def __init__(self, img_names, img_labels, data_trans=None, loader=default_loader):
        self.img_name = img_names
        self.img_label = img_labels
        self.trans = data_trans
        self.loader = loader

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, item):
        img_name = self.img_name[item]
        label = self.img_label[item]
        img = self.loader(img_name)
        img = self.trans(img)
        return img, label


# 将三个hos分别封装成dataloader
def get_data(pics_train, pics_test, pics_val):
    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    }
    img_names = {
        "train": [line.split(",")[0] for line in pics_train],
        "test": [line.split(",")[0] for line in pics_test],
        "val": [line.split(",")[0] for line in pics_val],
    }
    img_labels = {
        "train": [int(line.split(",")[1]) for line in pics_train],
        "test": [int(line.split(",")[1]) for line in pics_test],
        "val": [int(line.split(",")[1]) for line in pics_val],
    }
    image_datasets = {
        x: customData(img_names[x], img_labels[x], data_trans=data_transforms["train"])
        for x in ["train", "test", "val"]
    }

    return image_datasets


def main():
    pics_train, pics_test, pics_val = get_txt()
    hos_image_datastes = []
    for i in range(len(pics_train)):
        image_datasets = get_data(pics_train[i], pics_test[i], pics_val[i])
        hos_image_datastes.append(image_datasets)
    return hos_image_datastes[0], hos_image_datastes[1], hos_image_datastes[2]


hos1, hos2, hos3 = main()

if __name__ == "__main__":
    hos1, hos2, hos3 = main()
    print(len(hos1["train"]))
    print(len(hos1["test"]))
    print(len(hos1["val"]))
    print(len(hos2["train"]))
    print(len(hos2["test"]))
    print(len(hos2["val"]))
    print(len(hos3["train"]))
    print(len(hos3["test"]))
    print(len(hos3["val"]))
    print(
        len(hos1["train"])
        + len(hos1["test"])
        + len(hos1["val"])
        + len(hos2["train"])
        + len(hos2["test"])
        + len(hos2["val"])
        + len(hos3["train"])
        + len(hos3["test"])
        + len(hos3["val"])
    )
