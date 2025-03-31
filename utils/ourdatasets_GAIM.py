from PIL import Image
from torch.utils.data import Dataset
import random
from torchvision import transforms

random.seed(1)


def get_targets(img_type: str):
    """获取训练集和测试集的标签"""
    with open(f"/home/aaa/yangjie/Fed_win/utils/data_Txt/{img_type}_train.txt") as f:
        lines = f.readlines()
        train_targets = [int(line.strip().split(",")[1]) for line in lines]
    with open(f"/home/aaa/yangjie/Fed_win/utils/data_Txt/{img_type}_test.txt") as f:
        lines = f.readlines()
        test_targets = [int(line.strip().split(",")[1]) for line in lines]
    return train_targets, test_targets


# IMGA图像变换操作
data_transforms = {
    n: transforms.Compose(
        [
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    for n in ["train", "test", "val"]
}
# COVID-19图像变换操作
data_transforms_covid = {
    n: transforms.Compose(
        [
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    for n in ["train", "test", "val"]
}


def default_loader(path):
    """打开指定路径的图像"""
    try:
        img = Image.open(path)
        return img
    except:
        print("Cannot read image: {}".format(path))


class customData(Dataset):  #
    def __init__(self, txt_path, dataset="", data_trans=None, loader=default_loader):
        # 读取txt文件中的每一行并分割成图片路径和标签
        with open(txt_path) as input_file:
            lines = input_file.readlines()
            self.img_names = [line.strip().split(",")[0] for line in lines]
            self.img_labels = [int(line.strip().split(",")[1]) for line in lines]
        self.trans = data_trans
        self.dataset = dataset
        self.loader = loader

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, item):
        img_name = self.img_names[item]
        label = self.img_labels[item]
        img = self.loader(img_name)
        img = self.trans(img)
        return img, label


def get_dataset(dataset_type: str):
    if dataset_type == "IM" or dataset_type == "GA":
        image_datasets = {
            x: customData(
                txt_path=(
                    "/home/aaa/yangjie/Fed_win/utils/data_Txt/"
                    + f"{dataset_type}_"
                    + x
                    + ".txt"
                ),
                data_trans=data_transforms[x],
                dataset=x,
            )
            for x in ["train", "test", "val"]
        }
        return image_datasets
    elif dataset_type == "COVID-19":
        image_datasets = {
            x: customData(
                txt_path=(
                    "/home/aaa/yangjie/Fed_win/utils/data_Txt/"
                    + f"{dataset_type}_"
                    + x
                    + ".txt"
                ),
                data_trans=data_transforms_covid[x],
                dataset=x,
            )
            for x in ["train", "test", "val"]
        }
        return image_datasets


if __name__ == "__main__":
    image_datasets = get_dataset("MultiHosIM")
    dataset_train = image_datasets["train"]
    dataset_test = image_datasets["test"]
    dataset_val = image_datasets["val"]
    print(dataset_train)
    print(dataset_test)
