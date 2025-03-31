import sys

sys.path.append("/home/aaa/yangjie/Fed_win/")
from models.ResNet import ResNet18
from utils.ourdatasets import hos1, hos2, hos3
from utils.ourdatasets_GAIM import get_dataset, get_targets
from utils.options import args_parser
from utils.sampling import noniid_our, dirichlet_noniid


def get_muti_hos_data():
    """获取多医院的数据集"""
    dataset_train_our = hos1["train"] + hos2["train"] + hos3["train"]
    dataset_test_our = hos1["test"] + hos2["test"] + hos3["test"]
    dataset_val_our = hos1["val"] + hos2["val"] + hos3["val"]
    print("train:", len(dataset_train_our))
    print("test:", len(dataset_test_our))
    print("val:", len(dataset_val_our))
    dataset_train = dataset_train_our
    dataset_test = dataset_test_our
    dataset_val = dataset_val_our
    dict_users_train = {
        0: [i for i in range(len(hos1["train"]))],
        1: [
            i
            for i in range(len(hos1["train"]), len(hos1["train"]) + len(hos2["train"]))
        ],
        2: [
            i
            for i in range(
                len(hos1["train"]) + len(hos2["train"]), len(dataset_train_our)
            )
        ],
    }
    dict_users_test = {
        0: [i for i in range(len(hos1["test"]))],
        1: [i for i in range(len(hos1["test"]), len(hos1["test"]) + len(hos2["test"]))],
        2: [
            i
            for i in range(len(hos1["test"]) + len(hos2["test"]), len(dataset_test_our))
        ],
    }
    return dataset_train, dataset_test, dataset_val, dict_users_train, dict_users_test


def get_data(args):
    """img_type可选：IM、GA、COVID-19、MutiHosIM、MutiHosGA，获取dataloader"""
    img_type = args.dataset
    if "MultiHos" in img_type:
        dataset_train, dataset_test, dataset_val, dict_users_train, dict_users_test = (
            get_muti_hos_data()
        )

    if img_type in ["IM", "GA", "COVID-19"]:
        image_datasets = get_dataset(img_type)
        targets_train, targets_test = get_targets(img_type)
        dataset_train = image_datasets["train"]
        dataset_test = image_datasets["test"]
        dataset_val = image_datasets["val"]
        if args.dirichlet:
            dict_users_train, _ = dirichlet_noniid(args.num_users, targets_train, 0.1)
            dict_users_test, _ = dirichlet_noniid(args.num_users, targets_test, 0.1)
        else:
            dict_users_train, rand_set_all = noniid_our(
                dataset_train,
                args.num_users,
                args.shard_per_user,
                args.server_data_ratio,
                targets=targets_train,
            )
            dict_users_test, rand_set_all = noniid_our(
                dataset_test,
                args.num_users,
                args.shard_per_user,
                args.server_data_ratio,
                targets=targets_test,
                rand_set_all=rand_set_all,
            )
    return dataset_train, dataset_test, dataset_val, dict_users_train, dict_users_test


def get_model(args):
    """根据参数获取模型"""
    if args.model == "resnet18" and args.dataset in [
        "IM",
        "GA",
        "COVID-19",
        "MultiHosIM",
        "MultiHosGA",
    ]:
        net_glob = ResNet18(args).to(args.device)
    else:
        exit("Error: unrecognized model")
    return net_glob


if __name__ == "__main__":
    args = args_parser()
    args.dataset = "COVID-19"
    dirichlet = args.dirichlet
    dataset_train, dataset_test, dataset_val, dict_users_train, dict_users_test = (
        get_data(args)
    )
    print(len(dataset_train))
