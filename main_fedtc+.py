import copy
import pickle
import numpy as np
import torch
import os

from utils.options import args_parser
from utils.train_utils import get_data, get_model
from models.Update import LocalUpdateFedTKF
from models.test import test_img_cur_model
from models.Fed import FedAvg
import logging


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

if __name__ == "__main__":
    args = args_parser()
    logging.info(f"dataset:{args.dataset}")
    logging.info(f"Top:{args.Top}")
    logging.info(f"UDL:{args.UDL}")
    logging.info(f"GPU:{args.gpu}")
    logging.info(f"dirichlet:{args.dirichlet}")
    logging.info(f"local_upt_part:{args.local_upt_part}")
    logging.info(f"aggr_part:{args.aggr_part}")
    logging.info(f"frac:{args.frac}")
    # set seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(args.seed)
    # set device
    args.device = torch.device(
        "cuda:{}".format(args.gpu)
        if torch.cuda.is_available() and args.gpu != -1
        else "cpu"
    )
    # set save path
    base_dir = "./save/{}/{}_iid{}_num{}_C{}_le{}_m{}_wd{}/shard{}_sdr{}/{}/".format(
        args.dataset,
        args.model,
        args.iid,
        args.num_users,
        args.frac,
        args.local_ep,
        args.momentum,
        args.wd,
        args.shard_per_user,
        args.server_data_ratio,
        args.results_save,
    )
    algo_dir = "local_upt_{}_aggr_{}_top{}".format(
        args.local_upt_part, args.aggr_part, args.Top
    )
    if not os.path.exists(os.path.join(base_dir, algo_dir)):
        os.makedirs(os.path.join(base_dir, algo_dir), exist_ok=True)
    # get dataset
    dataset_train, dataset_test, dataset_val, dict_users_train, dict_users_test = (
        get_data(args=args)
    )
    logging.info(f"dataset_train:{len(dataset_train)}")
    logging.info(f"dataset_test:{len(dataset_test)}")
    logging.info(f"dataset_val:{len(dataset_val)}")
    logging.info(f"dict_users_train:{dict_users_train.keys()}")
    dict_users_train = {
        int(k): np.array(v, dtype=int) for k, v in dict_users_train.items()
    }
    dict_users_test = {
        int(k): np.array(v, dtype=int) for k, v in dict_users_test.items()
    }
    dict_save_path = os.path.join(base_dir, algo_dir, "dict_users.pkl")
    with open(dict_save_path, "wb") as handle:
        pickle.dump((dict_users_train, dict_users_test), handle)
    # init models
    net_glob, net_cur = get_model(args), get_model(args)
    net_glob.train()
    net_local_list = []
    for user_idx in range(args.num_users):
        net_local_list.append(copy.deepcopy(net_glob))
    results_save_path = os.path.join(base_dir, algo_dir, "results.csv")
    loss_train = []
    best_loss = None
    best_acc = None
    best_epoch = None
    lr = args.lr
    results = []
    # 第一个位置存储acc_val；第二个位置存储样本量；第三个位置copy模型参数
    local_weights_glob = [[0, 0, 0] for i in range(6)]
    w_glob = None

    for epoch in range(args.epochs):
        loss_locals, w_locals = [], []
        # 随机选取m个客户端
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        logging.info(
            "Round {}, lr: {:.6f}, idxs_users:{}".format(epoch, lr, idxs_users)
        )
        # update at local clients
        logging.info(f"idxs_users:{idxs_users}")
        for idx in idxs_users:
            local = LocalUpdateFedTKF(
                args=args, dataset=dataset_train, idxs=dict_users_train[idx]
            )
            net_local = copy.deepcopy(net_local_list[idx])
            if args.local_upt_part == "body":
                body_lr, head_lr = lr, 0.0
            elif args.local_upt_part == "head":
                body_lr, head_lr = 0.0, lr
            elif args.local_upt_part == "full":
                body_lr, head_lr = lr, lr
            w_local, loss = local.train(
                net=net_local.to(args.device),
                body_lr=body_lr,
                head_lr=head_lr,
            )

            net_cur.load_state_dict(copy.deepcopy(w_local))
            # test on server
            loss_locals.append(copy.deepcopy(loss))
            w_locals.append(copy.deepcopy(w_local))
        # TEA Aggregation
        w_aggregation = []
        for item in w_locals:
            w_aggregation.append(item)
        # 进行topK聚合
        for i in range(args.Top):
            if local_weights_glob[i][0] != 0:
                w_aggregation.append(local_weights_glob[i][2])
        logging.info(f"len(w_aggregation):{len(w_aggregation)}")
        w_glob = FedAvg(w_aggregation)
        # Broadcast
        update_keys = list(w_glob.keys())
        if args.aggr_part == "body":
            update_keys = [k for k in update_keys if args.UDL not in k]
        elif args.aggr_part == "head":
            update_keys = [k for k in update_keys if args.UDL in k]
        elif args.aggr_part == "full":
            pass
        w_glob = {k: v for k, v in w_glob.items() if k in update_keys}
        # 使用全局模型更新本地模型
        for user_idx in range(args.num_users):
            net_local_list[user_idx].load_state_dict(w_glob, strict=False)

        if (epoch + 1) in [args.epochs // 2, (args.epochs * 3) // 4]:
            lr *= 0.1
        loss_avg = sum(loss_locals) / len(loss_locals)
        loss_train.append(loss_avg)
        w_glob = copy.deepcopy(net_local_list[0].state_dict())
        if (epoch + 1) % args.test_freq == 0:
            acc_test, loss_test = test_img_cur_model(
                net_local_list[0], dataset_test, args
            )
            local_weights_glob[-1][0] = epoch + 1
            local_weights_glob[-1][1] = 1000
            local_weights_glob[-1][2] = copy.deepcopy(net_local_list[0].state_dict())
            local_weights_glob.sort(key=lambda x: x[0], reverse=True)
            logging.info([i[0] for i in local_weights_glob])
            logging.info(
                "Round {:3d}, Average loss {:.3f}, Test loss {:.3f}, Test accuracy: {:.2f}".format(
                    epoch, loss_avg, loss_test, acc_test
                )
            )
            if best_acc is None or acc_test > best_acc:
                best_acc = acc_test
                best_epoch = epoch
                best_save_path = os.path.join(base_dir, algo_dir, "best_model.pt")
                torch.save(local_weights_glob[0][2], best_save_path)
                for user_idx in range(args.num_users):
                    best_save_path = os.path.join(
                        base_dir, algo_dir, "best_local_{}.pt".format(user_idx)
                    )
                    torch.save(net_local_list[user_idx].state_dict(), best_save_path)
    logging.info("Best model, iter: {}, acc: {}".format(best_epoch, best_acc))
