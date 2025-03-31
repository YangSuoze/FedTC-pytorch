import argparse
import torch
def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epochs', type=int, default=32, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=30, help="number of users: K")
    parser.add_argument('--shard_per_user', type=int, default=2, help="classes per user")
    parser.add_argument('--frac', type=float, default=0.2, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=10, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=16, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=16, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.9, help="SGD momentum (default: 0.5)")
    parser.add_argument('--wd', type=float, default=0.0, help="weight decay (default: 0.0)")
    parser.add_argument('--local_ep_pretrain', type=int, default=10, help="the number of pretrain local ep")
    parser.add_argument('--mu', type=float, default=2.0, help="parameter for proximal local SGD")
    parser.add_argument('--gpu', type=int, default=3, help="GPU ID, -1 for CPU")
    parser.add_argument('--dataset', type=str, default='IM', help="name of dataset")
    parser.add_argument('--dirichlet', type=bool, default=True, help="False or True")
    parser.add_argument('--seed', type=int, default=2024, help='random seed (default: 1)')
    
    # model hyperparameters
    parser.add_argument('--Top', type=int, default=1, help="parameter for Top")
    parser.add_argument('--UDL', type=str, default='fc1', help="parameter for weight")
    # model arguments
    parser.add_argument('--model', type=str, default='resnet18', help='model name')
    # other arguments
    parser.add_argument('--iid', action='store_true', help='whether i.i.d or not')
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--test_freq', type=int, default=1, help='how often to test on val set')
    parser.add_argument('--results_save', type=str, default='/', help='define fed results save folder')
    
    # additional arguments
    parser.add_argument('--local_upt_part', type=str, default='body', help='body, head, or full')
    parser.add_argument('--aggr_part', type=str, default='body', help='body, head, or full')
    
    parser.add_argument('--server_data_ratio', type=float, default=0.0, help='The percentage of data that servers also have across data of all clients.')
    
    # arguments for a single model
    parser.add_argument('--body_lr', type=float, default=None, help="learning rate for the body of the model")
    parser.add_argument('--head_lr', type=float, default=None, help="learning rate for the head of the model")
    parser.add_argument('--body_m', type=float, default=None, help="momentum for the body of the model")
    parser.add_argument('--head_m', type=float, default=None, help="momentum for the head of the model")
        
    args = parser.parse_args()
    
    return args

if __name__=='__main__':
    args = args_parser()
