import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    '''federated arguments'''
    parser.add_argument('--epochs', type=int, default=240, help="rounds of training")
    parser.add_argument('--frac', type=float, default=1, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=1, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=10, help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    # pt, pt+loss, pt+loss+weight


    '''dataset'''
    parser.add_argument('--dataset', type=str, default='tinyimagenet', help="name of dataset")
    # cifar10; cifar100; tinyimagenet
    parser.add_argument('--dirichlet_alpha', type=float, default=0.1, help="dirichlet partition alpha")
    parser.add_argument('--pathological_shards', type=int, default=2, help="pathological partition each client class shards")
    parser.add_argument('--gen_dataloader', type=str, default='False', help="whether save dataloader to disk")

    parser.add_argument('--split', type=str, default='dirichlet', help="dataset split for clients")
    # dirichlet or pathological
    parser.add_argument('--num_users', type=int, default=20, help="number of users: K")


    args = parser.parse_args()
    return args
