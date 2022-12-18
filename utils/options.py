import argparse

mode = 'train'  # train mode or test mode


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
    parser.add_argument('--proposed_meth', type=str, default='pt+loss', help="GPU ID, -1 for CPU")
    # pt, pt+loss, pt+loss+weight


    '''dataset'''
    parser.add_argument('--dataset', type=str, default='tinyimagenet', help="name of dataset")
    # cifar10; cifar100; tinyimagenet
    parser.add_argument('--dirichlet_alpha', type=float, default=0.1, help="dirichlet separate partition")
    parser.add_argument('--pathological_shards', type=int, default=2, help="dirichlet separate partition")
    parser.add_argument('--gen_dataloader', type=str, default='False', help="dataset split for clients")

    if mode == 'train':
        parser.add_argument('--split', type=str, default='dirichlet', help="dataset split for clients")
        # dirichlet or pathological
        parser.add_argument('--num_users', type=int, default=20, help="number of users: K")
        parser.add_argument('--use_cache', type=str, default='False', help="True for load dataloader to cache")
    elif mode == 'debug':
        parser.add_argument('--split', type=str, default='pathological', help="dataset split for clients")
        parser.add_argument('--num_users', type=int, default=5, help="number of users: K")
        parser.add_argument('--use_cache', type=str, default='False', help="True for load dataloader to cache")
    '''model defination'''
    parser.add_argument('--FedPrompt_type', type=str, default=None, help="fed head and inner prompt")
    parser.add_argument('--pmt_num', type=int, default=10, help="prompt token number")

    '''Proposed methods'''
    parser.add_argument('--meth', type=str, default=None, help="personalized FL method")
    # FedPrompt; pFedPrompt; Linear; Full; pFedPrompt-pt

    '''pFedPrompt parameters'''
    parser.add_argument('--sim', type=str, default=None, help="Similarity")
    # E-concat; C-concat; E-average; C-average; ON

    args = parser.parse_args()
    return args
