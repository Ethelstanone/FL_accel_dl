import copy
import pickle
import shelve

import numpy as np
import torch

from Update import LocalUpdate
from data.loader import load_dataset, load_dataloader
from utils.options import args_parser

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    print('device: ', args.device)

    '''
    dataloader
    '''
    m = max(int(args.frac * args.num_users), 1)  # number of users
    if args.gen_dataloader == 'True':
        # load dataset
        dataset_train, dataset_test, dict_users, dict_users_test = load_dataset(args)
        # load dataloader
        load_dataloader(args, m, dataset_train, dataset_test, dict_users, dict_users_test)
    else:
        # directly load dataloader from disk
        pass

    train_list = shelve.open('train_loader_{}_{}cl'.format(args.dataset, args.num_users))
    test_list = shelve.open('test_loader_{}_{}cl'.format(args.dataset, args.num_users))
    fp = open('test_sample_num_{}_{}cl.pkl'.format(args.dataset, args.num_users), 'rb+')
    test_num = pickle.load(fp)

    '''
    model
    '''
    net_glob = ''' your model '''

    '''
    train
    '''
    for iters in range(args.epochs):
        print("\ncommunication round: {:3d}\n".format(iters))
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        for idx in idxs_users:
            print("client:", idx)
            # init each client
            local = LocalUpdate(args=args,
                                train_loader=train_list['{}'.format(idx)],
                                test_loader=test_list['{}'.format(idx)],
                                test_sample=test_num[idx])
            # client local training
            loss, acc, net_p = local.train(net=copy.deepcopy(net_glob).to(args.device))

        '''
        aggregation
        '''
        # upload local model/prompt to server for aggregation
        '''FedAvg etc.'''




