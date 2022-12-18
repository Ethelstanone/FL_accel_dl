from data.Cifar10 import cifar10_iid, cifar10_pathological_noniid, cifar10_dirichlet_noniid
from data.Cifar100 import *
from data.TinyImagenet import *
from torch.utils.data import DataLoader

import pickle
import shelve


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs.astype(np.int))

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

    def __len__(self):
        return len(self.idxs)


def load_dataset(args):
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(224),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    if args.dataset == 'cifar10':
        dataset_train = datasets.CIFAR10('./data/cifar', train=True, download=True, transform=trans)
        dataset_test = datasets.CIFAR10('./data/cifar', train=False, download=True, transform=trans)
        print("dataset: cifar10")
        if args.split == 'iid':
            dict_users = cifar10_iid(dataset_train, args.num_users)
        elif args.split == 'pathological':
            dict_users = cifar10_pathological_noniid(dataset_train, args.num_users, train=True,
                                                     shards=args.pathological_shards)
            dict_users_test = cifar10_pathological_noniid(dataset_test, args.num_users, train=False,
                                                          shards=args.pathological_shards)
            print("client dataset split method: pathological_shards={}".format(args.pathological_shards))
        elif args.split == 'dirichlet':
            dict_users = cifar10_dirichlet_noniid(dataset_train, args.num_users, alpha=args.dirichlet_alpha)
            dict_users_test = cifar10_dirichlet_noniid(dataset_test, args.num_users, alpha=args.dirichlet_alpha)
            print("client dataset split method: dirichlet_alpha={}".format(args.dirichlet_alpha))

    elif args.dataset == 'cifar100':
        dataset_train = datasets.CIFAR100('./data/cifar100', train=True, download=True, transform=trans)
        # print(type(list(dataset_train)))
        dataset_test = datasets.CIFAR100('./data/cifar100', train=False, download=True, transform=trans)
        print("dataset: cifar100")
        if args.split == 'pathological':
            dict_users = cifar100_pathological_noniid(dataset_test,
                                                      args.num_users,
                                                      train=False,
                                                      shards=args.pathological_shards)
            dict_users_test = cifar100_pathological_noniid(dataset_test,
                                                           args.num_users,
                                                           train=False,
                                                           shards=args.pathological_shards)
            print("client dataset split method: pathological_shards={}".format(args.pathological_shards))
        elif args.split == 'dirichlet':
            dict_users = cifar100_dirichlet_noniid(dataset_train, args.num_users, alpha=args.dirichlet_alpha)
            dict_users_test = cifar100_dirichlet_noniid(dataset_test, args.num_users, alpha=args.dirichlet_alpha)
            print("client dataset split method: dirichlet_alpha={}".format(args.dirichlet_alpha))

    elif args.dataset == 'tinyimagenet':
        path = './data/tiny-imagenet-200/'
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(224),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        dataset_train = data('train', transform=trans, url=path)
        dataset_test = data('val', transform=trans, url=path)
        print("dataset: tiny_imagenet")

        if args.split == 'pathological':
            dict_users = tinyimagenet_pathological_noniid(dataset=dataset_train,
                                                          num_users=args.num_users,
                                                          train=True,
                                                          shards=args.pathological_shards)
            dict_users_test = tinyimagenet_pathological_noniid(dataset=dataset_test,
                                                               num_users=args.num_users,
                                                               train=False,
                                                               shards=args.pathological_shards)
            print("client dataset split method: pathological_shards={}".format(args.pathological_shards))
        elif args.split == 'dirichlet':
            dict_users = tinyimagenet_dirichlet_noniid(dataset=dataset_train,
                                                       num_users=args.num_users,
                                                       alpha=args.dirichlet_alpha)
            dict_users_test = tinyimagenet_dirichlet_noniid(dataset=dataset_test,
                                                            num_users=args.num_users,
                                                            alpha=args.dirichlet_alpha)
            print("client dataset split method: dirichlet_alpha={}".format(args.dirichlet_alpha))

    else:
        exit('Error: unrecognized dataset')

    img_size = dataset_train[0][0].shape
    print('img_size: ', img_size)

    return dataset_train, dataset_test, dict_users, dict_users_test


def load_dataloader(args, m, train_set, test_set, dicts, dicts_test):
    # use shelve to accelerate dataloader

    # each client test sample number list
    test_sample_num = [0 for _ in range(m)]

    # save each client dataloader to a database
    for cl in range(m):
        # train dataset
        client_train = DatasetSplit(train_set, dicts[cl])
        # test dataset
        client_test = DatasetSplit(test_set, dicts_test[cl])
        # train dataloader
        ldr_train = DataLoader(client_train,
                               batch_size=args.local_bs,
                               shuffle=True,
                               num_workers=6,
                               pin_memory=True,
                               persistent_workers=True)
        # test dataloader
        ldr_test = DataLoader(client_test,
                              batch_size=args.local_bs,
                              shuffle=False,
                              num_workers=6,
                              pin_memory=True,
                              persistent_workers=True)
        # test sample number
        test_sample_num[cl] = len(ldr_test.dataset)

        # save to shelve
        with shelve.open('train_loader_{}_{}cl'.format(args.dataset, args.num_users)) as db_train:
            db_train['{}'.format(cl)] = list(ldr_train)

        with shelve.open('test_loader_{}_{}cl'.format(args.dataset, args.num_users)) as db_test:
            db_test['{}'.format(cl)] = list(ldr_test)

    with open('test_sample_num_{}_{}cl.pkl'.format(args.dataset, args.num_users), 'wb') as f:
        pickle.dump(test_sample_num, f, pickle.HIGHEST_PROTOCOL)
