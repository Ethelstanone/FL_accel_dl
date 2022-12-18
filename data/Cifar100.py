import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torchvision import datasets, transforms

from data.partition import partition_report


def cifar100_dirichlet_noniid(dataset, num_users, alpha):
    """
    Sample dirichlet non-IID client data from CIFAR-100 dataset
    :param alpha: dirichlet alpha determine non-IID level
    :param dataset: dataset for partition
    :param num_users: n clients
    :return: dict of image index
    """
    # clients imgs index dictionary
    dict_users = {i: np.array([]) for i in range(num_users)}

    train_labels = np.array(dataset.targets)
    n_classes = 100
    np.random.seed(0)
    label_distribution = np.random.dirichlet([alpha] * num_users, n_classes)
    # (K, N)的类别标签分布矩阵X，记录每个client占有每个类别的多少，
    # K代表分成n_classes个离散概率list，N代表每个概率list有num_users个概率值

    class_idcs = [np.argwhere(train_labels == y).flatten()
                  for y in range(n_classes)]
    # 记录每个K个类别对应的样本下标

    client_idcs = [[] for _ in range(num_users)]
    # 记录N个client分别对应样本集合的索引
    for c, fracs in zip(class_idcs, label_distribution):
        # np.split按照比例将类别为k的样本划分为了N个子集
        # for i, idcs 为遍历第i个client对应样本集合的索引
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1] * len(c)).astype(int))):
            client_idcs[i] += [idcs]

    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]

    for i in range(num_users):
        dict_users[i] = client_idcs[i]

    return dict_users


def cifar100_pathological_noniid(dataset, num_users, train=True, shards=2):
    """
    Sample pathological non-IID client data from CIFAR-100 dataset
    :param shards: each client choose n shards
    :param train: dataset for train or for test
    :param dataset: dataset for partition
    :param num_users: n clients
    :return: dict of image index
    """
    if train == True:  # 50000
        num_shards, num_imgs = 50, 1000
    else:  # 10000
        num_shards, num_imgs = 50, 200

    idx_shard = [i for i in range(num_shards)]
    # clients imgs index dictionary
    dict_users = {i: np.array([]) for i in range(num_users)}
    # clients labels dictionary
    dict_users_lbs = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)

    labels = np.array(dataset.targets)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]

    idxs = idxs_labels[0, :]
    lbs = idxs_labels[1, :]

    # divide and assign
    for i in range(num_users):
        np.random.seed(0)
        rand_set = set(np.random.choice(idx_shard, shards, replace=False))
        # print(rand_set)
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
            dict_users_lbs[i] = np.concatenate(
                (dict_users_lbs[i], lbs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
    return dict_users


if __name__ == '__main__':

    trans_cifar = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset_train = datasets.CIFAR100('../data/cifar100', train=True, download=True, transform=trans_cifar)
    dataset_test = datasets.CIFAR100('../data/cifar100', train=False, download=True, transform=trans_cifar)

    num_classes = len(dataset_train.classes)

    split_meth = 'pathological'
    num_users = 10

    if split_meth == 'dirichlet':
        dict_users_train = cifar100_dirichlet_noniid(dataset=dataset_train, num_users=num_users, alpha=0.01)
        dict_users_test = cifar100_dirichlet_noniid(dataset=dataset_test, num_users=num_users, alpha=0.01)
    elif split_meth == 'pathological':
        dict_users_train = cifar100_pathological_noniid(dataset=dataset_train, num_users=num_users, train=True)
        dict_users_test = cifar100_pathological_noniid(dataset=dataset_test, num_users=num_users, train=False)

    csv_file_train = "./split/cifar100/cifar100_train_{:.0f}clients_{}.csv".format(num_users, split_meth)
    png_file_train = "./split/cifar100/cifar100_train_{:.0f}clients_{}.png".format(num_users, split_meth)

    csv_file_test = "./split/cifar100/cifar100_test_{:.0f}clients_{}.csv".format(num_users, split_meth)
    png_file_test = "./split/cifar100/cifar100_test_{:.0f}clients_{}.png".format(num_users, split_meth)

    '''train dataset'''
    # 为了方便结果可视化，我们提供了数据划分报告生成的函数，允许生成结果报告以及写入文件
    partition_report(dataset_train.targets, dict_users_train,
                     class_num=num_classes,
                     verbose=False, file=csv_file_train)

    # 报告很容易用 csv.reader() 或 pandas.read_csv()进行解析
    hetero_dir_part_df = pd.read_csv(csv_file_train, header=1)
    hetero_dir_part_df = hetero_dir_part_df.set_index('client')
    col_names = [f"class{i}" for i in range(num_classes)]
    for col in col_names:
        hetero_dir_part_df[col] = (hetero_dir_part_df[col] * hetero_dir_part_df['Amount']).astype(int)
    # 选择前10个client的划分结果进行可视化
    hetero_dir_part_df[col_names].iloc[:10].plot.barh(stacked=True)
    # plt.tight_layout()
    plt.xlabel('sample num')
    plt.legend(loc=0, ncol=4, prop={'size': 3})  # 图例
    plt.savefig(png_file_train, dpi=400)  # , bbox_inches='tight'

    '''test dataset'''
    # 为了方便结果可视化，我们提供了数据划分报告生成的函数，允许生成结果报告以及写入文件
    partition_report(dataset_test.targets, dict_users_test,
                     class_num=num_classes,
                     verbose=False, file=csv_file_test)

    # 报告很容易用 csv.reader() 或 pandas.read_csv()进行解析
    hetero_dir_part_df = pd.read_csv(csv_file_test, header=1)
    hetero_dir_part_df = hetero_dir_part_df.set_index('client')
    col_names = [f"class{i}" for i in range(num_classes)]
    for col in col_names:
        hetero_dir_part_df[col] = (hetero_dir_part_df[col] * hetero_dir_part_df['Amount']).astype(int)
    # 选择前10个client的划分结果进行可视化
    hetero_dir_part_df[col_names].iloc[:10].plot.barh(stacked=True)
    # plt.tight_layout()
    plt.xlabel('sample num')
    plt.legend(loc=0, ncol=4, prop={'size': 3})
    plt.savefig(png_file_test, dpi=400)
