import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torchvision import datasets, transforms

from data.partition import partition_report


def cifar10_iid(dataset, num_users):
    """
    Sample IID client data from CIFAR-10 dataset
    :param dataset: dataset for partition
    :param num_users: n clients
    :return: dict of image index
    """
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def cifar10_pathological_noniid(dataset, num_users, train=True, shards=2):
    """
    Sample pathological non-IID client data from CIFAR-10 dataset
    :param shards: each client choose n shards
    :param train: dataset for train or for test
    :param dataset: dataset for partition
    :param num_users: n clients
    :return: dict of image index
    """
    if train == True:
        num_shards, num_imgs = 20, 2500
    else:
        num_shards, num_imgs = 20, 500

    idx_shard = [i for i in range(num_shards)]
    # clients imgs index dictionary
    dict_users = {i: np.array([]) for i in range(num_users)}
    # clients labels dictionary
    dict_users_lbs = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    # labels = dataset.train_labels.numpy()
    labels = np.array(dataset.targets)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]

    idxs = idxs_labels[0, :]
    lbs = idxs_labels[1, :]

    # divide and assign
    for i in range(num_users):
        np.random.seed(0)
        rand_set = set(np.random.choice(idx_shard, shards, replace=False))  # 随机选择几个shard

        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
            dict_users_lbs[i] = np.concatenate(
                (dict_users_lbs[i], lbs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
    return dict_users


def cifar10_dirichlet_noniid(dataset, num_users, alpha):
    """
    Sample dirichlet non-IID client data from CIFAR-10 dataset
    :param alpha: dirichlet alpha determine non-IID level
    :param dataset: dataset for partition
    :param num_users: n clients
    :return: dict of image index
    """
    # clients imgs index dictionary
    dict_users = {i: np.array([]) for i in range(num_users)}

    train_labels = np.array(dataset.targets)
    n_classes = 10
    np.random.seed(0)
    label_distribution = np.random.dirichlet([alpha] * num_users, n_classes)
    # (K, N)的类别标签分布矩阵X，记录每个client占有每个类别的多少

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


if __name__ == '__main__':

    trans_cifar = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
    dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)

    num_classes = len(dataset_train.classes)

    # dict_users = cifar10_pathological_noniid(dataset=dataset_train, num_users=5, train=True)
    # dict_users_test = cifar10_pathological_noniid(dataset=dataset_test, num_users=5, train=False)

    dict_users = cifar10_dirichlet_noniid(dataset=dataset_train, num_users=10, alpha=0.1)
    dict_users_test = cifar10_dirichlet_noniid(dataset=dataset_test, num_users=10, alpha=0.1)

    # print(dict_users)

    '''train dataset'''

    # 为了方便结果可视化，我们提供了数据划分报告生成的函数，允许生成结果报告以及写入文件
    csv_file = "./cifar10_train_10clients.csv"
    partition_report(dataset_train.targets, dict_users,
                     class_num=num_classes,
                     verbose=False, file=csv_file)

    # 报告很容易用 csv.reader() 或 pandas.read_csv()进行解析
    hetero_dir_part_df = pd.read_csv(csv_file, header=1)
    hetero_dir_part_df = hetero_dir_part_df.set_index('client')
    col_names = [f"class{i}" for i in range(num_classes)]
    for col in col_names:
        hetero_dir_part_df[col] = (hetero_dir_part_df[col] * hetero_dir_part_df['Amount']).astype(int)
    # 选择前10个client的划分结果进行可视化
    hetero_dir_part_df[col_names].iloc[:10].plot.barh(stacked=True)
    plt.tight_layout()
    plt.xlabel('sample num')
    plt.savefig(f"./cifar10_train_10clients.png", dpi=400)

    '''test dataset'''
    # 为了方便结果可视化，我们提供了数据划分报告生成的函数，允许生成结果报告以及写入文件
    csv_file = "./cifar10_test_10clients.csv"
    partition_report(dataset_test.targets, dict_users_test,
                     class_num=num_classes,
                     verbose=False, file=csv_file)

    # 报告很容易用 csv.reader() 或 pandas.read_csv()进行解析
    hetero_dir_part_df = pd.read_csv(csv_file, header=1)
    hetero_dir_part_df = hetero_dir_part_df.set_index('client')
    col_names = [f"class{i}" for i in range(num_classes)]
    for col in col_names:
        hetero_dir_part_df[col] = (hetero_dir_part_df[col] * hetero_dir_part_df['Amount']).astype(int)
    # 选择前10个client的划分结果进行可视化
    hetero_dir_part_df[col_names].iloc[:10].plot.barh(stacked=True)
    plt.tight_layout()
    plt.xlabel('sample num')
    plt.savefig(f"./cifar10_test_10clients.png", dpi=400)
