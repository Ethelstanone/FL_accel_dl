import os

import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


def tinyimagenet_dirichlet_noniid(dataset, num_users, alpha):
    """
    Sample dirichlet non-IID client data from Tiny Imagenet dataset
    :param alpha: dirichlet alpha determine non-IID level
    :param dataset: dataset for partition
    :param num_users: n clients
    :return: dict of image index
    """
    # clients imgs index dictionary
    dict_users = {i: np.array([]) for i in range(num_users)}

    train_labels = np.array(dataset.targets)
    n_classes = 200
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


def tinyimagenet_pathological_noniid(dataset, num_users, train=True, shards=2):
    """
    Sample non-IID client data from Tiny Imagenet dataset
    :param shards: each client choose n shards
    :param train: dataset for train or for test
    :param dataset: dataset for partition
    :param num_users: n clients
    :return: dict of image index
    """
    # train 10w
    if train == True:
        num_shards, num_imgs = 400, 250
    # test 1w
    else:
        num_shards, num_imgs = 40, 250

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

        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0
            )
            dict_users_lbs[i] = np.concatenate(
                (dict_users_lbs[i], lbs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0
            )
    return dict_users


class data(Dataset):
    def __init__(self, type, transform, url):
        self.type = type
        self.targets = []
        self.url = url

        labels_t = []
        image_names = []
        with open(self.url + 'wnids.txt') as wnid:
            for line in wnid:
                labels_t.append(line.strip('\n'))
        for label in labels_t:
            txt_path = self.url + 'train\\' + label + '\\' + label + '_boxes.txt'
            image_name = []
            with open(txt_path) as txt:
                for line in txt:
                    image_name.append(line.strip('\n').split('\t')[0])
            image_names.append(image_name)
        # labels = np.arange(200)

        val_labels_t = []  # len 10000
        val_labels = []  # len 10000
        val_names = []  # len 10000
        with open(self.url + 'val\\val_annotations.txt') as txt:
            for line in txt:
                val_names.append(line.strip('\n').split('\t')[0])
                val_labels_t.append(line.strip('\n').split('\t')[1])
        for i in range(len(val_labels_t)):
            for i_t in range(len(labels_t)):
                if val_labels_t[i] == labels_t[i_t]:
                    val_labels.append(i_t)
        val_labels = np.array(val_labels)

        if type == 'train':
            i = 0
            self.images = []
            for label in labels_t:
                image = []
                for image_name in image_names[i]:
                    image_path = os.path.join(self.url + 'train', label, 'images', image_name)
                    image.append(cv2.imread(image_path))
                self.images.append(image)
                i = i + 1
            self.images = np.array(self.images)
            self.images = self.images.reshape(-1, 64, 64, 3)

            for i in range(100000):
                self.targets.append(i // 500)

        elif type == 'val':
            self.val_images = []
            i = 0
            for val_image in val_names:
                val_image_path = os.path.join(self.url + 'val\\images', val_image)
                self.val_images.append(cv2.imread(val_image_path))
                self.targets.append(val_labels[i])
                i = i + 1
            self.val_images = np.array(self.val_images)
        self.transform = transform

    def __getitem__(self, index):

        labels_t = []
        image_names = []
        with open(self.url + 'wnids.txt') as wnid:
            for line in wnid:
                labels_t.append(line.strip('\n'))
        for label in labels_t:
            txt_path = self.url + 'train\\' + label + '\\' + label + '_boxes.txt'
            image_name = []
            with open(txt_path) as txt:
                for line in txt:
                    image_name.append(line.strip('\n').split('\t')[0])
            image_names.append(image_name)
        # labels = np.arange(200)

        val_labels_t = []  # len 10000
        val_labels = []  # len 10000
        val_names = []  # len 10000
        with open(self.url + 'val\\val_annotations.txt') as txt:
            for line in txt:
                val_names.append(line.strip('\n').split('\t')[0])
                val_labels_t.append(line.strip('\n').split('\t')[1])
        for i in range(len(val_labels_t)):
            for i_t in range(len(labels_t)):
                if val_labels_t[i] == labels_t[i_t]:
                    val_labels.append(i_t)
        val_labels = np.array(val_labels)

        label = []
        image = []
        if self.type == 'train':
            label = index // 500
            image = self.images[index]
        if self.type == 'val':
            label = val_labels[index]
            image = self.val_images[index]
        image = Image.fromarray(image)
        return self.transform(image), label

    def __len__(self):
        len = 0
        if self.type == 'train':
            len = self.images.shape[0]
        if self.type == 'val':
            len = self.val_images.shape[0]
        return len
