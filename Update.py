import torch
from torch import nn

import torch.nn.functional as F

from utils.options import args_parser

args = args_parser()


class LocalUpdate(object):
    def __init__(self, args, train_loader=None, test_loader=None, test_sample=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()

        self.ldr_train = train_loader
        self.ldr_test = test_loader
        self.test_sample_num = test_sample

    def train(self, net):
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        epoch_loss = []

        '''local training'''
        net.train()
        for iter in range(self.args.local_ep):
            batch_loss = []

            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                labels = labels.to(torch.long)
                images = images.to(self.args.device)
                labels = labels.to(self.args.device)

                net.zero_grad()
                log_probs = net(images)

                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        '''test'''
        net.eval()
        test_loss = 0
        correct = 0

        for idx, (data, target) in enumerate(self.ldr_test):
            target = target.to(torch.long)
            data = data.to(self.args.device)
            target = target.to(self.args.device)

            log_probs = net(data)
            # sum up batch loss
            test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
            # get the index of the max log-probability
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

        test_loss /= self.test_sample_num
        accuracy = 100.00 * correct / self.test_sample_num

        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, self.test_sample_num, accuracy))

        return sum(epoch_loss) / len(epoch_loss), accuracy, net.state_dict()
