#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.8

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import numpy as np

class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class."""

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs):
        self.args = args
        self.trainloader = self.train_val_test(dataset, list(idxs))
        self.device = args.device
        # self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.criterion = nn.NLLLoss().to(self.device)

    def train_val_test(self, dataset, idxs):
        """
        Returns train dataloader for a given dataset
        and user indexes.
        """
        idxs_train = idxs[:int(1 * len(idxs))]
        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.local_bs, shuffle=True, drop_last=True)

        return trainloader

    def update_weights(self, idx, model, global_round):
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)
        last_epoch_loss = 0  
        last_epoch_acc = 0   
        
        for iter in range(self.args.train_ep):
            batch_loss = []
            batch_acc = []  
            
            for batch_idx, (images, labels_g) in enumerate(self.trainloader):
                if len(self.trainloader.dataset) == 0:
                    print("Warning: The training dataset is empty!")
                # else:
                #     print(f"Training dataset has {len(self.trainloader.dataset)} samples.")

                images, labels = images.to(self.device), labels_g.to(self.device)

                model.zero_grad()
                log_probs ,protos= model(images)
                # labels = labels.long()
                # print("Labels range:", labels.min().item(), labels.max().item())
                loss = self.criterion(log_probs, labels)
                loss.backward()
                optimizer.step()

                _, y_hat = log_probs.max(1)
                acc_val = torch.eq(y_hat, labels.squeeze()).float().mean()

                # if self.args.verbose and (batch_idx % 10 == 0):
                if self.args.verbose:
                    print('| Global Round : {} | User: {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.3f} | Acc: {:.5f}'.format(
                        global_round, idx, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader),
                        loss.item(),
                        acc_val.item()))
                batch_loss.append(loss.item())
                batch_acc.append(acc_val.item())
            # epoch_loss.append(sum(batch_loss)/len(batch_loss))
            if len(batch_loss) > 0:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
                epoch_acc = sum(batch_acc) / len(batch_acc) 
            else:
                print("Warning: batch_loss is empty, skipping this epoch.")
                epoch_loss.append(0)
                epoch_acc = 0
            if iter == self.args.train_ep - 1:
                last_epoch_loss = epoch_loss[-1]
                last_epoch_acc = epoch_acc
        # return model.state_dict(), sum(epoch_loss) / len(epoch_loss), acc_val.item(), protos
        return model.state_dict(), last_epoch_loss, last_epoch_acc


def inference(args, model, test_dataset):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = args.device
    criterion = nn.NLLLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=128,
                            shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        outputs, protos = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)
    average_loss = loss / len(testloader) 
    accuracy = correct/total
    return accuracy, average_loss
