#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.8

import copy
import torch
from torchvision import datasets, transforms
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal, mnist_noniid_lt
from sampling import femnist_iid, femnist_noniid, femnist_noniid_unequal, femnist_noniid_lt
from sampling import cifar_iid, cifar100_noniid, cifar10_noniid, cifar100_noniid_lt, cifar10_noniid_lt, cifar100_noniid_alpha
from sampling import seller_buyer_split, untruthful_single_class_seller_buyer_split, seller_buyer_split_cifar100
import femnist
import numpy as np
from collections import Counter

trans_cifar10_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                               std=[0.229, 0.224, 0.225])])
trans_cifar10_val = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])])
trans_cifar100_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                                               std=[0.267, 0.256, 0.276])])
trans_cifar100_val = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                                              std=[0.267, 0.256, 0.276])])

def get_dataset(args, n_list, k_list):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """
    data_dir = args.data_dir + args.dataset
    if args.dataset == 'mnist':
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)

        if args.iid:
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:
            test_labels = np.array(test_dataset.targets) 
            global_distribution = Counter(test_labels)  
            user_groups, classes_list, client_data_sizes, client_data_distributions = cifar100_noniid_alpha(args, train_dataset, args.num_users, args.dir_alpha, random_seed = args.data_random_seed)
            return train_dataset, test_dataset, user_groups, client_data_sizes, client_data_distributions, global_distribution

    elif args.dataset == 'femnist':
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = femnist.FEMNIST(args, data_dir, train=True, download=True,
                                        transform=apply_transform)
        test_dataset = femnist.FEMNIST(args, data_dir, train=False, download=True,
                                       transform=apply_transform)

        if args.iid:
            user_groups = femnist_iid(train_dataset, args.num_users)
        else:
            test_labels = np.array(test_dataset.targets) 
            global_distribution = Counter(test_labels)  
            user_groups, classes_list, client_data_sizes, client_data_distributions = cifar100_noniid_alpha(args, train_dataset, args.num_users, args.dir_alpha, random_seed = args.data_random_seed)
            return train_dataset, test_dataset, user_groups, client_data_sizes, client_data_distributions, global_distribution

    elif args.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=trans_cifar10_train)
        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=trans_cifar10_val)

        if args.iid:
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
            test_labels = np.array(test_dataset.targets)
            global_distribution = Counter(test_labels)
            user_groups, classes_list, client_data_sizes, client_data_distributions = cifar100_noniid_alpha(args, train_dataset, args.num_users, args.dir_alpha, random_seed = args.data_random_seed)
            return train_dataset, test_dataset, user_groups, client_data_sizes, client_data_distributions, global_distribution

    elif args.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(data_dir, train=True, download=True, transform=trans_cifar100_train)
        test_dataset = datasets.CIFAR100(data_dir, train=False, download=True, transform=trans_cifar100_val)

        test_labels = np.array(test_dataset.targets)
        global_distribution = Counter(test_labels)
        if args.iid:
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
            user_groups, classes_list, client_data_sizes, client_data_distributions = cifar100_noniid_alpha(args, train_dataset, args.num_users, args.dir_alpha, random_seed = args.data_random_seed)
            return train_dataset, test_dataset, user_groups, client_data_sizes, client_data_distributions, global_distribution

    elif args.dataset == 'fmnist':
        apply_transform = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize((0.1307,), (0.3081,))])
        
        train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True,
                                              transform=apply_transform)

        test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True,
                                             transform=apply_transform)

        if args.iid:
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:
            test_labels = np.array(test_dataset.targets) 
            global_distribution = Counter(test_labels)  
            user_groups, classes_list, client_data_sizes, client_data_distributions = cifar100_noniid_alpha(args, train_dataset, args.num_users, args.dir_alpha, random_seed = args.data_random_seed)
            return train_dataset, test_dataset, user_groups, client_data_sizes, client_data_distributions, global_distribution

    elif args.dataset == 'kmnist':
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))])

        train_dataset = datasets.KMNIST(data_dir, train=True, download=True,
                                        transform=apply_transform)

        test_dataset = datasets.KMNIST(data_dir, train=False, download=True,
                                       transform=apply_transform)

        if args.iid:
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:
            test_labels = np.array(test_dataset.targets) 
            global_distribution = Counter(test_labels)  
            user_groups, classes_list, client_data_sizes, client_data_distributions = cifar100_noniid_alpha(args, train_dataset, args.num_users, args.dir_alpha, random_seed = args.data_random_seed)
            return train_dataset, test_dataset, user_groups, client_data_sizes, client_data_distributions, global_distribution

    elif args.dataset == 'emnist-d':
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.EMNIST(data_dir, split='digits', train=True, download=True,
                                        transform=apply_transform)

        test_dataset = datasets.EMNIST(data_dir, split='digits', train=False, download=True,
                                       transform=apply_transform)

        if args.iid:
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:
            test_labels = np.array(test_dataset.targets) 
            global_distribution = Counter(test_labels)  
            user_groups, classes_list, client_data_sizes, client_data_distributions = cifar100_noniid_alpha(args, train_dataset, args.num_users, args.dir_alpha, random_seed = args.data_random_seed)
            return train_dataset, test_dataset, user_groups, client_data_sizes, client_data_distributions, global_distribution

    elif args.dataset == 'emnist-l':
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.EMNIST(data_dir, split='letters', train=True, download=True,
                                        transform=apply_transform)

        test_dataset = datasets.EMNIST(data_dir, split='letters', train=False, download=True,
                                       transform=apply_transform)

        if args.iid:
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:
            test_labels = np.array(test_dataset.targets) 
            global_distribution = Counter(test_labels)  
            user_groups, classes_list, client_data_sizes, client_data_distributions = cifar100_noniid_alpha(args, train_dataset, args.num_users, args.dir_alpha, random_seed = args.data_random_seed)
            return train_dataset, test_dataset, user_groups, client_data_sizes, client_data_distributions, global_distribution




def average_weights(w):
    """
    Returns the average of the weights as a single dictionary.
    """
    w_avg = copy.deepcopy(w[0]) 
    for key in w[0].keys():
        if key[:4] != '....': 
            for i in range(1, len(w)):
                w_avg[key] += w[i][key]
            w_avg[key] = torch.div(w_avg[key], len(w))
    
    return w_avg


def size_average_weights(local_weights, local_sizes):
    """
    Returns the weighted average of the weights as a single dictionary.
    """
    total_size = sum(local_sizes)
    w_avg = copy.deepcopy(local_weights[0])
    
    weights_contrib = [size / total_size for size in local_sizes]

    for key in w_avg.keys():
        w_avg[key] = torch.zeros_like(w_avg[key], dtype=torch.float32)
        for i in range(len(local_weights)):
            w_avg[key] += weights_contrib[i] * local_weights[i][key]
    
    return w_avg, weights_contrib



def current_weights_contrib_average(missing_seller_weights, current_weights_contrib):
    w_avg = copy.deepcopy(missing_seller_weights[0])
    for key in w_avg.keys():
        w_avg[key] = torch.zeros_like(w_avg[key], dtype=torch.float32)
        for i in range(len(missing_seller_weights)):
            w_avg[key] += current_weights_contrib[i] * missing_seller_weights[i][key]
    
    return w_avg



def average_weights_sem(w, n_list):
    """
    Returns the average of the weights.
    """
    k = 2
    model_dict = {}
    for i in range(k):
        model_dict[i] = []

    idx = 0
    for i in n_list:
        if i< np.mean(n_list):
            model_dict[0].append(idx)
        else:
            model_dict[1].append(idx)
        idx += 1

    ww = copy.deepcopy(w)
    for cluster_id in model_dict.keys():
        model_id_list = model_dict[cluster_id]
        w_avg = copy.deepcopy(w[model_id_list[0]])
        for key in w_avg.keys():
            for j in range(1, len(model_id_list)):
                w_avg[key] += w[model_id_list[j]][key]
            w_avg[key] = torch.true_divide(w_avg[key], len(model_id_list))
            # w_avg[key] = torch.div(w_avg[key], len(model_id_list))
        for model_id in model_id_list:
            for key in ww[model_id].keys():
                ww[model_id][key] = w_avg[key]

    return ww

def average_weights_per(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w)
    for key in w[0].keys():
        if key[0:2] != 'fc':
            for i in range(1, len(w)):
                w_avg[0][key] += w[i][key]
            w_avg[0][key] = torch.true_divide(w_avg[0][key], len(w))
            # w_avg[0][key] = torch.div(w_avg[0][key], len(w))
            for i in range(1, len(w)):
                w_avg[i][key] = w_avg[0][key]
    return w_avg

def average_weights_het(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w)
    for key in w[0].keys():
        if key[0:4] != 'fc2.':
            for i in range(1, len(w)):
                w_avg[0][key] += w[i][key]
            # w_avg[0][key] = torch.true_divide(w_avg[0][key], len(w))
            w_avg[0][key] = torch.div(w_avg[0][key], len(w))
            for i in range(1, len(w)):
                w_avg[i][key] = w_avg[0][key]
    return w_avg

def agg_func(protos):
    """
    Returns the average of the weights.
    """

    for [label, proto_list] in protos.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            protos[label] = proto / len(proto_list)
        else:
            protos[label] = proto_list[0]

    return protos

def proto_aggregation(local_protos_list):
    agg_protos_label = dict()
    for idx in local_protos_list:
        local_protos = local_protos_list[idx]
        for label in local_protos.keys():
            if label in agg_protos_label:
                agg_protos_label[label].append(local_protos[label])
            else:
                agg_protos_label[label] = [local_protos[label]]

    for [label, proto_list] in agg_protos_label.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            agg_protos_label[label] = [proto / len(proto_list)]
        else:
            agg_protos_label[label] = [proto_list[0].data]

    return agg_protos_label


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Dataset   : {args.dataset}')
    print(f'    Global Rounds   : {args.rounds}')
    print(f'    Local Episodes   : {args.train_ep}')
    print(f'    Local Batch size  : {args.local_bs}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Momentum  : {args.momentum}')
    
    if args.auction:
        print('    \nAuction parameters:')
        print(f'    {args.num_users} sellers')
        print(f'    {args.winners} winners')
        if args.iid:
            print('    IID')
        else:
            print(f'    Non-IID: {args.dir_alpha} dirichlet / {args.data_random_seed} random_seed')
        if args.comp_cost:
            print(f'    Client Compute Cost: {args.comp_cost_random_seed} random_seed')
        if args.data_cost:
            # print('    Client Data Cost')
            if args.data_cost_L2:
                print('    Client Data Cost L2')
            elif args.data_cost_KL:
                print('    Client Data Cost KL')
        if args.client_random_cost:
            print('   Client Random Cost')
        if args.aggregation_cost:
            print('    Server Aggregation Cost')
            print(f'        Per_model_cal_cost: {args.per_model_cal_cost}')
        if args.self_noise:  
            print('    All clients design their own noise')  
        else:
            print('    We desing the noise for all clients: ')    
            print(f'        Noise std: {args.noise_std_dev}')
            print(f'        Noise mean: {args.noise_mean}')
        if args.causal:
            print('    Causal') 
            print(f'        Causal num layers: {args.causal_num_layers}')
            print(f'        Causal num epochs: {args.causal_num_epochs}')
            print(f'        Causal lr: {args.causal_learning_rate}')
            print(f'        Causal lr decay: {args.causal_learning_rate_decay}')
            print(f'        Causal weight decay: {args.causal_weight_decay}')  
    return