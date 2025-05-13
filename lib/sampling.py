#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.8


import numpy as np
from torchvision import datasets, transforms
import random
import torch


def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def mnist_noniid(args, dataset, num_users, n_list, k_list):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """

    # 60,000 training imgs -->  200 imgs/shard X 300 shards
    num_shards, num_imgs = 10, 6000
    idx_shard = [i for i in range(num_shards)]
    dict_users = {}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()
    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    label_begin = {}
    cnt=0
    for i in idxs_labels[1,:]:
        if i not in label_begin:
                label_begin[i] = cnt
        cnt+=1

    classes_list = []
    for i in range(num_users):
        n = n_list[i]
        k = k_list[i]
        k_len = args.train_shots_max
        classes = random.sample(range(0,args.num_classes), n)
        classes = np.sort(classes)
        print("user {:d}: {:d}-way {:d}-shot".format(i + 1, n, k))
        print("classes:", classes)
        user_data = np.array([])
        for each_class in classes:
            # begin = i*10 + label_begin[each_class.item()]
            begin = i * k_len + label_begin[each_class.item()]
            user_data = np.concatenate((user_data, idxs[begin : begin+k]),axis=0)
        dict_users[i] = user_data
        classes_list.append(classes)

    return dict_users, classes_list


def mnist_noniid_lt(args, test_dataset, num_users, n_list, k_list, classes_list):
    num_shards, num_imgs = 10, 1000
    idx_shard = [i for i in range(num_shards)]
    dict_users = {}
    idxs = np.arange(num_shards*num_imgs)
    labels = test_dataset.train_labels.numpy()
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    label_begin = {}
    cnt=0
    for i in idxs_labels[1,:]:
        if i not in label_begin:
                label_begin[i] = cnt
        cnt+=1

    for i in range(num_users):
        k = 40 
        classes = classes_list[i]
        print("local test classes:", classes)
        user_data = np.array([])
        for each_class in classes:
            begin = i*40 + label_begin[each_class.item()]
            user_data = np.concatenate((user_data, idxs[begin : begin+k]),axis=0)
        dict_users[i] = user_data


    return dict_users


def mnist_noniid_unequal(dataset, num_users):

    num_shards, num_imgs = 1200, 50
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    min_shard = 1
    max_shard = 30

    random_shard_size = np.random.randint(min_shard, max_shard+1,
                                          size=num_users)
    random_shard_size = np.around(random_shard_size /
                                  sum(random_shard_size) * num_shards)
    random_shard_size = random_shard_size.astype(int)

    if sum(random_shard_size) > num_shards:

        for i in range(num_users):
            rand_set = set(np.random.choice(idx_shard, 1, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

        random_shard_size = random_shard_size-1

        for i in range(num_users):
            if len(idx_shard) == 0:
                continue
            shard_size = random_shard_size[i]
            if shard_size > len(idx_shard):
                shard_size = len(idx_shard)
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)
    else:

        for i in range(num_users):
            shard_size = random_shard_size[i]
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

        if len(idx_shard) > 0:
            shard_size = len(idx_shard)
            k = min(dict_users, key=lambda x: len(dict_users.get(x)))
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            for rand in rand_set:
                dict_users[k] = np.concatenate(
                    (dict_users[k], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

    return dict_users


def femnist_iid(dataset, num_users):
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def femnist_noniid(args, num_users, n_list, k_list):
    dict_users = {}
    classes_list = []
    classes_list_gt = []

    for i in range(num_users):
        n = n_list[i]
        k = k_list[i]
        k_len = args.train_shots_max
        classes = random.sample(range(0, args.num_classes), n)
        classes = np.sort(classes)
        print("user {:d}: {:d}-way {:d}-shot".format(i + 1, n, k))
        print("classes:", classes)
        print("classes_gt:", classes)
        user_data = np.array([])
        for class_idx in classes:
            begin = class_idx * k_len * num_users + i * k_len
            user_data = np.concatenate((user_data, np.arange(begin, begin + k)),axis=0)
        dict_users[i] = user_data
        classes_list.append(classes)
        classes_list_gt.append(classes)

    return dict_users, classes_list, classes_list_gt

def femnist_noniid_lt(args, num_users, classes_list):
    dict_users = {}

    for i in range(num_users):
        k = args.test_shots
        classes = classes_list[i]
        user_data = np.array([])
        for class_idx in classes:
            begin = class_idx * k * num_users + i * k
            user_data = np.concatenate((user_data, np.arange(begin, begin + k)), axis=0)
        dict_users[i] = user_data

    return dict_users


def femnist_noniid_unequal(dataset, num_users):

    num_shards, num_imgs = 1200, 50
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    min_shard = 1
    max_shard = 30

    random_shard_size = np.random.randint(min_shard, max_shard+1,
                                          size=num_users)
    random_shard_size = np.around(random_shard_size /
                                  sum(random_shard_size) * num_shards)
    random_shard_size = random_shard_size.astype(int)

    if sum(random_shard_size) > num_shards:

        for i in range(num_users):
            rand_set = set(np.random.choice(idx_shard, 1, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

        random_shard_size = random_shard_size-1

        for i in range(num_users):
            if len(idx_shard) == 0:
                continue
            shard_size = random_shard_size[i]
            if shard_size > len(idx_shard):
                shard_size = len(idx_shard)
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)
    else:

        for i in range(num_users):
            shard_size = random_shard_size[i]
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

        if len(idx_shard) > 0:
            shard_size = len(idx_shard)
            k = min(dict_users, key=lambda x: len(dict_users.get(x)))
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            for rand in rand_set:
                dict_users[k] = np.concatenate(
                    (dict_users[k], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

    return dict_users

def cifar10_noniid(args, dataset, num_users, n_list, k_list):
    num_shards, num_imgs = 10, 5000
    dict_users = {}
    idxs = np.arange(num_shards * num_imgs)
    labels = np.array(dataset.targets)
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    label_begin = {}
    cnt = 0
    for i in idxs_labels[1, :]:
        if i not in label_begin:
            label_begin[i] = cnt
        cnt += 1

    classes_list = []
    classes_list_gt = []
    k_len = args.train_shots_max
    for i in range(num_users):
        n = n_list[i]
        k = k_list[i]
        classes = random.sample(range(0, args.num_classes), n)
        classes = np.sort(classes)
        print("user {:d}: {:d}-way {:d}-shot".format(i + 1, n, k))
        print("classes:", classes)
        user_data = np.array([])
        for each_class in classes:
            begin = i * k_len + label_begin[each_class.item()]
            user_data = np.concatenate((user_data, idxs[begin: begin + k]), axis=0)
        dict_users[i] = user_data
        classes_list.append(classes)
        classes_list_gt.append(classes)

    return dict_users, classes_list, classes_list_gt

def cifar10_noniid_lt(args, test_dataset, num_users, n_list, k_list, classes_list):
    num_shards, num_imgs = 10, 1000
    dict_users = {}
    idxs = np.arange(num_shards*num_imgs)
    labels = np.array(test_dataset.targets)
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    label_begin = {}
    cnt=0
    for i in idxs_labels[1,:]:
        if i not in label_begin:
                label_begin[i] = cnt
        cnt+=1

    for i in range(num_users):
        k = args.test_shots
        classes = classes_list[i]
        print("local test classes:", classes)
        user_data = np.array([])
        for each_class in classes:
            begin = i * k + label_begin[each_class.item()]
            user_data = np.concatenate((user_data, idxs[begin : begin+k]),axis=0)
        dict_users[i] = user_data


    return dict_users


def cifar_iid(dataset, num_users):
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users



def cifar100_noniid(args, dataset, num_users, n_list, k_list):
    num_shards, num_imgs = 100, 500
    dict_users = {}
    idxs = np.arange(num_shards*num_imgs)
    labels = np.array(dataset.targets)
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    label_begin = {}
    cnt=0
    for i in idxs_labels[1,:]:
        if i not in label_begin:
                label_begin[i] = cnt
        cnt+=1

    classes_list = []
    for i in range(num_users):
        n = n_list[i]
        k = k_list[i]
        classes = random.sample(range(0,args.num_classes), n)
        classes = np.sort(classes)
        print("user {:d}: {:d}-way {:d}-shot".format(i + 1, n, k))
        print("classes:", classes)
        user_data = np.array([])
        for each_class in classes:
            begin = label_begin[each_class.item()] + i*5
            user_data = np.concatenate((user_data, idxs[begin : begin+k]),axis=0)
        dict_users[i] = user_data
        classes_list.append(classes)

    return dict_users, classes_list


def cifar100_noniid_lt(test_dataset, num_users, classes_list):
    num_shards, num_imgs = 100, 100
    idx_shard = [i for i in range(num_shards)]
    dict_users = {}
    idxs = np.arange(num_shards*num_imgs)
    labels = np.array(test_dataset.targets)
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    label_begin = {}
    cnt=0
    for i in idxs_labels[1,:]:
        if i not in label_begin:
                label_begin[i] = cnt
        cnt+=1

    for i in range(num_users):
        k = 5
        classes = classes_list[i]
        print("local test classes:", classes)
        user_data = np.array([])
        for each_class in classes:
            begin = random.randint(0,90) + label_begin[each_class.item()]
            user_data = np.concatenate((user_data, idxs[begin : begin+k]),axis=0)
        dict_users[i] = user_data


    return dict_users


def seller1_buyer_split(dataset, num_users, num_classes, num_sellers):
    seller_idx = 0
    buyer_idxs = [i for i in range(1, num_users)]
    dict_users = {} 
    idxs = np.arange(len(dataset))
    labels = np.array(dataset.targets)

    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    half_classes = num_classes // 2

    seller_classes = random.sample(range(num_classes), half_classes)
    buyer_classes = [cls for cls in range(num_classes) if cls not in seller_classes]

    seller_data = []
    for cls in seller_classes:
        cls_indices = idxs[labels == cls]
        seller_data.extend(cls_indices)
    dict_users[seller_idx] = np.array(seller_data)

    for buyer_idx in buyer_idxs:
        buyer_data = []
        selected_classes = random.sample(buyer_classes, 3)
        for cls in selected_classes:
            cls_indices = idxs[labels == cls]
            buyer_data.extend(np.random.choice(cls_indices, size=1000, replace=False))
        dict_users[buyer_idx] = np.array(buyer_data)

    return dict_users


def seller_buyer_split(dataset, num_users, num_classes, num_sellers):
    num_buyers = num_users - num_sellers

    dict_users = {} 
    idxs = np.arange(len(dataset))
    labels = np.array(dataset.targets)

    total_data = len(idxs)
    min_buyer_data = total_data // (10 * num_users)
    max_buyer_data = total_data // (5 * num_users)
    buyer_data_indices = []

    for buyer_idx in range(num_sellers, num_users):
        num_samples = random.randint(min_buyer_data, max_buyer_data)
        buyer_indices = np.random.choice(idxs, size=num_samples, replace=False)
        dict_users[buyer_idx] = buyer_indices
        buyer_data_indices.extend(buyer_indices)
        idxs = np.setdiff1d(idxs, buyer_indices)

    remaining_data = idxs
    remaining_labels = labels[remaining_data]
    remaining_data_count = len(remaining_data)
    per_class_count = {cls: np.sum(remaining_labels == cls) for cls in range(num_classes)}

    classes_per_part = num_classes // 2
    parts = []
    part_classes_list = []
    for _ in range(5):
        part_classes = random.sample(range(num_classes), k=random.randint(classes_per_part - 1, classes_per_part + 1))
        part_classes_list.append(part_classes)
        part_indices = []
        for cls in part_classes:
            cls_indices = remaining_data[remaining_labels == cls]
            part_indices.extend(cls_indices)
        
        min_part_data = (remaining_data_count - min(per_class_count.values())) // 5
        max_part_data = remaining_data_count // 5
        num_samples = random.randint(min_part_data, max_part_data)
        if len(part_indices) < num_samples:
            print(f"Warning: Requested {num_samples} samples but only {len(part_indices)} available. Adjusting sample size.")
        part_indices = np.random.choice(part_indices, size=num_samples, replace=False)
        parts.append(part_indices)

    # Step 5: Assign parts to sellers
    for seller_idx in range(num_sellers):
        dict_users[seller_idx] = parts[seller_idx]
        print(f"Seller {seller_idx} assigned classes: {part_classes_list[seller_idx]}")

    return dict_users


def untruthful_single_class_seller_buyer_split(dataset, num_users, num_classes, num_sellers, untruthful_seller_ids):

    if isinstance(untruthful_seller_ids, str):
        untruthful_seller_ids = list(map(int, untruthful_seller_ids.split(",")))
    else:
        untruthful_seller_ids = list(map(int, untruthful_seller_ids))

    dict_users = {}
    idxs = np.arange(len(dataset))
    labels = np.array(dataset.targets) 


    total_data = len(idxs)
    min_buyer_data = total_data // (10 * num_users)
    max_buyer_data = total_data // (5 * num_users)
    buyer_data_indices = []

    for buyer_idx in range(num_sellers, num_users): 
        num_samples = random.randint(min_buyer_data, max_buyer_data)
        buyer_indices = np.random.choice(idxs, size=num_samples, replace=False)
        dict_users[buyer_idx] = buyer_indices
        buyer_data_indices.extend(buyer_indices)
        idxs = np.setdiff1d(idxs, buyer_indices)

    remaining_data = idxs
    remaining_labels = labels[remaining_data]
    remaining_data_count = len(remaining_data)
    per_class_count = {cls: np.sum(remaining_labels == cls) for cls in range(num_classes)}


    for seller_idx in untruthful_seller_ids:
        single_class = random.choice([cls for cls in range(num_classes) if per_class_count[cls] > 0])
        class_indices = remaining_data[remaining_labels == single_class]
        min_part_data = len(class_indices) // num_sellers
        max_part_data = len(class_indices)
        num_samples = random.randint(min_part_data, max_part_data)
        dict_users[seller_idx] = np.random.choice(class_indices, size=num_samples, replace=False)
        remaining_data = np.setdiff1d(remaining_data, dict_users[seller_idx])
        remaining_labels = labels[remaining_data]
        per_class_count[single_class] -= num_samples
        print(f"Untruthful Seller {seller_idx} assigned single class {single_class} with {num_samples} samples.")

    truthful_sellers = [i for i in range(num_sellers) if i not in set(untruthful_seller_ids)]
    num_truthful_sellers = len(truthful_sellers)
    num_parts = 5 - len(untruthful_seller_ids)
    if num_parts <= 0:
        raise ValueError("Not enough parts to allocate to truthful sellers.")

    parts = []
    part_classes_list = []

    for _ in range(num_parts):
        part_classes = random.sample([cls for cls in range(num_classes) if per_class_count[cls] > 0], 
                                     k=random.randint(num_classes // 2 - 1, num_classes // 2 + 1))
        part_classes_list.append(part_classes)
        part_indices = []
        for cls in part_classes:
            cls_indices = remaining_data[remaining_labels == cls]
            part_indices.extend(cls_indices)

        min_part_data = (remaining_data_count - min(per_class_count.values())) // num_parts
        max_part_data = remaining_data_count // num_parts
        num_samples = random.randint(min_part_data, max_part_data)
        part_indices = np.random.choice(part_indices, size=min(len(part_indices), num_samples), replace=False)
        parts.append(part_indices)

    for idx, seller_idx in enumerate(truthful_sellers[:num_truthful_sellers]):
        dict_users[seller_idx] = parts[idx]
        print(f"Truthful Seller {seller_idx} assigned classes: {part_classes_list[idx]} with {len(parts[idx])} samples.")


    return dict_users


def seller_buyer_split_cifar100(dataset, num_users, num_classes, num_sellers):
    num_buyers = num_users - num_sellers

    dict_users = {} 
    idxs = np.arange(len(dataset))
    labels = np.array(dataset.targets) 

    total_data = len(idxs)
    min_buyer_data = total_data // (2 * num_users)
    max_buyer_data = total_data // (num_users)
    buyer_data_indices = []

    for buyer_idx in range(num_sellers, num_users): 
        num_samples = random.randint(min_buyer_data, max_buyer_data)
        buyer_indices = np.random.choice(idxs, size=num_samples, replace=False)
        dict_users[buyer_idx] = buyer_indices
        buyer_data_indices.extend(buyer_indices)
        idxs = np.setdiff1d(idxs, buyer_indices)

    remaining_data = idxs
    remaining_labels = labels[remaining_data]
    remaining_data_count = len(remaining_data)
    per_class_count = {cls: np.sum(remaining_labels == cls) for cls in range(num_classes)}

    classes_per_part = num_classes // 2
    parts = []
    part_classes_list = []
    for _ in range(5):
        part_classes = random.sample(range(num_classes), k=random.randint(classes_per_part - 2, classes_per_part + 2))
        part_classes_list.append(part_classes)
        part_indices = []
        for cls in part_classes:
            cls_indices = remaining_data[remaining_labels == cls]
            part_indices.extend(cls_indices)
        
        min_part_data = (remaining_data_count - min(per_class_count.values())) // 5
        max_part_data = remaining_data_count // 5
        num_samples = random.randint(min_part_data, max_part_data)
        part_indices = np.random.choice(part_indices, size=num_samples, replace=False)
        parts.append(part_indices)

    for seller_idx in range(num_sellers):
        dict_users[seller_idx] = parts[seller_idx]
        print(f"Seller {seller_idx} assigned classes: {part_classes_list[seller_idx]}")

    return dict_users

import matplotlib.pyplot as plt
import os


def cifar100_noniid_alpha(args, dataset, num_users, alpha=0.5, save_dir="client_distributions", random_seed=42):
    if random_seed is not None:
        np.random.seed(random_seed)
    
    os.makedirs(save_dir, exist_ok=True)

    total_samples = len(dataset.targets)
    num_classes = args.num_classes
    max_samples_per_client = total_samples // num_users
    labels = np.array(dataset.targets)

    idxs = np.arange(total_samples)
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    label_begin = {}
    cnt = 0
    for i in idxs_labels[1, :]:
        if i not in label_begin:
            label_begin[i] = cnt
        cnt += 1

    classes_list = []
    user_stats = []
    dict_users = {}
    client_data_sizes = [] 
    data_distributions = []
    remaining_idxs = set(idxs)

    for i in range(num_users):
        client_data_size = np.random.randint(args.local_bs*2, max_samples_per_client + 1)
        client_data_sizes.append(client_data_size)
        
        proportions = np.random.dirichlet(np.ones(num_classes) * alpha, 1).flatten()
        
        user_data = np.array([], dtype=int)
        class_data_sizes = {}
        
        for class_idx in range(num_classes):
            num_images = round(proportions[class_idx] * client_data_size)
            if num_images > 0 and class_idx in label_begin:
                begin = label_begin[class_idx]
                selected_idxs = np.intersect1d(
                    idxs[begin: begin + num_images], list(remaining_idxs), assume_unique=True
                )
                user_data = np.concatenate((user_data, selected_idxs), axis=0)
                class_data_sizes[class_idx] = len(selected_idxs)
                label_begin[class_idx] += len(selected_idxs)
                remaining_idxs -= set(selected_idxs)
            
        remaining_needed = client_data_size - len(user_data)
        if remaining_needed > 0:
            remaining_samples = list(remaining_idxs)[:remaining_needed]
            user_data = np.concatenate((user_data, remaining_samples), axis=0)
            remaining_idxs -= set(remaining_samples)

        
        dict_users[i] = user_data
        user_classes = np.where(proportions > 0)[0]  
        classes_list.append(user_classes)
        
        user_stats.append({
            "user_id": i,
            "data_size": len(user_data),
            "classes": list(class_data_sizes.keys()),
            "class_data_sizes": class_data_sizes
        })
        data_distributions.append(class_data_sizes)
    print("Client Data Statistics:")
    for user in user_stats:
        print(f"Client {user['user_id']}:")
        print(f"  Total Data Size: {user['data_size']}")
        print(f"  Classes: {user['classes']}")
        print(f"  Class Data Sizes: {user['class_data_sizes']}")
        print("-" * 30)
    return dict_users, classes_list, client_data_sizes, data_distributions

if __name__ == '__main__':
    dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,),
                                                            (0.3081,))
                                   ]))
    num = 100
    d = mnist_noniid(dataset_train, num)
