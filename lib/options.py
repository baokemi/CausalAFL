#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.8

import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--rounds', type=int, default=200,
                        help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=20,
                        help="number of users, contain seller and buyers")
    parser.add_argument('--winners', type=int, default=10,
                    help="number of winner: K")
    parser.add_argument('--train_ep', type=int, default=5,
                        help="the number of local episodes: E")
    parser.add_argument('--local_bs', type=int, default=512,
                        help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--device', type=str, default='cuda',
                    help='device to use for training (cuda or cpu)')
    parser.add_argument('--gpu', type=int, default=0, help="To use cuda, set \
                        to a specific GPU ID. Default set to use CPU.")

    # model arguments
    parser.add_argument('--num_channels', type=int, default=1, help="number \
                        of channels of imgs")
    parser.add_argument('--norm', type=str, default='batch_norm',
                        help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32,
                        help="number of filters for conv nets -- 32 for \
                        mini-imagenet, 64 for omiglot.")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than \
                        strided convolutions")
    parser.add_argument('--stride', type=int, default=1,
                        help="stride of CNN")

    # other arguments
    parser.add_argument('--data_dir', type=str, default='../data/', help="directory of dataset")
    parser.add_argument('--dataset', type=str, default='cifar10', help="name \
                        of dataset: cifar10, cifar100, fmnist, emnist-d, emnist-l")
    parser.add_argument('--num_classes', type=int, default=10, help="number \
                        of classes, cifar10 is 10, cifar100 is 100, fmnist is 10, emnist-d is 10, emnist-l is 26.")
    parser.add_argument('--optimizer', type=str, default='sgd', help="type \
                        of optimizer")
    parser.add_argument('--iid', type=int, default=0,
                        help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--unequal', type=int, default=0,
                        help='whether to use unequal data splits for  \
                        non-i.i.d setting (use 0 for equal splits)')
    parser.add_argument('--verbose', type=int, default=1, help='verbose')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--dir_alpha', type=float, default=0.1, help='dirichlet parameter')

    # Local arguments
    parser.add_argument('--ways', type=int, default=5, help="num of classes")
    parser.add_argument('--shots', type=int, default=100, help="num of shots")
    parser.add_argument('--train_shots_max', type=int, default=110, help="num of shots")
    parser.add_argument('--test_shots', type=int, default=15, help="num of shots")
    parser.add_argument('--stdev', type=int, default=1, help="stdev of ways")
    
    # auction
    # parser.add_argument('--auction', action='store_true', help="Whether to run the auction")
    parser.add_argument('--auction', type=bool, default='True', help="Whether to run the auction")

    # cost
    parser.add_argument('--comp_cost', action='store_true', help="Whether comp_cost")
    parser.add_argument('--data_cost', action='store_true',  help="Whether data_cost")
    parser.add_argument('--data_cost_L2', action='store_true',  help="Whether data_cost_L2")
    parser.add_argument('--data_cost_KL', action='store_true',  help="Whether data_cost_KL")
    parser.add_argument('--client_random_cost', action='store_true', help="Whether random_cost")
    parser.add_argument('--aggregation_cost', action='store_true', help="Whether aggregation_cost")
    parser.add_argument('--per_model_cal_cost', type=float, default=0.0001, help='per model calculate cost')
    
    # payment
    parser.add_argument('--payment', type=int, default=20,
                        help="Server pay to winners")
    
    # causal
    parser.add_argument('--causal', action='store_true', help="Whether to causal analyze")
    parser.add_argument('--self_noise', action='store_true', help="Whether client to determine the noise")
    parser.add_argument('--noise_mean', type=float, default=1, help='noise_mean')
    parser.add_argument('--noise_std_dev', type=float, default=0.5, help='noise_std_dev')
    parser.add_argument('--causal_num_layers', type=int, default=4, help='causal num_layers')
    parser.add_argument('--causal_num_epochs', type=int, default=10, help='causal num_epochs')  
    parser.add_argument('--causal_learning_rate', type=float, default=1e-2, help='causal learning_rate')
    parser.add_argument('--causal_learning_rate_decay', type=float, default=0.01, help='causal learning_rate_decay')        
    parser.add_argument('--causal_weight_decay', type=float, default=1e-4, help='causal weight_decay')   
    
    # Seed 
    parser.add_argument('--data_random_seed', type=int, default=42, help="Data non-iid random seed")
    parser.add_argument('--comp_cost_random_seed', type=int, default=42, help="Compute cost random seed")  
    
    args = parser.parse_args()
    return args
