#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.8

import copy, sys, os, json
import numpy as np
from tqdm import tqdm
import torch
import torch.utils.model_zoo as model_zoo
from pathlib import Path
import random

lib_dir = (Path(__file__).parent / "." / "lib").resolve()
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))
mod_dir = (Path(__file__).parent / "." / "lib" / "models").resolve()
if str(mod_dir) not in sys.path:
    sys.path.insert(0, str(mod_dir))

from lib.resnet import resnet18
from lib.models import CNNFashion_Mnist, CNNKMNIST, CNNEMNIST, CNNEMNIST_L
from lib.options import args_parser
from lib.update import LocalUpdate, inference
from lib.utils import get_dataset, average_weights, exp_details, agg_func, size_average_weights, reputation_weighted_average, current_weights_contrib_average
from lib.eval_fixed import reverse_auction_eval_fixed_k_profit, reverse_auction_bid_eval_fixed_k_profit
from lib.auction import generate_values, generate_truthful_bids, emulate_untruthful

from causal.cost import generate_comp_cost, generate_data_cost, generate_data_cost_KL
import itertools
from causal.TMcausal import est_all_thetas, get_shap_val
import pyro
from causal.dmavae_gpu import DMA_VAE
VALUE_RANGE = (0, 1)
VAL_TIMES = 20




def convert_numpy_to_python(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {int(k) if isinstance(k, np.integer) else k: convert_numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_python(item) for item in obj]
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    return obj


def get_ranking(performances, bids):
    scores = performances * bids
    return np.argsort(scores)[::-1] 

def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.numel() * param.element_size() 
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.numel() * buffer.element_size() 
    
    total_size = param_size + buffer_size  
    return total_size / (1024**2)  


def Auction_FedAvg_task(args, train_dataset, test_dataset, user_groups, client_data_sizes, client_data_distributions, global_distribution):
    # args.stride = [1,1]
    args.stride = [args.stride, args.stride]
    global_model = resnet18(args, pretrained=False)
    if args.dataset == 'fmnist':
        global_model = CNNFashion_Mnist(args)
    elif args.dataset == 'kmnist':
        global_model = CNNKMNIST(args)
    elif args.dataset == 'emnist-d':
        global_model = CNNEMNIST(args)
    elif args.dataset == 'emnist-l':
        global_model = CNNEMNIST_L(args)
    global_model = global_model.to(args.device)
    
    try: 
        global_model.train()
        client_idxs = np.arange(args.num_users)
        

        if args.client_random_cost:
            client_random_cost = generate_values(args.num_users)
        else: 
            client_random_cost = np.zeros(args.num_users)

            if args.comp_cost:
                comp_cost = generate_comp_cost(args.num_users, client_data_sizes, args.comp_cost_random_seed)
            else:
                comp_cost = np.zeros(args.num_users)

            if args.data_cost:
                if args.data_cost_L2:
                    data_cost = generate_data_cost(args.num_users, client_data_distributions, global_distribution)
                elif args.data_cost_KL:
                    data_cost = generate_data_cost_KL(args.num_users, client_data_distributions, global_distribution)
            else:
                data_cost = np.zeros(args.num_users)
    
        
        global_model_size = get_model_size(global_model)

        normalized_ATE = np.zeros((args.rounds, args.num_users))
        for round in tqdm(range(args.rounds)):
            local_weights, local_losses, local_performances, local_protos = [], [], [], []
            local_sizes = []
            print(f'\n | Global Training Round : {round} |\n')

            if round  < 10:
                divisor = 10
            elif round < 100:
                divisor = 200
            elif round < 200:
                divisor = 500
            else:
                divisor = 1000    
            print(f'divisor: {divisor}')
            client_cost = comp_cost * (round + 1) / divisor + data_cost + client_random_cost

            if args.self_noise:
                mean_client_cost = np.mean(client_cost)
                std_dev_client_cost = np.std(client_cost)
                noise = np.random.normal(mean_client_cost, std_dev_client_cost, size=args.num_users)
            else:
                mean_client_cost = args.noise_mean
                std_dev_client_cost = args.noise_std_dev
                noise = np.random.normal(mean_client_cost, std_dev_client_cost, size=args.num_users)
            if args.causal and round > 0:
                client_bids = client_cost + noise  +  (round + 1) / 10 * normalized_ATE[round - 1]
            else:
                client_bids = client_cost + noise 

            global_model.train()
            client_training_metrics = {}
            for idx in client_idxs:
                local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx])
                w, loss, acc = local_model.update_weights(idx=idx, model=copy.deepcopy(global_model), global_round=round)
                local_weights.append(copy.deepcopy(w))
                local_losses.append(copy.deepcopy(loss))
                local_performances.append(acc)
                local_sizes.append(len(user_groups[idx]))
                client_training_metrics[idx] = {
                    "train_loss": loss,
                    "train_accuracy": acc
                }
            K = args.winners
            client_performances = np.array(local_performances)
            
            winners, sorted_clients = reverse_auction_bid_eval_fixed_k_profit(client_bids, K)
            for winner_idx in winners:
                train_loss_idx = client_training_metrics[winner_idx]["train_loss"]
                train_acc_idx = client_training_metrics[winner_idx]["train_accuracy"]
                print(f'Winner Buyer Client {winner_idx} Training Loss: {train_loss_idx:.4f}, Training Accuracy: {train_acc_idx:.2f}, bid: {client_bids[winner_idx]:.2f}')
            
            winners_weights = [local_weights[winner] for winner in winners]
            winners_sizes = [local_sizes[winner] for winner in winners]
            global_weights, _ = size_average_weights(winners_weights, winners_sizes) 
            global_model.load_state_dict(global_weights)
            test_acc, test_loss = inference(args, global_model, test_dataset)
            test_acc_with_winner = test_acc * 100

            payments = np.zeros(args.num_users)
            sum_payments = args.payment
            
            if args.aggregation_cost and args.per_model_cal_cost != 0: 
                aggregation_cost = global_model_size * args.winners * args.per_model_cal_cost
            else:
                aggregation_cost = 0
            utility_server = test_acc * 1000 - aggregation_cost - sum_payments

            utility_winners = sum_payments - np.sum(np.fromiter((client_cost[x] for x in winners), dtype=float))

            social_welfare = utility_server + utility_winners
            
            
            marginal_contribution = np.zeros(args.num_users)  
            all_marginal_contribution = 0
            total_payment_relative_value = 0
            payment = np.zeros(args.num_users)
            for client_idx in client_idxs:
                if client_idx in winners:
                    remaining_winners = [w for w in winners if w != client_idx]
                    if remaining_winners:
                        remaining_weights = [local_weights[remaining_winner] for remaining_winner in remaining_winners]
                        remaining_sizes = [local_sizes[remaining_winner] for remaining_winner in remaining_winners]
                        global_weights_without_winner, _ = size_average_weights(remaining_weights, remaining_sizes)

                        global_model.load_state_dict(global_weights_without_winner)
                        test_acc_without_winner, test_loss_without_winner = inference(args, global_model, test_dataset)
                        if args.aggregation_cost and args.per_model_cal_cost != 0: 
                            aggregation_cost = global_model_size * (args.winners - 1) * args.per_model_cal_cost
                        else:
                            aggregation_cost = 0
                        utility_server_without_winner = test_acc_without_winner * 1000 - aggregation_cost - sum_payments
                        utility_winners_without_winner = sum_payments - np.sum(np.fromiter((client_cost[x] for x in winners if x != client_idx), dtype=float))
                        # utility_winners_without_winner = sum_payments - np.sum(np.fromiter((client_cost[x] for x in winners), dtype=float))
                        social_welfare_without_winner = utility_server_without_winner + utility_winners_without_winner
                        payment[client_idx] = abs(social_welfare - social_welfare_without_winner)
                        total_payment_relative_value += payment[client_idx]
                    else:
                        test_acc_without_winner = 0

                    marginal_contribution[client_idx] = test_acc_with_winner - test_acc_without_winner
                    all_marginal_contribution += marginal_contribution[client_idx]
            
            for client_idx in client_idxs:
                payment[client_idx] = payment[client_idx] / total_payment_relative_value * sum_payments

            for client_idx in client_idxs:
                print(f'payment to client{client_idx}: {payment[client_idx]}, cost is {client_cost[client_idx]}')
          
            
            if args.causal:
                pyro.clear_param_store()
                x_data = torch.tensor(
                            np.column_stack((client_data_sizes, data_cost, np.zeros(args.num_users))),
                            dtype=torch.float32
                        ).to(args.device)
                t_data = torch.tensor(client_cost, dtype=torch.float32).unsqueeze(-1).to(args.device).squeeze(-1)
                m_data = torch.tensor(client_bids, dtype=torch.float32).unsqueeze(-1).to(args.device).squeeze(-1)
                y_data = torch.tensor(marginal_contribution, dtype=torch.float32).unsqueeze(-1).to(args.device).squeeze(-1)

                dmavae = DMA_VAE(
                    feature_dim=x_data.shape[1], 
                    latent_Ztm_dim=1,
                    latent_Zmy_dim=1,
                    hidden_dim=128,
                    num_layers=args.causal_num_layers,
                    num_samples=10,
                    device = args.device,
                    treatment_values = client_cost
                ).to(args.device)
                dmavae.fit(x_data, m_data, t_data, y_data,
                    num_epochs=args.causal_num_epochs,
                    batch_size= args.num_users,
                    learning_rate=args.causal_learning_rate,
                    learning_rate_decay=args.causal_learning_rate_decay, 
                    weight_decay=args.causal_weight_decay)
                
                NDE, NIEr, NIE, ATE = dmavae.effect_estimation(x_data)
                NDE = NDE.cpu().detach().numpy()
                NIEr = NIEr.cpu().detach().numpy()
                NIE = NIE.cpu().detach().numpy()
                ATE = ATE.cpu().detach().numpy()
                NDE_res = np.mean(NDE)
                NIEr_res = np.mean(NIEr)
                NIE_res = np.mean(NIE)
                ATE_res = np.mean(ATE)
                normalized_ATE[round] = (ATE - np.min(ATE)) / (np.max(ATE) - np.min(ATE))
                print(f'NDE: {NDE_res:.4f}, NIEr: {NIEr_res:.4f}, NIE: {NIE_res:.4f}, ATE: {ATE_res:.4f}')
            
            
            print(f'\nAvg Training Stats after {round} global rounds:')
            print(f'Test Loss : {test_loss}')
            print('Test Accuracy: {:.2f}% \n'.format(100 * test_acc))
            print(f'Social Welfare : {social_welfare}')
            
            
    finally:
        del global_model
        torch.cuda.empty_cache()
        print("GPU memory released.")

if __name__ == '__main__':
    args = args_parser()
    exp_details(args)

    torch.cuda.set_device(args.gpu)

    n_list, k_list = 1, 1

    if args.auction:
        train_dataset, test_dataset, user_groups, client_data_sizes, client_data_distributions, global_distribution = get_dataset(args, n_list, k_list)
        Auction_FedAvg_task(args, train_dataset, test_dataset, user_groups, client_data_sizes, client_data_distributions, global_distribution)
    else :
        train_dataset, test_dataset, user_groups, user_groups_lt, classes_list, classes_list_gt = get_dataset(args, n_list, k_list)
