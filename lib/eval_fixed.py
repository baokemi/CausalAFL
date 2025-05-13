import copy
import numpy as np

def eval_fixed_k_profit(buyer_client_values, buyer_client_bids, performances, buyer_client_protos, K):
    profit = np.zeros(len(buyer_client_values) + 1)
    seller_profit = 0
    sorted_clients = np.argsort(performances * buyer_client_bids) 
    winners = sorted_clients[-K:] + 1  
    threshold_client = sorted_clients[-(K + 1)] if len(sorted_clients) > K else sorted_clients[0]
    payment = buyer_client_bids[threshold_client] * performances[threshold_client] 
    gains = buyer_client_values[winners - 1] * performances[winners - 1]
    profit[winners] = gains - payment
    sellers_profit = payment * len(winners)
    social_welfare = np.sum(gains)
    return profit, sellers_profit, social_welfare, winners


def sellers_eval_fixed_k_profit(buyer_client_values, buyer_client_bids, buyer_performances, buyer_client_protos, K, num_seller):
    profit = np.zeros(len(buyer_client_values) + num_seller)
    sellers_profit = 0

    sorted_clients = np.argsort(buyer_performances * buyer_client_bids)
    winners_buyer = sorted_clients[-K:] 
    threshold_client = sorted_clients[-(K + 1)] if len(sorted_clients) > K else sorted_clients[0]

    payments = np.zeros(K)
    for i in range(K):
        if i < K - 1:
            next_client = sorted_clients[-(i + 2)]  
            payments[i] = buyer_client_bids[next_client] * buyer_performances[next_client]
        else:
            threshold_client = sorted_clients[-(K + 1)]
            payments[i] = buyer_client_bids[threshold_client] * buyer_performances[threshold_client]     
    gains = buyer_client_values[winners_buyer] * buyer_performances[winners_buyer]
    winners = winners_buyer + num_seller
    profit[winners] = gains - payments
    

    sellers_profit = np.sum(payments)
    
    social_welfare = np.sum(gains)
    return profit, sellers_profit, social_welfare, winners, sorted_clients


def eval_fixed_k_profit_changing_ranking(buyer_client_values, buyer_client_bids, target_ranking, performances, K, untruthful_client_id, num_seller):
    untruthful_buyer_sorted_clients = np.argsort(performances * buyer_client_bids)
    untruthful_buyer_ids = untruthful_client_id - num_seller  

    buyer_sorted_clients = [i for i in untruthful_buyer_sorted_clients if i != untruthful_buyer_ids] 

    winners = []
    profit = np.zeros(len(buyer_client_bids) + num_seller)
    if target_ranking <= K:
        target_index = len(buyer_sorted_clients) - target_ranking + 1
        buyer_sorted_clients.insert(target_index, untruthful_buyer_ids)
    
    winners = np.array(buyer_sorted_clients[-K:])

    payments = np.zeros(K)
    for i in range(K):
        if i < K - 1:
            next_client = buyer_sorted_clients[-(i + 2)] 
            payments[i] = buyer_client_bids[next_client] * performances[next_client]
        else:
            threshold_client = buyer_sorted_clients[-(K + 1)]
            payments[i] = buyer_client_bids[threshold_client] * performances[threshold_client]

    gains = buyer_client_values[winners] * performances[winners] 

    winners = winners + num_seller
    profit[winners] = gains - payments

    sellers_profit = np.sum(payments)
    social_welfare = np.sum(gains)
    return profit, sellers_profit, social_welfare, winners, untruthful_buyer_sorted_clients


def eval_fixed_k_profit_changing_performance(buyer_client_values, buyer_client_bids, performances, K, untruthful_client_id, sensitivity):
    hacked_perfs = copy.deepcopy(performances) 
    hacked_perfs[untruthful_client_id-1] = performances[untruthful_client_id-1] * sensitivity
    hacked_bids = copy.deepcopy(buyer_client_bids)
    sorted_clients = np.argsort(hacked_perfs * hacked_bids)
    winners = sorted_clients[-K:] + 1
    threshold_client = sorted_clients[-(K + 1)] if len(sorted_clients) > K else sorted_clients[0]

    payments = np.zeros(K)
    for i in range(K):
        if i < K - 1:
            next_client = sorted_clients[-(i + 2)] 
            payments[i] = buyer_client_bids[next_client] * performances[next_client]
        else:
            threshold_client = sorted_clients[-(K + 1)]
            payments[i] = buyer_client_bids[threshold_client] * performances[threshold_client]

    gains = buyer_client_values[winners - 1] * hacked_perfs[winners - 1]
    profit = np.zeros(len(buyer_client_values))
    profit[winners - 1] = gains - payments
    sellers_profit = np.sum(payments)
    social_welfare = np.sum(gains) 
    return profit, sellers_profit, social_welfare, winners, sorted_clients + 1


def eval_fixed_k_profit_changing_score(buyer_client_values, buyer_client_bids, performances, K, untruthful_client_id, sensitivity):
    hacked_perfs = copy.deepcopy(performances)
    hacked_bids = copy.deepcopy(buyer_client_bids)
    
    hacked_perfs[untruthful_client_id-1] = performances[untruthful_client_id-1] * sensitivity 
    hacked_bids[untruthful_client_id-1] = 1.0 
    

    sorted_clients = np.argsort(hacked_perfs * hacked_bids)
    winners = sorted_clients[-K:] + 1
    threshold_client = sorted_clients[-(K + 1)] if len(sorted_clients) > K else sorted_clients[0]

    payments = np.zeros(K)
    for i in range(K):
        if i < K - 1:
            next_client = sorted_clients[-(i + 2)] 
            payments[i] = buyer_client_bids[next_client] * performances[next_client]
        else:
            threshold_client = sorted_clients[-(K + 1)]
            payments[i] = buyer_client_bids[threshold_client] * performances[threshold_client]

    
    gains = buyer_client_values[winners - 1] * hacked_perfs[winners - 1]
    profit = np.zeros(len(buyer_client_values))
    profit[winners - 1] = gains - payments
    sellers_profit = np.sum(payments)
    social_welfare = np.sum(gains)
    
    return profit, sellers_profit, social_welfare, winners, sorted_clients + 1


def reverse_auction_eval_fixed_k_profit(client_bids, client_performances, K):

    sorted_clients = np.argsort(-(client_bids * client_performances))
    winners_client = sorted_clients[:K]
    threshold_client = sorted_clients[-(K + 1)] if len(sorted_clients) > K else sorted_clients[0]

    payments = np.zeros(K)
    for i in range(K):
        if i < K - 1:
            next_client = sorted_clients[i + 1] 
            payments[i] = client_bids[next_client] 
        else:
            threshold_client = sorted_clients[K]
            payments[i] = client_bids[threshold_client] 
    
    return payments, winners_client, sorted_clients


def reverse_auction_bid_eval_fixed_k_profit(client_bids, K):
    sorted_clients = np.argsort(client_bids) 
    winners_client = sorted_clients[:K]
    threshold_client = sorted_clients[-(K + 1)] if len(sorted_clients) > K else sorted_clients[0]
    
    return winners_client, sorted_clients