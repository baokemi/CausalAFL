
# CausalAFL Project Guide

Welcome to the CausalAFL project's pretraining guide. 
CausalAFL simulates a market-driven competition, where the server acts as the buyer, and the clients act as sellers, with the server selecting the winning clients in each communication round according to their submitted bids.
CausalAFL systematically models the causal relationships among client information, bid, and marginal contribution. By introducing a generative network to estimate mediation pathways and adjust client bidding strategies, CausalAFL improves budget-constrained social welfare and provides rational bids, ultimately achieving optimal model performance.


## 1. Auction command

```
python main_causal.py --dataset cifar10 --num_classes 10 --lr 0.1 --num_users 20 --winners 10 --rounds 200 --train_ep 5 --dir_alpha 0.01 --comp_cost --data_cost --data_cost_L2 --aggregation_cost --per_model_cal_cost 0.0001 --noise_mean 1 --noise_std_dev 0.5 --device cuda --gpu 0 --causal --local_bs 512
```
