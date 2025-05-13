import numpy as np
from sklearn.linear_model import LinearRegression
from math import comb

def gen_auction_data(client_cost, noise_std=0.1):
    np.random.seed(0)
    N = len(client_cost)
    noise = np.random.normal(0, noise_std, N)
    true_bids = client_cost + 0.5 * client_cost**2 + noise
    return true_bids

def true_theta(C):
    if C[0]:
        mCost, mCostSq = (0.5, 0.25)
    else:
        mCost, mCostSq = (0, 0)
    mBids = mCost + 0.5 * mCostSq if C[1] else mCost
    return mBids

def est_all_thetas(client_cost, client_bids, all_combos):
    res = {}
    for C in all_combos:
        if C[0]: 
            X = client_cost.reshape(-1, 1)
        else: 
            X = np.ones_like(client_cost).reshape(-1, 1)
        reg = LinearRegression().fit(X, client_bids)
        t = reg.predict(X).mean()
        res[''.join(map(lambda x: str(int(x)), C))] = t
    return res


def get_shap_val(K, res_dict, all_combos_minus1):
    shap = np.zeros(K + 1)
    for k in range(K + 1):
        sv = 0.0
        for C in all_combos_minus1:
            C_padded = np.pad(C, (0, max(0, K + 1 - len(C))), constant_values=False)

            C1 = np.insert(C_padded, k, True)
            C0 = np.insert(C_padded, k, False)

            key_C1 = ''.join(map(lambda x: str(int(x)), C1))
            key_C0 = ''.join(map(lambda x: str(int(x)), C0))

            if key_C1 not in res_dict or key_C0 not in res_dict:
                print(f"Warning: Key {key_C1} or {key_C0} not found in res_dict")
                continue

            chg = res_dict[key_C1] - res_dict[key_C0]
            sv += chg / ((K + 1) * comb(K, np.sum(C)))

        shap[k] = sv
    return shap
