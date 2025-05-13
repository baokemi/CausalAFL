import numpy as np
import copy
VALUE_RANGE = (0, 1)

def generate_values(N, random_type='uniform'):
    if random_type == 'uniform':
        return np.round(np.random.random(N) * (VALUE_RANGE[1] - VALUE_RANGE[0]), 3)
    else:
        raise NotImplementedError
    
def generate_truthful_bids(buyer_client_values):
    return copy.deepcopy(buyer_client_values)


def emulate_untruthful(buyer_client_bids, untruthful_client, granularity=100):
    untruthful_cases = []
    manipulates = []
    for i in range(granularity):
        case = copy.deepcopy(buyer_client_bids)
        untruthful_bid = np.round(
            (VALUE_RANGE[1] - VALUE_RANGE[0]) / granularity * (i + 1), 3)
        case[untruthful_client - 1] = untruthful_bid
        untruthful_cases.append(case)
        manipulates.append(untruthful_bid)
    return untruthful_cases, manipulates
