import numpy as np

def generate_comp_cost(num_users, client_data_sizes):
    CPU_CYCLES_PER_SAMPLE = 20 
    CAPACITANCE_COEFFICIENT_RANGE = (1e-28, 2e-28)
    CPU_CYCLE_FREQUENCY_RANGE = (1e9, 2e9)

    capacitance_coefficients = np.random.uniform(*CAPACITANCE_COEFFICIENT_RANGE, num_users) 
    cpu_frequencies = np.random.uniform(*CPU_CYCLE_FREQUENCY_RANGE, num_users) 

    comp_cost = []
    for i in range(num_users):
        client_data_size = client_data_sizes[i]  # d_i
        cost = capacitance_coefficients[i] * CPU_CYCLES_PER_SAMPLE * client_data_size * (cpu_frequencies[i] ** 2)
        comp_cost.append(cost)

    return np.array(comp_cost)

def generate_data_cost(num_users, data_distributions, global_distribution):
    total_global_samples = sum(global_distribution.values())
    normalized_global_distribution = {k: v / total_global_samples for k, v in global_distribution.items()}

    data_cost = []
    for i in range(num_users):
        total_local_samples = sum(data_distributions[i].values())
        normalized_local_distribution = {k: v / total_local_samples for k, v in data_distributions[i].items()}
        discrepancy = 0
        for class_id in normalized_global_distribution.keys():
            global_prob = normalized_global_distribution.get(class_id, 0)
            local_prob = normalized_local_distribution.get(class_id, 0)
            discrepancy += (local_prob - global_prob) ** 2
        discrepancy = discrepancy ** 0.5
        data_cost.append(discrepancy)

    return np.array(data_cost)