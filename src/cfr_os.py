import numpy as np

from poker import Node

epsilon = 0.1


def update_node_values_outcome_sampling(node: Node, strategy_profile: dict, s: float):
    node.num_updates += 1

    if node.terminal:
        return node.eu / s, 1

    if node.player == -1:  # chance player
        # sample child note
        child_node = np.random.choice(list(node.children.values()))
        return update_node_values_outcome_sampling(child_node, strategy_profile, s)

    node.pi_i_sum += node.pi_i

    strategy_node = strategy_profile[node.player][node.information]

    # sample action (from epsion*uniform + (1-epsilon)*strategy_profile)
    sampling_distribution = np.array(
        [epsilon / len(node.children) for _ in range(len(node.children))]
    )
    for i, (action, child_node) in enumerate(node.children.items()):
        sampling_distribution[i] += (1 - epsilon) * strategy_node[action]

    i_sampled = np.random.choice(
        list(range(len(node.children))),
        p=sampling_distribution,
    )
    action_sampled = list(node.children.keys())[i_sampled]
    child_node_sampled = node.children[action_sampled]

    p = strategy_profile[node.player][node.information][action_sampled]
    node.pi_sigma_sum[action_sampled] += node.pi_i * p

    node_eu, pi_tail = update_node_values_outcome_sampling(
        child_node_sampled,
        strategy_profile,
        s * sampling_distribution[i_sampled],
    )
    node.eu = node_eu

    node.cv = node.pi_mi * node_eu
    for action, child_node in node.children.items():
        W = child_node.eu * node.pi_mi
        if action == action_sampled:
            cfr_sampled = W * ((pi_tail / child_node.pi) - (pi_tail / node.pi))
        else:
            cfr_sampled = (-1) * W * (pi_tail / node.pi)

        if node.player == 0:
            cfr_sampled = cfr_sampled
        else:
            cfr_sampled = (-1) * cfr_sampled

        node.cfr[action] += cfr_sampled

    return node_eu, pi_tail * p
