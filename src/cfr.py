from copy import copy

import numpy as np

from poker import Node


def update_pi(
    node: Node,
    strategy_profile: dict,
    average_strategy_profile: dict,
    pi_mi_list: list,
    pi_i_list: list,
    true_pi_mi_list: list,
):
    node.pi = pi_mi_list[node.player] * pi_i_list[node.player]
    node.true_pi_mi = true_pi_mi_list[node.player]
    node.pi_mi = pi_mi_list[node.player]
    node.pi_i = pi_i_list[node.player]

    if node.terminal:
        return
    for action, child_node in node.children.items():
        next_pi_mi_list = copy(pi_mi_list)
        next_pi_i_list = copy(pi_i_list)
        next_true_pi_mi_list = copy(true_pi_mi_list)
        for player in strategy_profile.keys():
            if player == node.player:
                next_pi_i_list[player] *= strategy_profile[node.player][
                    node.information
                ][action]
            else:
                next_pi_mi_list[player] *= strategy_profile[node.player][
                    node.information
                ][action]
                next_true_pi_mi_list[player] *= average_strategy_profile[node.player][
                    node.information
                ][action]
        update_pi(
            child_node,
            strategy_profile,
            average_strategy_profile,
            next_pi_mi_list,
            next_pi_i_list,
            next_true_pi_mi_list,
        )


def update_node_values(root_note, strategy_profile, args):
    if args.cs:
        update_node_values_chance_sampling(root_note, strategy_profile)
    elif args.os:
        update_node_values_outcome_sampling(root_note, strategy_profile)
    else:
        update_node_values_vanilla(root_note, strategy_profile)


def update_node_values_vanilla(node: Node, strategy_profile: dict):
    node.num_updates += 1

    if node.terminal:
        return node.eu

    node_eu = 0  # ノードに到達した後得られる期待利得
    node.pi_i_sum += node.pi_i

    for action, child_node in node.children.items():
        p = strategy_profile[node.player][node.information][action]
        node.pi_sigma_sum[action] += node.pi_i * p
        node_eu += p * update_node_values_vanilla(child_node, strategy_profile)
    node.eu = node_eu
    node.cv = node.pi_mi * node_eu

    for action, child_node in node.children.items():
        if node.player == 0:
            cfr = node.pi_mi * child_node.eu - node.cv
        else:
            cfr = (-1) * (node.pi_mi * child_node.eu - node.cv)
        node.cfr[action] += cfr

    return node_eu


def update_node_values_chance_sampling(node: Node, strategy_profile: dict):
    node.num_updates += 1

    if node.terminal:
        return node.eu

    if node.player == -1:  # chance player
        # sample child note
        child_node = np.random.choice(list(node.children.values()))
        return update_node_values_chance_sampling(child_node, strategy_profile)

    node_eu = 0  # ノードに到達した後得られる期待利得
    node.pi_i_sum += node.pi_i

    for action, child_node in node.children.items():
        p = strategy_profile[node.player][node.information][action]
        node.pi_sigma_sum[action] += node.pi_i * p
        node_eu += p * update_node_values_chance_sampling(child_node, strategy_profile)
    node.eu = node_eu
    node.cv = node.pi_mi * node_eu

    for action, child_node in node.children.items():
        if node.player == 0:
            cfr = node.pi_mi * child_node.eu - node.cv
        else:
            cfr = (-1) * (node.pi_mi * child_node.eu - node.cv)
        node.cfr[action] += cfr

    return node_eu


epsilon = 0.1


def update_node_values_outcome_sampling(node: Node, strategy_profile: dict, s: float):
    node.num_updates += 1

    if node.terminal:
        return node.eu / s, 1

    if node.player == -1:  # chance player
        # sample child note
        child_node = np.random.choice(list(node.children.values()))
        return update_node_values_outcome_sampling(child_node, strategy_profile, s)

    node_eu = 0  # ノードに到達した後得られる期待利得
    node.pi_i_sum += node.pi_i

    # sample action (from epsion*uniform + (1-epsilon)*strategy_profile)
    sampling_distribution = np.array(
        [epsilon / len(node.children) for _ in range(len(node.children))]
    )
    for i, (action, child_node) in enumerate(node.children.items()):
        sampling_distribution[i] += (1 - epsilon) * strategy_profile[node.player][
            node.information
        ][action]
    action = np.random.choice(
        list(node.children.keys()),
        p=list(strategy_profile[node.player][node.information].values()),
    )
    child_node = node.children[action]

    p = strategy_profile[node.player][node.information][action]
    node.pi_sigma_sum[action] += node.pi_i * p
    node_eu += p * update_node_values_outcome_sampling(
        child_node,
        strategy_profile,
        s * strategy_profile[node.player][node.information][action],
    )

    node.eu = node_eu
    node.cv = node.pi_mi * node_eu

    for action, child_node in node.children.items():
        if node.player == 0:
            cfr = node.pi_mi * child_node.eu - node.cv
        else:
            cfr = (-1) * (node.pi_mi * child_node.eu - node.cv)
        node.cfr[action] += cfr

    return node_eu


def get_initial_strategy_profile(node: Node, num_players=None, strategy_profile=None):
    if node.terminal:
        return strategy_profile
    if strategy_profile is None:
        strategy_profile = {
            player: {} for player in range(-1, num_players)
        }  # chance nodeのために+1する
    if node.information not in strategy_profile[node.player]:
        strategy_profile[node.player][node.information] = {
            action: 1 / len(node.children) for action in node.children
        }
    for child in node.children.values():
        strategy_profile = get_initial_strategy_profile(
            child, strategy_profile=strategy_profile
        )
    return strategy_profile


def update_strategy(
    strategy_profile: dict, average_strategy_profile: dict, information_sets: dict
):
    for player, information_policy in strategy_profile.items():
        if player == -1:
            continue

        for information, strategy in information_policy.items():
            cfr = {}
            for action in strategy.keys():
                # 同一ノードにおけるcfrの和を計算
                cfr_in_info = 0
                average_strategy_denominator = 0
                average_strategy_numerator = 0
                for same_info_node in information_sets[player][information]:
                    cfr_in_info += same_info_node.cfr[action]
                    average_strategy_denominator += same_info_node.pi_i_sum
                    average_strategy_numerator += same_info_node.pi_sigma_sum[action]
                cfr[action] = max(cfr_in_info, 0)

                if not average_strategy_denominator == 0:  ###
                    average_strategy_profile[player][information][action] = (
                        average_strategy_numerator / average_strategy_denominator
                    )

            cfr_sum = sum([cfr_values for cfr_values in cfr.values()])
            for action in strategy.keys():
                if cfr_sum > 0:
                    strategy[action] = cfr[action] / cfr_sum
                else:
                    strategy[action] = 1 / len(strategy)
    return
