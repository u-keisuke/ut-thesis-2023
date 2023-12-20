import itertools
import random
import time

import numpy as np

from game import KuhnPoker

game = KuhnPoker()

START_TIME = time.time()


def normalize(vector):
    vector_plus = np.maximum(0, vector)
    total = np.sum(vector_plus)
    return vector_plus / total if total > 0 else np.ones(len(vector)) / len(vector)


def make_strategy(regret):
    """make probability distribution proportional regret+""",
    return normalize(regret)


def CFR(h, i, t, pi0, pi1, table):
    """
    Algorithm 1 in the tutorial
    - h: history
    - i: [0,1] player to learn
    - t: iteration number
    - pi0, pi1: reach probability for both players
    - table: map from history to info_set
    """
    if game.is_terminal(h):
        return game.utility(h, i)
    pl = game.turn(h)

    infoset = game.make_info_set(h, pl)
    if infoset not in table:
        table[infoset] = {
            "regret_sum": np.zeros(game.N_ACTIONS),
            "strategy_sum": np.zeros(game.N_ACTIONS),
            "regret_update": np.zeros(game.N_ACTIONS),
            "visit": 0,
        }
    node = table[infoset]
    node["visit"] += 1

    # regret matching
    strategy = make_strategy(node["regret_sum"])

    action_value = np.zeros(game.N_ACTIONS)  # line 14
    for a in range(game.N_ACTIONS):
        pp = [1, 1]
        pp[pl] = strategy[a]
        action_value[a] = CFR(
            h + game.ACTION_CHAR[a], i, t, pi0 * pp[0], pi1 * pp[1], table
        )
    node_value = np.dot(strategy, action_value)  # line 21

    # update
    if pl == i:
        pi, p_i = (pi0, pi1) if i == 0 else (pi1, pi0)
        node["regret_update"] += p_i * (action_value - node_value)
        node["strategy_sum"] += pi * strategy

    return node_value


def CFR_fixed_opponent(h, i, t, pi0, pi1, table_0, table_1):
    """
    Algorithm 1 in the tutorial
    - h: history
    - i: [0,1] player to learn
    - t: iteration number
    - pi0, pi1: reach probability for both players
    - table: map from history to info_set
    """
    if game.is_terminal(h):
        return game.utility(h, i)
    pl = game.turn(h)

    table = table_0 if pl == 0 else table_1

    infoset = game.make_info_set(h, pl)
    if infoset not in table:
        table[infoset] = {
            "regret_sum": np.zeros(game.N_ACTIONS),
            "strategy_sum": np.zeros(game.N_ACTIONS),
            "regret_update": np.zeros(game.N_ACTIONS),
            "visit": 0,
        }
    node = table[infoset]
    node["visit"] += 1

    # regret matching
    strategy = make_strategy(node["regret_sum"])

    action_value = np.zeros(game.N_ACTIONS)  # line 14
    for a in range(game.N_ACTIONS):
        pp = [1, 1]
        pp[pl] = strategy[a]
        action_value[a] = CFR_fixed_opponent(
            h + game.ACTION_CHAR[a], i, t, pi0 * pp[0], pi1 * pp[1], table, table_1
        )
    node_value = np.dot(strategy, action_value)  # line 21

    # update
    if pl == i:
        pi, p_i = (pi0, pi1) if i == 0 else (pi1, pi0)
        node["regret_update"] += p_i * (action_value - node_value)
        node["strategy_sum"] += pi * strategy

    return node_value


def solve_cfr(T):
    table = {}
    for t in range(T):
        for i in range(2):
            for hands in itertools.permutations("012", 2):
                h = "".join(hands)
                value = CFR(h, i, t, 1 / 6, 1 / 6, table)
            for info_set, node in table.items():
                node["regret_sum"] += node["regret_update"]
                node["regret_update"].fill(0)

        # log_schedule = lambda x: (10 ** (len(str(x)) - 1))
        log_schedule = lambda x: 1
        if t % log_schedule(t) == 0:
            _, br0_eval1 = solve_cfr_fixed_opponent(10000, 0, table.copy())
            _, br1_eval0 = solve_cfr_fixed_opponent(10000, 1, table.copy())
            log(
                f"{t}, {br0_eval1}, {br1_eval0}, {br0_eval1 + br1_eval0}",
                "vanilla_cfr_br.csv",
            )

    return table


def solve_cfr_fixed_opponent(T, i, learned_table):
    table = {}
    for t in range(T):
        value_sum_player_0 = 0
        for hands in itertools.permutations("012", 2):
            h = "".join(hands)
            if i == 0:
                value = CFR_fixed_opponent(h, i, t, 1 / 6, 1 / 6, table, learned_table)
            else:
                value = -CFR_fixed_opponent(h, i, t, 1 / 6, 1 / 6, learned_table, table)
            value_sum_player_0 += value * 1 / 6
        for info_set, node in table.items():
            node["regret_sum"] += node["regret_update"]
            node["regret_update"].fill(0)

    return table, value_sum_player_0


def solve_chance_sampling_mccfr(T):
    table = {}
    for t in range(T):
        for i in range(2):
            hands = random.choice(list(itertools.permutations("012", 2)))
            h = "".join(hands)
            value = CFR(h, i, t, 1 / 6, 1 / 6, table)
            for info_set, node in table.items():
                node["regret_sum"] += node["regret_update"]
                node["regret_update"].fill(0)

        log_schedule = lambda x: (10 ** (len(str(x)) - 1))
        if t % log_schedule(t) == 0:
            try:
                exploitability = 0
                for i in range(2):
                    for hands in itertools.permutations("012", 2):
                        exploitability += compute_exploitability(
                            "".join(hands), i, table
                        )
            except:
                exploitability = 0
            log(f"{t}, {exploitability}", "mccfr_chance_sampling.csv")

    return table


def calculate_p_i(h, i, table):
    """iにとってのp_iを計算する"""
    p_i = 1
    for j in range(len(h)):
        # chance node
        if j == 0 or j == 1:
            p_i *= 1 / 6
        # player node
        elif j % 2 != i:
            # 部分history
            infoset = game.make_info_set(h[:j], 1 - i)
            strategy = make_strategy(table[infoset]["regret_sum"])
            p_i *= strategy[game.ACTION_CHAR.index(h[j])]
        else:
            pass
    return p_i


def possible_history(infoset):
    """infosetにおける可能なhistoryを列挙する"""
    if infoset[0] == "?":
        return [str(c) + infoset[1:] for c in [0, 1, 2] if str(c) != infoset[1]]
    elif infoset[1] == "?":
        return [
            infoset[0] + str(c) + infoset[2:] for c in [0, 1, 2] if str(c) != infoset[0]
        ]
    else:
        raise ValueError


def compute_exploitability(h, i, table):
    """
    exploitability when opponent player chooses best response for current strategy
    h: histoty
    i: player to exploit
    table: map from history to info_set
    """
    best_reponse_strategy, best_reponse_value = solve_cfr_fixed_opponent(
        1000, i, table.copy()
    )

    print(h, game.make_info_set(h, i))
    if game.is_terminal(h):
        print("terminal", game.utility(h, i))
        return game.utility(h, i)

    pl = game.turn(h)
    infoset = game.make_info_set(h, pl)

    if pl == i:
        best_response_value = None
        best_weighted_expected_utility = -float("inf")
        for a in range(game.N_ACTIONS):
            current_weighted_expected_utility = 0
            current_node_expected_utility = 0

            for h_ in possible_history(infoset):
                print(h, "pb"[a], "possible history", h_)
                expected_utility = compute_exploitability(
                    h_ + game.ACTION_CHAR[a], i, table
                )
                if h == h_:
                    current_node_expected_utility = expected_utility
                current_weighted_expected_utility += expected_utility * calculate_p_i(
                    h_, i + 1, table
                )
                print(h, "pb"[a], "expected utility", expected_utility)

            if best_weighted_expected_utility < current_weighted_expected_utility:
                best_weighted_expected_utility = current_weighted_expected_utility
                best_response_value = current_node_expected_utility
        print("best response value", best_response_value)
        return best_response_value

    else:
        expected_utility = 0
        node = table[infoset]
        # strategy = make_strategy(node["regret_sum"])
        strategy = normalize(node["strategy_sum"])
        for a in range(game.N_ACTIONS):
            s = compute_exploitability(h + game.ACTION_CHAR[a], i, table)
            expected_utility += strategy[a] * s
        print("expected utility", expected_utility)
        return expected_utility


def log(s, file_name):
    time_passed = time.time() - START_TIME
    with open(file_name, "a") as f:
        f.write(f"{time_passed}, {s}\n")


if __name__ == "__main__":
    T = 100

    table = solve_cfr(T)
    # table = solve_chance_sampling_mccfr(T)

    np.set_printoptions(precision=3, floatmode="fixed", suppress=True)
    for info_set, node in table.items():
        print(f'{info_set:8} {normalize(node["strategy_sum"])}')

    import json

    # convert numpy array to list
    for info_set, node in table.items():
        node["regret_sum"] = node["regret_sum"].tolist()
        node["strategy_sum"] = node["strategy_sum"].tolist()
        node["regret_update"] = node["regret_update"].tolist()

    with open("table.json", "w") as f:
        json.dump(table, f)
