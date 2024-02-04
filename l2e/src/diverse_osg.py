import copy

import numpy as np
import torch.optim as optim
from loss_mmd import get_policy_mmd_loss
from loss_policy import calc_test_policy_loss
from models import PokerNet
from play import calc_avg_reward


def diverse_osg(
    policy_b,
    policy_o_1,
    n_opponents,
    n_step,
    n_sample,
    alpha,
    alpha_mmd,
    policy_o_base=None,
    verbose=False,
):
    """Algorithm 3: Diverse-OSG"""
    policy_o_list = [policy_o_1]
    results_list = [{0: {"test_loss": 0, "test_avg_reward": 0}}]

    for i in range(2, n_opponents + 1):
        print(f"{i=} / {n_opponents=}")

        if policy_o_base is None:
            policy_o_i = PokerNet()
        else:
            policy_o_i = copy.deepcopy(policy_o_base)
        optimizer_o_i = optim.Adam(policy_o_i.parameters(), lr=alpha)

        results = {}
        for t in range(n_step):
            loss, mmd_min = get_policy_mmd_loss(
                policy_o_i,
                policy_o_list,
                policy_env=policy_b,
                n_sample=n_sample,
                alpha_mmd=alpha_mmd,
            )
            optimizer_o_i.zero_grad()
            loss.backward()
            optimizer_o_i.step()

            # log
            if t % (n_step // 10) == 0:
                test_policy_loss = calc_test_policy_loss(
                    policy_o_i,
                    policy_b,
                    n_sample=n_sample,
                )
                test_avg_reward = calc_avg_reward(policy_o_i, policy_b)
                results[t] = {
                    "test_policy_loss": test_policy_loss,
                    "test_avg_reward": test_avg_reward,
                }
                if verbose:
                    print(
                        f"{t=:4}, {test_policy_loss=:.4f}, {test_avg_reward=:.4f}, {mmd_min=:.4f}"
                    )

        policy_o_list.append(policy_o_i)
        results_list.append(results)

    return policy_o_list, results_list
