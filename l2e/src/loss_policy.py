import numpy as np
import torch
from play import play_one_game


def get_policy_loss(policy, policy_env, n_sample):
    """n_sample回ゲームをランダムにプレイして、その平均のpolicy_lossを返す"""
    policy_loss_list = []
    for _ in range(n_sample):
        _, log_prob_list, reward = play_one_game(policy, policy_env=policy_env)
        policy_loss = -(torch.stack(log_prob_list) * reward).mean()
        policy_loss_list.append(policy_loss)

    E_policy_loss = torch.stack(policy_loss_list).mean()

    return E_policy_loss


def calc_test_policy_loss(policy, policy_env, n_sample):
    # test loss
    list_loss = []
    for _ in range(1000):
        policy_loss_test = get_policy_loss(policy, policy_env, n_sample=n_sample)
        list_loss.append(policy_loss_test.item())
    return np.mean(list_loss)
