import numpy as np
import torch
import torch.optim as optim
from envs import ACTION_SPACE, PokerEnv
from models import PokerNet
from play import get_policy_loss, play_one_game
from tqdm import tqdm


def g(tau: tuple, L: int = 20):
    """eq.(12) function g
    e.g.)
    input: ((5,), ('raise', 'call', '2', 'call', 'raise', 'call'))
    output: np.array([1, 0, 1, ..., 0, 1, 0])
    """
    vec = torch.zeros(L, dtype=torch.float32)
    hand_jqk = tau[0][0] // 2  # (J1,J2,Q1,...,K2)を(J,Q,K)に変換
    actions = tau[1]  # call, raise, fold, 0~5 (chance nodeのみ)

    vec[hand_jqk] = 1.0
    for i, a in enumerate(actions):
        if a in ACTION_SPACE:  # player node
            idx = 3 + i * 3 + ACTION_SPACE.index(a)
        elif a in [str(h) for h in range(6)]:  # chance node
            a_jqk = int(a) // 2
            idx = 3 + i * 3 + a_jqk

        if idx >= L:
            break
        vec[idx] = 1.0

    return vec


def k(tau_1, tau_2, h=1):
    """eq.(12)"""
    numerator = torch.sum((g(tau_1) - g(tau_2)) ** 2)
    return torch.exp(-numerator / (2 * h))


def mmd(policy_1, policy_2, policy_env, n_samples=8):
    """eq.(11)"""

    E_1_list, E_2_list, E_3_list = [], [], []
    for _ in range(n_samples):
        env = PokerEnv(policy_env)
        tau_1, tau_2 = (
            play_one_game(policy_1, env=env)[0],
            play_one_game(policy_1, env=env)[0],
        )
        E_1_list.append(k(tau_1, tau_2))

        tau_1, tau_2 = (
            play_one_game(policy_1, env=env)[0],
            play_one_game(policy_2, env=env)[0],
        )
        E_2_list.append(k(tau_1, tau_2))

        tau_1, tau_2 = (
            play_one_game(policy_2, env=env)[0],
            play_one_game(policy_2, env=env)[0],
        )
        E_3_list.append(k(tau_1, tau_2))

    E_1 = torch.stack(E_1_list).mean()
    E_2 = torch.stack(E_2_list).mean()
    E_3 = torch.stack(E_3_list).mean()

    return E_1 + E_3 - 2 * E_2


def get_mmd_loss(policy, policy_ref_list, policy_env, n_samples=8):
    """eq.(14) term 1"""
    min_mmd = float("Inf")  # np.inf
    for policy_ref in policy_ref_list:
        mmd_loss = mmd(policy, policy_ref, policy_env, n_samples=n_samples)
        if mmd_loss < min_mmd:
            min_mmd = mmd_loss
    return -min_mmd


def diverse_osg(policy_b, policy_o_1, n_opponents, n_steps, n_sample, alpha, alpha_mmd):
    """Algorithm 3: Diverse-OSG"""
    policy_o_list = [policy_o_1]
    for i in range(n_opponents):
        policy_o_i = PokerNet()
        optimizer_o_i = optim.Adam(policy_o_i.parameters(), lr=alpha)

        list_loss = []
        for t in tqdm(range(n_steps)):
            policy_loss = get_policy_loss(
                policy_o_i, policy_env=policy_b, n_sample=n_sample
            )
            mmd_loss = get_mmd_loss(
                policy_o_i, policy_o_list, policy_env=policy_b, n_samples=n_sample
            )
            loss = policy_loss + alpha_mmd * mmd_loss

            optimizer_o_i.zero_grad()
            loss.backward()
            optimizer_o_i.step()

            # log
            list_loss.append(policy_loss.item())
            # Print running average (学習出来ているかのチェック)
            if t % (n_steps // 10) == 0:
                print(f"{t=}, {np.mean(list_loss[-1000:])=}")

        policy_o_list.append(policy_o_i)

    return policy_o_list
