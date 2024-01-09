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
    actions = tau[1]  # call, raise, fold, 0~5 (chance nodeのみ)

    round = 0
    counter = 0
    for a in actions:
        if a in ["0", "1", "2", "3", "4", "5"]:  # chance node
            round += 1
            counter = 0
        elif a in ACTION_SPACE:  # player node
            # round 0の最長は4アクション(call,raise,raise,call)なので12
            idx = round * 12 + counter * 3 + ACTION_SPACE.index(a)
            counter += 1

        if idx >= L:
            break
        vec[idx] = 1.0

    return vec


def k(tau_a, tau_b, h=1):
    """eq.(12)"""
    numerator = torch.sum((g(tau_a) - g(tau_b)) ** 2)
    return torch.exp(-numerator / (2 * h))


def mmd2(policy_1, policy_2, policy_env, n_samples=8):
    """eq.(11)"""
    mmd2_list = []
    for _ in range(10):
        env = PokerEnv(policy_env)
        policy_1_tau_list = [
            play_one_game(policy_1, env=env)[0] for _ in range(n_samples)
        ]
        policy_2_tau_list = [
            play_one_game(policy_2, env=env)[0] for _ in range(n_samples)
        ]

        term_1 = torch.stack(
            [
                k(tau_a, tau_b)
                for tau_a in policy_1_tau_list
                for tau_b in policy_1_tau_list
            ]
        ).mean()
        term_2 = torch.stack(
            [
                k(tau_a, tau_b)
                for tau_a in policy_1_tau_list
                for tau_b in policy_2_tau_list
            ]
        ).mean()
        term_3 = torch.stack(
            [
                k(tau_a, tau_b)
                for tau_a in policy_2_tau_list
                for tau_b in policy_2_tau_list
            ]
        ).mean()

        mmd2 = term_1 - 2 * term_2 + term_3
        mmd2_list.append(mmd2)

    return torch.stack(mmd2_list).mean()


def get_mmd_loss(policy, policy_ref_list, policy_env, n_samples=8):
    """eq.(14) term 1"""
    min_mmd = float("Inf")  # np.inf
    for policy_ref in policy_ref_list:
        mmd_loss = mmd2(policy, policy_ref, policy_env, n_samples=n_samples)
        if mmd_loss < min_mmd:
            min_mmd = mmd_loss
    return -min_mmd


def diverse_osg(
    policy_b,
    policy_o_1,
    n_opponents,
    n_steps,
    n_sample_policy_loss,
    n_sample_mmd_loss,
    alpha,
    alpha_mmd,
):
    """Algorithm 3: Diverse-OSG"""
    policy_o_list = [policy_o_1]
    for i in range(2, n_opponents):
        policy_o_i = PokerNet()
        optimizer_o_i = optim.Adam(policy_o_i.parameters(), lr=alpha)

        list_loss = []
        for t in tqdm(range(n_steps)):
            policy_loss = get_policy_loss(
                policy_o_i, policy_env=policy_b, n_sample=n_sample_policy_loss
            )
            mmd_loss = get_mmd_loss(
                policy_o_i,
                policy_o_list,
                policy_env=policy_b,
                n_samples=n_sample_mmd_loss,
            )
            loss = policy_loss + alpha_mmd * mmd_loss

            optimizer_o_i.zero_grad()
            loss.backward()
            optimizer_o_i.step()

            # log
            list_loss.append(policy_loss.item())
            # Print running average (学習出来ているかのチェック)
            if t % (n_steps // 10) == 0:
                print(f"{t=}, {np.mean(list_loss[-(n_steps//10):])=}")

        policy_o_list.append(policy_o_i)

    return policy_o_list
