import numpy as np
import torch.optim as optim
from loss_mmd import get_policy_mmd_loss
from models import PokerNet
from tqdm import tqdm


def diverse_osg(
    policy_b,
    policy_o_1,
    n_opponents,
    n_steps,
    n_sample,
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
            loss = get_policy_mmd_loss(
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
            list_loss.append(loss.item())
            # Print running average (学習出来ているかのチェック)
            if t % (n_steps // 10) == 0:
                print(f"{t=}, {np.mean(list_loss[-(n_steps//10):])=}")

        policy_o_list.append(policy_o_i)

    return policy_o_list
