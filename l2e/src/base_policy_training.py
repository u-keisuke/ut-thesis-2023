import copy

import torch
import torch.optim as optim
from play import get_policy_loss


def train_base_policy(policy_b, policy_o_list, alpha, beta, n_sample=20):
    policy_b.train()

    loss_oi_boi_list = []
    for policy_oi in policy_o_list:
        policy_oi.eval()

        # 元のpolicy_bの状態を保存
        policy_b_state_dict = copy.deepcopy(policy_b.state_dict())

        # eq.(3)
        optimizer_alpha = optim.Adam(policy_b.parameters(), lr=alpha)
        optimizer_alpha.zero_grad()
        loss_oi_b = get_policy_loss(policy_b, policy_oi, n_sample=n_sample)  # eq.(4)
        loss_oi_b.backward()
        optimizer_alpha.step()

        # policy_bを1エピソードだけ学習させたpolicy_b_oiで，policy_oiに対してlossを計算
        # eq.(6)の中身
        loss_oi_boi = get_policy_loss(policy_b, policy_oi, n_sample=n_sample)
        loss_oi_boi_list.append(loss_oi_boi)

        # policy_bを元の状態に戻す
        policy_b.load_state_dict(policy_b_state_dict)

    # eq.(6)
    loss = torch.stack(loss_oi_boi_list).sum()
    optimizer_beta = optim.Adam(policy_b.parameters(), lr=beta)
    optimizer_beta.zero_grad()
    loss.backward()
    optimizer_beta.step()

    return policy_b
