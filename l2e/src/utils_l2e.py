import numpy as np
import torch
import torch.nn.functional as F


def select_valid_action(out, valid_actions_str: list, action_space: list):
    valid_actions_idx = [action_space.index(a) for a in valid_actions_str]

    out_valid = out[:, valid_actions_idx]
    actions_prob = F.softmax(out_valid, dim=1)

    # selected_action = np.random.choice(valid_actions_idx, p=actions_prob.reshape(-1))

    # torch.multinomialを使って、確率に基づいてアクションをサンプリング
    selected_action_idx = torch.multinomial(actions_prob, num_samples=1)

    # 選択されたアクションのインデックスを取得し、元のアクションスペースにマッピング
    selected_action = valid_actions_idx[selected_action_idx.item()]

    return selected_action
