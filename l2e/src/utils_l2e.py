import numpy as np
import torch.nn.functional as F


def select_valid_action(out, valid_actions_str, action_space):
    valid_actions_idx = [action_space.index(a) for a in valid_actions_str]
    out_valid = out[:, valid_actions_idx]
    actions_prob = F.softmax(out_valid, dim=1).detach().numpy()
    selected_action = np.random.choice(valid_actions_idx, p=actions_prob.reshape(-1))
    return selected_action
