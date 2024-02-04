import numpy as np
import torch.nn.functional as F
from envs import PokerEnv
from tqdm import tqdm
from utils_l2e import select_valid_action


def play_one_game(policy, policy_env=None, env=None, if_debug=False):
    """1ゲームをプレイして、最終的なinfosetを返す.
    envが与えられた場合,カードなどは変えずにそのまま使う.
    """
    assert policy_env is not None or env is not None
    if env is None:
        env = PokerEnv(policy_env)

    env.start()
    state_vec = env.transform_infoset_to_vec(env.info_set)
    finished = False

    log_prob_list = []
    while not finished:
        state_vec = env.transform_infoset_to_vec(env.info_set)
        out = policy(state_vec)
        action_idx = select_valid_action(
            out, env._current_node.children.keys(), env.action_space
        )  # 出力確率に基づいてactionを選択

        log_probs = F.log_softmax(out, dim=1)
        log_prob_list.append(log_probs.squeeze(0)[action_idx])

        reward, finished = env.step(action_idx)  # terminal以外はreward=0

    final_reward = reward
    trajectory = env.info_set

    if if_debug:
        return trajectory, log_prob_list, final_reward, env
    else:
        return trajectory, log_prob_list, final_reward


def calc_avg_reward(policy, policy_env):
    list_reward = []
    for _ in tqdm(range(1000)):
        _, _, reward = play_one_game(policy, policy_env)
        list_reward.append(reward.item())
    return np.mean(list_reward)
