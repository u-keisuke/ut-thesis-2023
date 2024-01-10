import torch.nn.functional as F
from envs import PokerEnv
from utils_l2e import select_valid_action


def play_one_game(policy, policy_env=None, env=None):
    """1ゲームをプレイして、最終的なinfosetを返す.
    envが与えられた場合,カードなどは変えずにそのまま使う.
    """
    assert policy_env is not None or env is not None
    if env is None:
        env = PokerEnv(policy_env)

    env.start()
    state = env.output_state()
    finished = False

    log_prob_list = []
    while not finished:
        out = policy(state)
        action_idx = select_valid_action(
            out, env._current_node.children.keys(), env.action_space
        )  # 出力確率に基づいてactionを選択

        log_probs = F.log_softmax(out, dim=1)
        log_prob_list.append(log_probs.squeeze(0)[action_idx])

        state, reward, finished = env.step(action_idx)  # terminal以外はreward=0
    final_reward = reward
    trajectory = env._info_set

    return trajectory, log_prob_list, final_reward
