import torch
from envs import ACTION_SPACE, PokerEnv
from play import play_one_game


def get_policy_mmd_loss(policy_j, policy_i, policy_env, n_sample, alpha_mmd=0.8):
    """n_sample回ゲームをランダムにプレイして、その平均のpolicy_lossを返す
    i, jはL2Eの論文と同じ: jが学習中のpolicy, iが参照policy
    """
    if type(policy_i) is list:
        policy_i_list = policy_i

        mmd_min = float("Inf")  # np.inf
        for p in policy_i_list:
            mmd_i_j = mmd2(policy_j, p, policy_env, n_samples=n_sample)
            if mmd_i_j < mmd_min:
                mmd_min = mmd_i_j
                policy_i = p

    loss_list = []
    for _ in range(10):
        env = PokerEnv(policy_env)
        # traj_list = [(tau, log_probs, reward), ...]
        traj_j_list = [play_one_game(policy_j, env=env) for _ in range(n_sample)]
        traj_i_list = [play_one_game(policy_i, env=env) for _ in range(n_sample)]

        for tau_1, log_probs_1, reward_1 in traj_j_list:
            term_2_list, term_3_list = [], []
            for tau_2, _, _ in traj_i_list:
                term_2_list.append(k(tau_1, tau_2))
            for tau_2, _, _ in traj_j_list:
                term_3_list.append(k(tau_1, tau_2))
            term_2 = torch.stack(term_2_list).mean()
            term_3 = torch.stack(term_3_list).mean()
            reward_mmd = alpha_mmd * (-2 * term_2 + term_3)

            loss = -(torch.stack(log_probs_1) * (reward_1 + reward_mmd)).mean()
            loss_list.append(loss)

    return torch.stack(loss_list).mean()


def mmd2(policy_j, policy_i, policy_env, n_sample):
    """n_sample回ゲームをランダムにプレイして、その平均のpolicy_lossを返す
    i, jはL2Eの論文と同じ: jが学習中のpolicy, iが参照policy
    """
    mmd2_list = []

    for _ in range(10):
        env = PokerEnv(policy_env)
        # traj_list = [(tau, log_probs, reward), ...]
        tau_j_list = [play_one_game(policy_j, env=env)[0] for _ in range(n_sample)]
        tau_i_list = [play_one_game(policy_i, env=env)[0] for _ in range(n_sample)]

        term_1 = torch.stack(
            [k(tau_1, tau_2) for tau_1 in tau_i_list for tau_2 in tau_i_list]
        ).mean()
        term_2 = torch.stack(
            [k(tau_1, tau_2) for tau_1 in tau_j_list for tau_2 in tau_i_list]
        ).mean()
        term_3 = torch.stack(
            [k(tau_1, tau_2) for tau_1 in tau_j_list for tau_2 in tau_j_list]
        ).mean()

        mmd2 = term_1 - 2 * term_2 + term_3

        mmd2_list.append(mmd2)

    return torch.stack(mmd2_list).mean()


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
