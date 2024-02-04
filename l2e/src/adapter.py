import copy

import numpy as np
import torch.optim as optim
from loss_policy import calc_test_policy_loss, get_policy_loss
from models import PokerNet
from play import calc_avg_reward


class Adapter:
    """adapt policy to policy_env"""

    def __init__(self, policy_opponent, n_episode, n_sample):
        """
        policy_opponent: policy to adapt to
        n_episode: number of episodes to train
        n_sample: number of samples to estimate policy loss
        """
        self.policy_opponent = policy_opponent
        self.n_episode = n_episode
        self.n_sample = n_sample

        # save initial state
        self.policy_opponent_initial_state = copy.deepcopy(
            self.policy_opponent.state_dict()
        )

    def adapt(self, policy_b_trained, optimizer_type, alpha, verbose=False):
        self.reset()

        if optimizer_type == "adam":
            optimizer = optim.Adam(policy_b_trained.parameters(), lr=alpha)
        elif optimizer_type == "SGD":
            optimizer = optim.SGD(policy_b_trained.parameters(), lr=alpha)
        else:
            raise NotImplementedError

        results = {}
        for episode in range(self.n_episode):
            policy_loss = get_policy_loss(
                policy_b_trained, self.policy_opponent, n_sample=self.n_sample
            )
            optimizer.zero_grad()
            policy_loss.backward()
            optimizer.step()

            # log
            if episode % (self.n_episode // 10) == 0:
                test_loss = calc_test_policy_loss(
                    policy_b_trained,
                    self.policy_opponent,
                    n_sample=self.n_sample,
                )
                test_avg_reward = calc_avg_reward(
                    policy_b_trained, self.policy_opponent
                )
                results[episode] = {
                    "test_loss": test_loss,
                    "test_avg_reward": test_avg_reward,
                }
                if verbose:
                    print(f"{episode=:4}, {test_loss=:.4f}, {test_avg_reward=:.4f}")

        return results

    def reset(self):
        self.policy_opponent.load_state_dict(self.policy_opponent_initial_state)


class AdapterToRandom(Adapter):
    """adapt policy_b_trained to random policy"""

    def __init__(self, n_episode, n_sample):
        policy_o_random = PokerNet()
        policy_o_random.eval()
        super().__init__(policy_o_random, n_episode, n_sample)
