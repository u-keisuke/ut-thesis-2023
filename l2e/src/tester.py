import copy

import numpy as np
import torch.optim as optim
from models import PokerNet
from play import get_policy_loss


class tester_random:
    """adapt policy_b_trained to random policy"""

    def __init__(self):
        self.policy_o_random = PokerNet().eval()
        self.policy_o_random.eval()
        # save initial state
        self.policy_o_random_initial_state = copy.deepcopy(
            self.policy_o_random.state_dict()
        )

    def test(self, policy_b_trained, alpha=0.01):
        self.reset()
        self.optimizer = optim.Adam(policy_b_trained.parameters(), lr=alpha)

        list_loss = []
        for episode in range(10000):
            policy_loss = get_policy_loss(
                policy_b_trained, self.policy_o_random, n_sample=10
            )
            self.optimizer.zero_grad()
            policy_loss.backward()
            self.optimizer.step()

            # log
            list_loss.append(policy_loss.item())
            if episode % 1000 == 0:
                print(f"{episode=}, {np.mean(list_loss[-1000:])=}")

    def reset(self):
        self.policy_o_random.load_state_dict(self.policy_o_random_initial_state)
