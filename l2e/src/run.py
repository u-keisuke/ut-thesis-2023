import copy

import numpy as np
import torch.optim as optim
from base_policy_training import train_base_policy
from diverse_osg import diverse_osg
from envs import get_policy_loss
from hard_osg import hard_osg
from models import PokerNet


def main(epochs=50):
    alpha = 0.1
    beta = 0.01
    policy_b = PokerNet()  # B
    policy_o = PokerNet()  # O
    policy_o_list = [policy_o]

    test(copy.deepcopy(policy_b), alpha=alpha)

    for e in range(epochs):
        print(f"{e=}")

        print("hard_osg")
        policy_o = hard_osg(policy_b, n_epochs=20, n_sample=20, alpha=alpha)

        print("diverse_osg")
        policy_o_list += diverse_osg(
            policy_b,
            policy_o,
            n_opponents=3,
            n_steps=1000,
            n_sample=10,
            alpha=alpha,
            alpha_mmd=0.1,
        )

        print("train_base_policy")
        policy_o_list_sampled = np.random.choice(
            policy_o_list, size=min(40, len(policy_o_list))
        )
        policy_b = train_base_policy(
            policy_b, policy_o_list_sampled, alpha=alpha, beta=beta, n_sample=20
        )

        print("test")
        test(copy.deepcopy(policy_b), alpha=alpha)


def test(policy_b_trained, alpha=0.01):
    policy_o_random = PokerNet().eval()
    optimizer = optim.Adam(policy_b_trained.parameters(), lr=alpha)

    list_loss = []
    for episode in range(1000):
        policy_loss = get_policy_loss(policy_b_trained, policy_o_random, n_sample=10)
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        # log
        list_loss.append(policy_loss.item())

        # Print running average
        if episode % 100 == 0:
            print(f"{episode=}, {np.mean(list_loss[-1000:])=}")


if __name__ == "__main__":
    main()
