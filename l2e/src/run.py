import copy

import numpy as np
from base_policy_training import train_base_policy
from diverse_osg import diverse_osg
from hard_osg import hard_osg
from models import PokerNet
from tester import tester_random


def main(epochs=50):
    alpha = 0.1
    beta = 0.01
    policy_b = PokerNet()  # B
    policy_o = PokerNet()  # O
    policy_o_list = [policy_o]

    tester = tester_random()

    for e in range(epochs):
        print(f"{e=}")

        print("hard_osg")
        policy_o = hard_osg(policy_b, n_epochs=20, n_sample=20, alpha=alpha)

        print("diverse_osg")
        policy_o_list += diverse_osg(
            policy_b,
            policy_o,
            n_opponents=5,
            n_steps=10000,
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
        tester.test(copy.deepcopy(policy_b), alpha=alpha)


if __name__ == "__main__":
    main()
