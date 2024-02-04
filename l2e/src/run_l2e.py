import copy
import pickle

import numpy as np
from adapter import AdapterRandom
from base_policy_training import train_base_policy
from diverse_osg import diverse_osg
from hard_osg import hard_osg
from models import PokerNet


def main(epochs=300):
    alpha = 0.1
    beta = 0.01
    policy_b = PokerNet()  # B
    policy_o = PokerNet()  # O
    policy_o_list = [policy_o]

    adapter = AdapterRandom(
        n_episode=10,
        n_sample=10,
    )

    for e in range(epochs):
        print(f"{e=}")

        print("hard_osg")
        policy_o = hard_osg(
            copy.deepcopy(policy_b), n_epochs=20, n_sample=20, alpha=alpha
        )  # n_sample=20

        print("diverse_osg")
        policy_o_list += diverse_osg(
            copy.deepcopy(policy_b),
            policy_o,
            n_opponents=5,
            n_step=500,
            n_sample=10,
            alpha=alpha,
            alpha_mmd=0.8,
        )

        policy_o_list_sampled = np.random.choice(
            policy_o_list, size=min(40, len(policy_o_list))
        )

        print("train_base_policy")
        policy_b = train_base_policy(
            policy_b, policy_o_list_sampled, alpha=alpha, beta=beta, n_sample=20
        )

        print("test - adapt to random policy")
        adapter.adapt(copy.deepcopy(policy_b), optimizer_type="Adam", alpha=0.01)

        # save policy_o_list
        with open(f"policy_o_list_{e:04}.pickle", "wb") as f:
            pickle.dump(policy_o_list, f)


if __name__ == "__main__":
    main()
