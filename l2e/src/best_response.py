from adapter import Adapter
from models import PokerNet


def get_best_response(
    policy_opponent,
    optimizer_type,
    lr=0.01,
    n_episode=10000,
    n_sample=10,
    verbose=False,
):
    policy = PokerNet()
    policy_opponent.eval()

    adapter = Adapter(policy_opponent, n_episode=n_episode, n_sample=n_sample)
    results = adapter.adapt(policy, optimizer_type, alpha=lr, verbose=verbose)

    return policy, results
