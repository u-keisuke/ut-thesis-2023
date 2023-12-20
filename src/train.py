import argparse
import os
import pickle
import time
from copy import deepcopy

from tqdm import tqdm

from cfr import get_exploitability, update_node_values, update_pi, update_strategy
from poker import KuhnPoker, Node
from utils import get_csv_saver, get_logger


def get_initial_strategy_profile(node: Node, num_players=None, strategy_profile=None):
    if node.terminal:
        return strategy_profile
    if strategy_profile is None:
        strategy_profile = {
            player: {} for player in range(-1, num_players)
        }  # chance nodeのために+1する
    if node.information not in strategy_profile[node.player]:
        strategy_profile[node.player][node.information] = {
            action: 1 / len(node.children) for action in node.children
        }
    for child in node.children.values():
        strategy_profile = get_initial_strategy_profile(
            child, strategy_profile=strategy_profile
        )
    return strategy_profile


def train(num_iter, log_schedule, args):
    game = KuhnPoker()
    strategy_profile = get_initial_strategy_profile(game.root, game.num_players)
    average_strategy_profile = deepcopy(strategy_profile)
    average_strategy_profile_dict = {}

    for t in tqdm(range(num_iter)):
        update_pi(
            game.root,
            strategy_profile,
            average_strategy_profile,
            [1.0 for _ in range(game.num_players + 1)],
            [1.0 for _ in range(game.num_players + 1)],
            [1.0 for _ in range(game.num_players + 1)],
        )

        update_node_values(game.root, strategy_profile, args)

        update_strategy(
            strategy_profile, average_strategy_profile, game.information_sets
        )

        # log & save
        if t % log_schedule(t) == 0:
            # exploitability
            exploitability, tmp = get_exploitability(game, average_strategy_profile)
            utility_br0_ev1, utility_ev0_br1 = tmp

            # store average_strategy_profile
            average_strategy_profile_dict[t] = deepcopy(average_strategy_profile)

            # time
            time_now = time.time()
            time_elapsed_ms = (time_now - TIME_START) * 1000

            logger.debug(f"{t}, {time_elapsed_ms}, {exploitability}")
            log_exploitability(
                [t, time_elapsed_ms, exploitability, utility_br0_ev1, utility_ev0_br1]
            )

    # save average_strategy_profile
    with open(os.path.join(FOLDER_SAVE, "average_strategy_profile.pkl"), "wb") as f:
        pickle.dump(average_strategy_profile_dict, f)

    return average_strategy_profile


def main(args):
    num_updates = int(1e4)
    average_strategy_profile = train(
        num_updates, lambda x: (10 ** (len(str(x)) - 1)), args
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cs", action="store_true")  # chance sampling
    parser.add_argument("--os", action="store_true")  # outcome sampling
    args = parser.parse_args()

    if args.cs:
        type_sampling = "cs"
    elif args.os:
        type_sampling = "os"
    else:
        type_sampling = "vanilla"

    TIME_START = time.time()
    STR_TIME_START = time.strftime(
        f"%Y%m%d-%H%M%S-{type_sampling}", time.localtime(TIME_START)
    )
    FOLDER_SAVE = f"../logs/{STR_TIME_START}"

    logger = get_logger(__name__, os.path.join(FOLDER_SAVE, f"{__name__}.log"))
    log_exploitability = get_csv_saver(os.path.join(FOLDER_SAVE, f"exploitability.csv"))

    logger.debug(args)

    main(args)
