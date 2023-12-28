import argparse
import os
import pickle
import time
from copy import deepcopy

from tqdm import tqdm

from cfr import (
    get_initial_strategy_profile,
    update_node_values,
    update_pi,
    update_strategy,
)
from exploitability import get_exploitability
from poker import KuhnPoker, LeducPoker
from utils import get_csv_saver, get_logger


def train(num_iter, log_schedule, args):
    if args.game == "kuhn":
        game = KuhnPoker()
    elif args.game == "leduc":
        game = LeducPoker()

    # save game tree
    with open(os.path.join(FOLDER_SAVE, "game_tree_initial.pkl"), "wb") as f:
        pickle.dump(game, f)

    strategy_profile = get_initial_strategy_profile(game.root, game.num_players)
    average_strategy_profile = deepcopy(strategy_profile)
    average_strategy_profile_dict = {}

    # save initial strategy profile
    with open(os.path.join(FOLDER_SAVE, "strategy_profile_initial.pkl"), "wb") as f:
        pickle.dump(strategy_profile, f)

    for t in tqdm(range(num_iter)):
        update_pi(
            game.root,
            strategy_profile,
            average_strategy_profile,
            [1.0 for _ in range(game.num_players + 1)],
            [1.0 for _ in range(game.num_players + 1)],
            [1.0 for _ in range(game.num_players + 1)],
        )

        update_node_values(game.root, strategy_profile, args.sampling)

        update_strategy(
            strategy_profile, average_strategy_profile, game.information_sets
        )

        # log & save
        if log_schedule(t):
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
    log_schedule = lambda t: t % (10 ** (len(str(t)) - 1)) == 0
    # log_schedule = lambda t: (t == 1000) or (t == 10000)
    average_strategy_profile = train(num_updates, log_schedule, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sampling",
        "-s",
        choices=["vanilla", "cs", "os"],
        default="vanilla",
    )
    parser.add_argument("--game", "-g", choices=["kuhn", "leduc"], default="kuhn")
    args = parser.parse_args()

    TIME_START = time.time()
    STR_TIME_START = time.strftime("%Y%m%d-%H%M%S", time.localtime(TIME_START))
    FOLDER_SAVE = f"../logs/{STR_TIME_START}-{args.game}-{args.sampling}"

    logger = get_logger(__name__, os.path.join(FOLDER_SAVE, f"{__name__}.log"))
    log_exploitability = get_csv_saver(os.path.join(FOLDER_SAVE, f"exploitability.csv"))

    logger.debug(args)

    main(args)
