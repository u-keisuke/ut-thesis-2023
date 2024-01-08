import argparse
import os
import pickle
import time
from copy import deepcopy

from tqdm import tqdm

import cfr
import cfr_cs
from cfr import get_initial_strategy_profile
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

    time_update_cumu = 0
    for t in tqdm(range(num_iter)):
        start_iter = time.time()

        if args.sampling == "vanilla":
            cfr.update_pi(
                game.root,
                strategy_profile,
                average_strategy_profile,
                [1.0 for _ in range(game.num_players + 1)],
                [1.0 for _ in range(game.num_players + 1)],
                [1.0 for _ in range(game.num_players + 1)],
            )
            cfr.update_node_values_vanilla(game.root, strategy_profile)
            cfr.update_strategy(
                strategy_profile, average_strategy_profile, game.information_sets
            )

        elif args.sampling == "cs":
            cfr_cs.update_pi(
                game.root,
                strategy_profile,
                average_strategy_profile,
                [1.0 for _ in range(game.num_players + 1)],
                [1.0 for _ in range(game.num_players + 1)],
                [1.0 for _ in range(game.num_players + 1)],
            )
            cfr_cs.update_node_values(game.root, strategy_profile)
            cfr_cs.update_strategy(
                strategy_profile, average_strategy_profile, game.information_sets
            )

        end_iter = time.time()
        time_update_cumu += end_iter - start_iter

        # log & save
        if log_schedule(t):
            # calculate exploitability
            start_exp = time.time()
            exploitability, tmp, MEMO = get_exploitability(
                game, average_strategy_profile
            )
            utility_br0_ev1, utility_ev0_br1 = tmp
            end_exp = time.time()
            time_exp = end_exp - start_exp

            # store average_strategy_profile
            average_strategy_profile_dict[t] = deepcopy(average_strategy_profile)

            # logger
            logger.debug(
                f"{t=:6d}, {time_update_cumu=:10.4f}, {exploitability=:7.4f}, {time_exp=:7.4f}"
            )
            log_exploitability(
                [t, time_update_cumu, exploitability, utility_br0_ev1, utility_ev0_br1]
            )

    # save game tree
    with open(os.path.join(FOLDER_SAVE, "game_tree_final.pkl"), "wb") as f:
        pickle.dump(game, f)

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
