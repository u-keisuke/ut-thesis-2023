import os
import pickle
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.append(os.path.join("../../cfr/src"))
import poker
from utils_l2e import select_valid_action

FOLDER = "20240103-221639-leduc-vanilla"  # FOLDER = "20240104-110439-kuhn-vanilla"
with open(f"../../cfr/logs/{FOLDER}/game_tree_initial.pkl", "rb") as f:
    GAME = pickle.load(f)
with open(f"../../cfr/logs/{FOLDER}/average_strategy_profile.pkl", "rb") as f:
    SP = pickle.load(f)

INFOSET_LIST = list(SP[9000][0].keys()) + list(SP[9000][1].keys())
NUM_INPUTS = len(INFOSET_LIST)
NUM_ACTIONS = 3
ACTION_SPACE = ["call", "raise", "fold"]


class PokerEnv:
    def __init__(self, opponent_policy):
        self._game_tree = GAME
        self._opponent_policy = opponent_policy
        self._opponent_policy.eval()
        self.action_space = ACTION_SPACE
        self._raise_amount = [1, 2]
        self.distribute_hands()

    def distribute_hands(self):
        # ランダムに先手後手を決める
        self._player = np.random.choice([0, 1])
        self._player_opponent = 1 - self._player
        # ランダムに手札を決める
        self._hands_players_str = np.random.choice(list(GAME.root.children.keys()))
        self._hands_players = self._hands_players_str.split(",")  # list(str)
        self._hand_chance = np.random.choice(
            [str(h) for h in range(6) if str(h) not in self._hands_players]
        )  # round 0終了後に出される公開カード

    def start(self):
        """ゲームを初期化する．ただしハンドや先手後手は変えない．"""
        self._current_node = GAME.root.children[self._hands_players_str]
        self._info_set = ((int(self._hands_players[self._player]),), ())
        self._num_round = 0
        self.step_env()

    def output_state(self):
        """info_setをone-hot vector (torch.tensor)に変換する
        input: info_set
        output: one-hot vector (torch.tensor)
        """
        vec = np.zeros(NUM_INPUTS)
        vec[INFOSET_LIST.index(self._info_set)] = 1
        vec = torch.from_numpy(vec).float().unsqueeze(0)
        return vec

    def step(self, action_idx: int):
        self.step_player(action_idx)
        self.step_env()
        return (
            self.output_state() if not self._current_node.terminal else None,
            self._current_node.eu,
            self._current_node.terminal,
        )

    def step_player(self, action_idx: int):
        """ゲームを進める
        input: playerのアクション
        output: next_state, reward, done, info
        """
        action_str = self.action_space[action_idx]
        self._current_node = self._current_node.children[action_str]
        self._info_set = (self._info_set[0], self._info_set[1] + (action_str,))

    def step_env(self):
        """player以外の行動を決定する"""
        while (
            self._current_node.player != self._player
            and not self._current_node.terminal
        ):
            if self._current_node.player == -1:  # chance node
                action_str = self._hand_chance
                self._num_round += 1
            elif self._current_node.player == self._player_opponent:
                input_data = self.output_state()
                out = self._opponent_policy(input_data)  # call, raise, fold
                probs = F.softmax(out, dim=1)
                action = select_valid_action(
                    probs, self._current_node.children.keys(), self.action_space
                )
                action_str = self.action_space[action]
            else:
                raise ValueError(
                    f"invalid player: {self._current_node.player}, {self._player}, {self._player_opponent}"
                )

            self._current_node = self._current_node.children[action_str]
            self._info_set = (self._info_set[0], self._info_set[1] + (action_str,))
