import os
import pickle
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.append(os.path.join("../../cfr/src"))
import poker

DEVICE = "cpu"

FOLDER = "20240103-221639-leduc-vanilla"  # FOLDER = "20240104-110439-kuhn-vanilla"
with open(f"../../cfr/logs/{FOLDER}/game_tree_initial.pkl", "rb") as f:
    GAME = pickle.load(f)
with open(f"../../cfr/logs/{FOLDER}/average_strategy_profile.pkl", "rb") as f:
    SP = pickle.load(f)

SP_list = list(SP[9000][0].keys()) + list(SP[9000][1].keys())

NUM_INPUTS = len(SP_list)
NUM_ACTIONS = 3
ACTION_SPACE = ["call", "raise", "fold"]


def preprocess_input_tabular(info_set):
    input_data = np.zeros(NUM_INPUTS)
    input_data[SP_list.index(info_set)] = 1

    input_data = torch.from_numpy(input_data).float().unsqueeze(0)
    input_data = input_data.to(DEVICE)

    return input_data


class PokerEnv:
    def __init__(self, opponent_policy):
        self._game_tree = GAME
        self._opponent_policy = opponent_policy
        self._opponent_policy.eval()
        self.action_space = ACTION_SPACE
        self._raise_amount = [1, 2]
        self.reset()

    def _distribute_hands(self):
        # select randomly
        self._hands_players_str = np.random.choice(list(GAME.root.children.keys()))
        self._hands_players = self._hands_players_str.split(",")
        self._current_node = GAME.root.children[self._hands_players_str]

    def reset(self):
        self._distribute_hands()
        self._hand_chance = None
        self._num_round = 0
        self._pots = [1, 1]
        # ランダムに先手後手を決める
        self._player = np.random.choice([0, 1])
        self._player_opponent = 1 - self._player
        self._info_set = ((int(self._hands_players[self._player]),), ())

        self._step_env()

        return self.output_state()

    def output_state(self):
        """return preprocess_input(
            self._hands_players[self._current_node.player],
            self._hand_chance,
            self._pots[self._current_node.player],
            self._pots[1 - self._current_node.player]
        )"""
        return preprocess_input_tabular(self._info_set)

    def step(self, action_idx: int):
        """playerの行動を決定する"""
        action_str = self.action_space[action_idx]
        self._current_node = self._current_node.children[action_str]
        self._info_set = (self._info_set[0], self._info_set[1] + (action_str,))

        self._step_env()

        if self._current_node.terminal:
            return None, self._current_node.eu, self._current_node.terminal, None
        else:
            return (
                self.output_state(),
                self._current_node.eu,
                self._current_node.terminal,
                None,
            )

    def _step_env(self):
        """player以外の行動を決定する"""
        while self._current_node.player != self._player:
            if self._current_node.terminal:
                break

            action_str = "hogehoge"
            if self._current_node.player == -1:
                self._hand_chance = np.random.choice(
                    list(self._current_node.children.keys())
                )
                action_str = self._hand_chance
                self._num_round += 1
            elif self._current_node.player == self._player_opponent:
                input_data = self.output_state()
                out = self._opponent_policy(input_data)  # call, raise, fold
                probs = F.softmax(out, dim=1)
                action = select_valid_action(probs, self._current_node.children.keys())
                action_str = self.action_space[action]

            self._current_node = self._current_node.children[action_str]
            self._info_set = (self._info_set[0], self._info_set[1] + (action_str,))

    def _update_pots(self, action):
        # pots update
        if action == "call":
            self._pots[self._current_node.player] = self._pots[
                1 - self._current_node.player
            ]
        elif action == "raise":
            self._pots[self._current_node.player] = (
                self._pots[1 - self._current_node.player]
                + self._raise_amount[self._num_round]
            )
        elif action == "fold":
            pass


def select_valid_action(out, valid_actions_str, action_space: list = ACTION_SPACE):
    valid_actions_idx = [action_space.index(a) for a in valid_actions_str]
    out_valid = out[:, valid_actions_idx]
    actions_prob = F.softmax(out_valid, dim=1).detach().numpy()
    selected_action = np.random.choice(valid_actions_idx, p=actions_prob.reshape(-1))
    return selected_action


def get_policy_loss(policy, policy_env, n_sample):
    policy_loss_list = []
    env = PokerEnv(policy_env)
    for _ in range(n_sample):
        state = env.reset()
        done = False

        log_prob_list = []
        while not done:
            out = policy(state)
            action_idx = select_valid_action(
                out, env._current_node.children.keys()
            )  # 出力確率に基づいてactionを選択
            next_state, reward, done, _ = env.step(action_idx)  # terminal以外はreward=0
            log_probs = F.log_softmax(out, dim=1)
            log_prob_list.append(log_probs.squeeze(0)[action_idx])
            state = next_state

        policy_loss = -(torch.stack(log_prob_list) * reward).mean()
        policy_loss_list.append(policy_loss)

    E_policy_loss = torch.stack(policy_loss_list).mean()

    return E_policy_loss
