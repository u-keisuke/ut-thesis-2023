# CFR

import copy
from collections import deque
from itertools import combinations

from utils import add_list_to_dict


class Node:
    def __init__(self, player, terminal, eu=0):  # 末端ノード以外はeu(utility)=0
        self.children = {}
        self.player = player
        self.terminal = terminal
        self.private_cards = []
        self.history = []
        self.information = ((), ())  # (private card, history)

        self.pi = 0
        self.pi_mi = 0  # pi_-i
        self.pi_i = 0  # pi_i
        self.true_pi_mi = 0  # used only when computing exploitability
        self.eu = eu
        self.cv = 0
        self.cfr = (
            {}
        )  # counter-factual regret of not taking action a at history h(not information I)

        self.pi_i_sum = 0  # denominator of average strategy (sumは全てのiterationについて)
        self.pi_sigma_sum = {}  # numerator of average strategy (sumは全てのiterationについて)
        self.num_updates = 0

    def expand_child_node(
        self, action, next_player, terminal, utility=0, private_cards=None
    ):
        """現在のノードの子ノードを作成して（すでにあればそれを）返す"""
        if action in self.children:
            return self.children[action]

        next_node = Node(next_player, terminal, utility)
        self.children[action] = next_node
        self.cfr[action] = 0
        self.pi_sigma_sum[action] = 0
        next_node.private_cards = (
            self.private_cards if private_cards is None else private_cards
        )
        next_node.history = self.history + [
            action
        ]  # if self.player != -1 else self.history
        next_node.information = (
            next_node.private_cards[next_player],
            tuple(next_node.history),
        )
        return next_node


class Card:
    def __init__(self, rank, suit=None):
        self.rank = rank
        self.suit = suit

    def __str__(self):
        if self.suit is None:
            return str(self.rank)
        else:
            return str(self.rank) + str(self.suit)


class KuhnPoker:
    def __init__(self):
        self.num_players = 2
        self.deck = [i for i in range(3)]
        self.information_sets = {player: {} for player in range(-1, self.num_players)}
        self.root = self._build_game_tree()

    def _build_game_tree(self):
        stack = deque()
        next_player = -1
        root = Node(next_player, False)
        add_list_to_dict(self.information_sets[next_player], root.information, root)
        for hand_0 in combinations(self.deck, 1):
            for hand_1 in combinations(self.deck, 1):
                if set(hand_0) & set(hand_1):
                    continue
                private_cards = [hand_0, hand_1, ()]  # p1, p2, chance player
                next_player = 0
                node = root.expand_child_node(
                    str(*hand_0) + "," + str(*hand_1),
                    next_player,
                    False,
                    private_cards=private_cards,
                )
                add_list_to_dict(
                    self.information_sets[next_player], node.information, node
                )
                stack.append(node)
                for action in ["check", "bet"]:  # player 0 actions
                    next_player = 1
                    node = node.expand_child_node(action, next_player, False)
                    add_list_to_dict(
                        self.information_sets[next_player], node.information, node
                    )
                    stack.append(node)
                    if action == "check":  # player 0 actions
                        for action in ["check", "bet"]:  # player 1 actions
                            if action == "check":
                                utility = self._compute_utility(
                                    action, next_player, hand_0, hand_1
                                )
                                next_player = -1
                                node = node.expand_child_node(
                                    action, next_player, True, utility
                                )
                                add_list_to_dict(
                                    self.information_sets[next_player],
                                    node.information,
                                    node,
                                )
                                node = stack.pop()
                            if action == "bet":
                                next_player = 0
                                node = node.expand_child_node(
                                    action, next_player, False
                                )
                                add_list_to_dict(
                                    self.information_sets[next_player],
                                    node.information,
                                    node,
                                )
                                stack.append(node)
                                for action in ["fold", "call"]:  # player 0 actions
                                    utility = self._compute_utility(
                                        action, next_player, hand_0, hand_1
                                    )
                                    next_player = -1
                                    node = node.expand_child_node(
                                        action, next_player, True, utility
                                    )
                                    add_list_to_dict(
                                        self.information_sets[next_player],
                                        node.information,
                                        node,
                                    )
                                    node = stack.pop()
                    if action == "bet":
                        stack.append(node)
                        for action in ["fold", "call"]:  # player 1 actions
                            utility = self._compute_utility(
                                action, next_player, hand_0, hand_1
                            )
                            next_player = -1
                            node = node.expand_child_node(
                                action, next_player, True, utility
                            )
                            add_list_to_dict(
                                self.information_sets[next_player],
                                node.information,
                                node,
                            )
                            node = stack.pop()
        return root

    def _compute_utility(self, action, player, hand_0, hand_1):
        """player 0(先手)にとってのutilityを計算する"""
        card_0, card_1 = hand_0[0], hand_1[0]
        is_win = card_0 > card_1
        if action == "fold":
            utility = 1 if player == 1 else -1
        elif action == "check":
            utility = 1 if is_win else -1
        elif action == "call":
            utility = 2 if is_win else -2
        else:
            utility = 0
        return utility


class LeducPoker:
    def __init__(self):
        self.num_players = 2
        self.deck = [i for i in range(6)]  # J1, J2, Q1, Q2, K1, K2
        self.information_sets = {player: {} for player in range(-1, self.num_players)}
        self._game_pattern = [
            ("call", "call"),
            ("call", "raise", "fold"),
            ("call", "raise", "call"),
            ("call", "raise", "raise", "fold"),
            ("call", "raise", "raise", "call"),
            ("raise", "fold"),
            ("raise", "call"),
            ("raise", "raise", "fold"),
            ("raise", "raise", "call"),
        ]
        self._raise_amount_round_1 = 1
        self._raise_amount_round_2 = 2
        self.root = self._build_game_tree()

    def _build_game_tree(self):
        next_player = -1  # 手番のプレイヤー
        root = Node(next_player, False)
        add_list_to_dict(self.information_sets[next_player], root.information, root)

        for hand_0 in combinations(self.deck, 1):
            for hand_1 in combinations(self.deck, 1):
                for hand_chance in combinations(self.deck, 1):
                    if (
                        (set(hand_0) & set(hand_1))
                        | (set(hand_0) & set(hand_chance))
                        | (set(hand_1) & set(hand_chance))
                    ):
                        continue
                    private_cards = [hand_0, hand_1, ()]  # p1, p2, chance player

                    next_player = 0
                    node = root.expand_child_node(
                        str(*hand_0) + "," + str(*hand_1),
                        next_player,
                        False,
                        private_cards=private_cards,
                    )
                    add_list_to_dict(
                        self.information_sets[next_player],
                        node.information,
                        node,
                    )

                    node_after_hands = node
                    for actions_round_1 in self._game_pattern:
                        for actions_round_2 in self._game_pattern:
                            node = node_after_hands

                            is_only_round_1 = actions_round_1[-1] == "fold"
                            next_player = 0
                            for i_action, action in enumerate(actions_round_1):
                                is_round_terminal = (
                                    False
                                    if i_action < len(actions_round_1) - 1
                                    else True
                                )
                                # round 1で終了する場合
                                if is_only_round_1:
                                    # round_1の最終アクションの場合
                                    if is_round_terminal:
                                        utility = self._compute_utility(
                                            hand_0,
                                            hand_1,
                                            hand_chance,
                                            is_only_round_1,
                                            actions_round_1,
                                            actions_round_2,
                                        )
                                        next_player = -1
                                        node = node.expand_child_node(
                                            action, next_player, True, utility
                                        )
                                    # round_1の途中アクションの場合
                                    else:
                                        next_player = 1 - next_player
                                        node = node.expand_child_node(
                                            action, next_player, False
                                        )
                                # round 2まで続く場合
                                else:
                                    # round_1の最終アクションの場合
                                    if is_round_terminal:
                                        next_player = -1
                                        node = node.expand_child_node(
                                            action, next_player, False
                                        )
                                    # round_1の途中アクションの場合
                                    else:
                                        next_player = 1 - next_player
                                        node = node.expand_child_node(
                                            action, next_player, False
                                        )
                                add_list_to_dict(
                                    self.information_sets[next_player],
                                    node.information,
                                    node,
                                )

                            # round 2まで続く場合
                            if not is_only_round_1:
                                next_player = 0
                                node = node.expand_child_node(
                                    str(*hand_chance),
                                    next_player,
                                    False,
                                )
                                add_list_to_dict(
                                    self.information_sets[next_player],
                                    node.information,
                                    node,
                                )

                                next_player = 0
                                for i_action, action in enumerate(actions_round_2):
                                    is_terminal = (
                                        False
                                        if i_action < len(actions_round_2) - 1
                                        else True
                                    )
                                    # round 2の最終アクションの場合
                                    if is_terminal:
                                        utility = self._compute_utility(
                                            hand_0,
                                            hand_1,
                                            hand_chance,
                                            is_only_round_1,
                                            actions_round_1,
                                            actions_round_2,
                                        )
                                        next_player = -1
                                        node = node.expand_child_node(
                                            action, next_player, True, utility
                                        )
                                    # round 2の途中アクションの場合
                                    else:
                                        next_player = 1 - next_player
                                        node = node.expand_child_node(
                                            action, next_player, False
                                        )
                                    add_list_to_dict(
                                        self.information_sets[next_player],
                                        node.information,
                                        node,
                                    )

        return root

    def _compute_utility(
        self,
        hand_0,
        hand_1,
        hand_chance,
        is_only_round_1,
        actions_round_1,
        actions_round_2=None,
    ):
        card_0, card_1, card_chance = hand_0[0], hand_1[0], hand_chance[0]
        pots = self._calc_pot(actions_round_1, [1, 1], self._raise_amount_round_1)
        if is_only_round_1:
            # foldでゲーム終了が確定している
            is_win = len(actions_round_1) % 2 == 0
            utility = pots[1] if is_win else -pots[0]
        else:
            assert actions_round_2 is not None
            pots = self._calc_pot(actions_round_2, pots, self._raise_amount_round_2)
            if actions_round_2[-1] == "fold":
                is_win = len(actions_round_2) % 2 == 0
                utility = pots[1] if is_win else -pots[0]
            else:
                is_win = self._is_win_pl_0(card_0, card_1, card_chance)
                utility = pots[1] if is_win else -pots[0]

        return utility

    def _is_win_pl_0(self, card_0, card_1, card_chance):
        """player 0が勝ったかどうか"""
        # J,Q,Kのみを考える
        card_0, card_1, card_chance = card_0 % 3, card_1 % 3, card_chance % 3

        if (card_0 == card_chance) & (card_1 != card_chance):
            return True
        elif (card_0 != card_chance) & (card_1 == card_chance):
            return False
        else:
            if max(card_0, card_chance) > max(card_1, card_chance):
                return True
            elif max(card_0, card_chance) < max(card_1, card_chance):
                return False
            else:
                return card_0 > card_1

    def _calc_pot(self, actions, pots, raise_amount):
        player = 0
        for action in actions:
            if action == "call":
                pots[player] = pots[1 - player]
            elif action == "raise":
                pots[player] = pots[1 - player] + raise_amount
            elif action == "fold":
                pass
            player = 1 - player
        return pots

    # judge if it has ended
    def _is_terminal(self, actions):
        return True


if __name__ == "__main__":
    kuhn_poker = KuhnPoker()
