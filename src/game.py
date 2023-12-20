class KuhnPoker:
    def __init__(self):
        self.PASS = 0
        self.BET = 1
        self.N_ACTIONS = 2
        self.ACTION_CHAR = "pb"

    def is_terminal(self, h):
        return h.endswith("bp") or h.endswith("bb") or h.endswith("pp")

    def _scale(self, winner, player, weight):
        assert winner == 0 or winner == 1, "winner must be 0 or 1"
        assert player == 0 or player == 1, "player must be 0 or 1"
        sign = 1 if winner == player else -1
        return sign * weight

    def utility(self, h, player):
        """utility of terminal history h with respect to player"""
        assert self.is_terminal(h)
        assert player == 0 or player == 1
        if h.endswith("bp"):
            winner = len(h) % 2
            return self._scale(winner, player, 1)
        winner = 0 if int(h[0]) > int(h[1]) else 1
        weight = 1 if h.endswith("pp") else 2
        return self._scale(winner, player, weight)

    def make_info_set(self, h, i):
        """hide invisible information for player i of history h"""
        return ("?" + h[1:]) if i == 1 else (h[0] + "?" + h[2:])

    def turn(self, h):
        """player to act at history h"""
        return len(h) % 2
