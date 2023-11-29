import numpy as np

from cfr import calculate_p_i, make_strategy, possible_history


def test_make_strategy():
    regret_sum = np.array([1, 1])
    strategy = make_strategy(regret_sum)
    assert np.array_equal(strategy, np.array([0.5, 0.5]))


def test_calculate_p_i():
    h = "01b"
    i = 1
    table = {
        "0?": {
            "regret_sum": np.array([1, 1]),
            "strategy_sum": np.array([0.16666667, 0.16666667]),
            "regret_update": np.array([0.0, 0.0]),
            "visit": 4,
        }
    }
    assert calculate_p_i(h, i, table) == 1 / 6 * 1 / 6 * 1 / 2

    h = "12pb"
    i = 0
    table = {
        "?2p": {
            "regret_sum": np.array([2, 1]),
            "strategy_sum": np.array([0.16666667, 0.16666667]),
            "regret_update": np.array([0.0, 0.0]),
            "visit": 4,
        }
    }
    assert calculate_p_i(h, i, table) == 1 / 6 * 1 / 6 * 1 / 3


def test_possible_history():
    infoset = "0?bp"
    assert possible_history(infoset) == ["01bp", "02bp"]
