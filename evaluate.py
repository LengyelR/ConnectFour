import os
import logging
import numpy as np
from collections import Counter

import connect4
import inference
import search

_logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(self, folder, prev_gen, new_gen):
        self.prev_gen = prev_gen
        self.new_gen = new_gen
        self.engine = connect4.GameEngine()

        root = os.path.join(folder, 'model')

        new_model = inference.FrozenModel(os.path.join(root, new_gen, 'frozen_model.pb'))
        old_model = inference.FrozenModel(os.path.join(root, prev_gen, 'frozen_model.pb'))
        new = search.Mcts(800, new_model, self.engine)
        old = search.Mcts(800, old_model, self.engine)
        self.mcts = {1: new, 2: old}

    def _play(self, swap):
        s = self.engine.empty_board()
        player = 1
        while True:
            idx = 3 - player if swap else player
            pi, children = self.mcts[idx].search(s, player)

            a = np.random.choice(7, p=pi)
            best_node = children[a]
            if best_node.terminal is not None:
                break

            s = best_node.state
            player = 3 - player

        winner = best_node.player if best_node.terminal == 1 else 0
        return winner, best_node.state

    def compare(self, n, swap):
        stats = []
        for i in range(n):
            winner, s = self._play(swap)
            stats.append(winner)
            self.engine.show_board(s)
            print()

            _logger.debug(f'winner: {winner} (as first player:{not swap})')
        return Counter(stats)


def main(folder, prev_gen, new_gen):
    ev = Evaluator(folder, prev_gen, new_gen)

    res1 = ev.compare(100, False)
    _logger.info(res1, 'as 1st player')

    res2 = ev.compare(100, True)
    _logger.info(res2, 'as 2nd player')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--folder', '-o',
        type=str,
        default='',
        help='Root folder.'
    )
    parser.add_argument(
        '--prev_gen', '-pg',
        type=str,
        default='gen-0',
        help='Generation for player 2.'
    )
    parser.add_argument(
        '--new_gen', '-ng',
        type=str,
        default='gen-1',
        help='Generation for player 1.'
    )

    flags, _ = parser.parse_known_args()
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh = logging.FileHandler(os.path.join(flags.folder, 'evaluations.log'))
    fh.setFormatter(formatter)

    _logger.setLevel(logging.DEBUG)
    _logger.addHandler(fh)

    main(flags.folder, flags.prev_gen, flags.new_gen)
