import os
import logging
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
        root = {}
        idx = 2 if swap else 1
        s = self.engine.empty_board()

        # first 2 moves to populate root dictionary
        root1 = search.Node(0, 0, s, 2)
        _, root1 = self.mcts[idx].search(root1)
        root[idx] = root1
        idx = 3 - idx

        root2 = search.Node(0, 0, root1.state, 1)
        root2.depth += 1
        _, root2 = self.mcts[idx].search(root2)
        root[idx] = root2

        while True:
            idx = 3 - idx

            # moving root to next state, based on other player's action
            previous_action = root[3 - idx].action
            root[idx] = root[idx].children[previous_action]
            pi, root[idx] = self.mcts[idx].search(root[idx])

            if root[idx].terminal is not None:
                break

        winner = root[idx].player if root[idx].terminal == 1 else 0
        return winner, root[idx].state

    def compare(self, n, swap):
        stats = []
        for i in range(n):
            winner, s = self._play(swap)
            stats.append(winner)

            print()
            self.engine.show_board(s)
            _logger.debug(f'winner: {winner} (as first player:{not swap})')
        return Counter(stats)


def main(folder, prev_gen, new_gen):
    _logger.info(f'{prev_gen}  vs  {new_gen}')
    ev = Evaluator(folder, prev_gen, new_gen)

    res1 = ev.compare(100, False)
    _logger.info(f'{res1}  (as 1st player)')

    res2 = ev.compare(100, True)
    _logger.info(f'{res2}  (as 2nd player)')


if __name__ == "__main__":
    import argparse
    import utils
    utils.try_init_colorama()

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
