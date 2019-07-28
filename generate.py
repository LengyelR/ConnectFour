import numpy as np
import os
import uuid
import pickle
import logging

import search
import connect4
import inference


_logger = logging.getLogger(__name__)


def _update_match(winner, game):
    if winner == 0:
        z = 0
    else:
        z = 1 if winner == 1 else -1

    for step in game:
        step.append(z)


def self_play(gen, iteration, tau, folder='', batches=10, n=100):
    """
    :param gen: folder name of the generation (e.g. gen-4)
    :param iteration: execute this many mcts iterations
    :param tau: mcts parameter, controlling exploration
    :param folder: root folder (training/generation/guid/data will be created here)
    :param batches: number of batches (each batch contains n games)
    :param n: number of games to play in a batch
    :return:
    """

    frozen_model_path = os.path.join('model', gen, 'frozen_model.pb')

    guid = uuid.uuid4()
    folder = os.path.join(folder, 'training', gen, str(guid))
    os.makedirs(folder)

    engine = connect4.GameEngine()
    model = inference.FrozenModel(frozen_model_path)
    mcts = search.Mcts(iteration, model, engine)

    for batch_no in range(batches):
        self_play_data = []
        _logger.debug(f'generating: {guid} -> {batch_no}')

        for i in range(n):
            s = engine.empty_board()
            player = 1
            game_steps = []
            while True:
                pi, children = mcts.search(s, player, tau)
                a = np.random.choice(7, p=pi)
                best_node = children[a]
                game_steps.append([s, pi, best_node.Q, best_node.player])

                if best_node.terminal is not None:
                    break

                s = best_node.state
                player = 3 - player

            winner = best_node.player if best_node.terminal == 1 else 0
            _update_match(winner, game_steps)

            _logger.debug(f'game{i} has been finished. Winner is {winner}')
            self_play_data.extend(game_steps)
        _logger.debug(f'saving: {guid} -> {batch_no}')
        data_name = f'{batch_no}_{n}_self_play.pkl'
        full_path = os.path.join(folder, data_name)
        with open(full_path, 'wb') as f:
            pickle.dump(self_play_data, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--batches', '-b',
        type=int,
        default=20,
        help='Number of batches. Each batch of games will be saved separately.'
    )
    parser.add_argument(
        '--batch_size', '-n',
        type=int,
        default=250,
        help='The size of each batch. This many games will be played in a batch.'
    )
    parser.add_argument(
        '--generation', '-g',
        type=str,
        default='gen-0',
        help='Network generation to be used for self-play.'
    )
    parser.add_argument(
        '--folder', '-o',
        type=str,
        default='',
        help='Root folder.'
    )
    parser.add_argument(
        '--iter', '-i',
        type=int,
        default=800,
        help='Mcts iteration step count.'
    )
    parser.add_argument(
        '--tau', '-t',
        type=float,
        default=1.0,
        help='Temperature parameter for the exponentiated visit counts.'
    )

    flags, _ = parser.parse_known_args()

    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh = logging.FileHandler(os.path.join(flags.folder, 'selfplay.log'))
    fh.setFormatter(formatter)

    _logger.setLevel(logging.DEBUG)
    _logger.addHandler(fh)

    self_play(flags.generation,
              flags.iter,
              flags.tau,
              flags.folder,
              flags.batches,
              flags.batch_size)
