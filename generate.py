import numpy as np
import os
import uuid
import pickle
import logging

import search
import connect4
import inference

formatter = logging.Formatter('%(asctime)s - %(message)s')
fh = logging.FileHandler('selfplay.log')
fh.setFormatter(formatter)
logger = logging.getLogger('selfplay')
logger.setLevel(logging.DEBUG)
logger.addHandler(fh)


def self_play(frozen_model_path, batches=10, n=100):
    """
    :param frozen_model_path: path to a *.pb file
    :param batches: number of batches (each batch contains n games)
    :param n: number of games to play in a batch
    :return:
    """
    guid = uuid.uuid4()
    folder = os.path.join('training', str(guid))
    os.makedirs(folder)

    engine = connect4.GameEngine()
    model = inference.FrozenModel(frozen_model_path)
    mcts = search.Mcts(400, model, engine)

    for batch_no in range(batches):
        self_play_data = []
        logger.debug(f'generating: {guid} -> {batch_no}')

        for i in range(n):
            s = engine.empty_board()
            player = 1
            winner = None
            game_steps = []
            while winner is None:
                pi, q_values = mcts.search(s, player, 0.01)
                a = np.random.choice(7, p=pi)
                q = q_values[a]
                game_steps.append((s, pi, q, player))

                r, c, s = engine.move(a, s, player)
                winner = engine.has_player_won(r, c, s, player)

                if np.count_nonzero(s) == 42:
                    winner = 0
                player = 3 - player
            self_play_data.extend(game_steps)
            logger.debug(f'game{i} has been finished. Winner is {winner}')
        logger.debug(f'saving: {guid} -> {batch_no}')
        data_name = f'{batch_no}_self_play.pkl'
        full_path = os.path.join(folder, data_name)
        with open(full_path, 'wb') as f:
            pickle.dump(self_play_data, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    self_play(os.path.join('model', 'frozen_model.pb'), 5, 3)
