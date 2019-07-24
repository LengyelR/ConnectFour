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


def self_play(gen, batches=10, n=100):
    """
    :param gen: folder name of the generation (e.g. gen-4)
    :param batches: number of batches (each batch contains n games)
    :param n: number of games to play in a batch
    :return:
    """

    frozen_model_path = os.path.join('model', gen, 'frozen_model.pb')

    guid = uuid.uuid4()
    folder = os.path.join('training', gen, str(guid))
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
            game_steps = []
            while True:
                pi, children = mcts.search(s, player, 0.01)
                a = np.random.choice(7, p=pi)
                best_node = children[a]
                game_steps.append((s, pi, best_node.Q, best_node.player))

                if best_node.terminal is not None:
                    break

                s = best_node.state
                player = 3 - player
            self_play_data.extend(game_steps)

            winner = best_node.player if best_node.terminal == 1 else 0
            logger.debug(f'game{i} has been finished. Winner is {winner}')
        logger.debug(f'saving: {guid} -> {batch_no}')
        data_name = f'{batch_no}_self_play.pkl'
        full_path = os.path.join(folder, data_name)
        with open(full_path, 'wb') as f:
            pickle.dump(self_play_data, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    self_play('gen-0', 5, 3)
