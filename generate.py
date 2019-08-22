import os
import uuid
import pickle
import logging
import time
import random
import ray

import search
import connect4
import inference

_logger = logging.getLogger(__name__)


class Process:
    def __init__(self, gen, iteration, tau, folder='', num_batches=10, batch_size=100, num_workers=100):
        """
        :param gen: folder name of the generation (e.g. gen-4)
        :param iteration: execute this many mcts iterations
        :param tau: mcts parameter, controlling exploration
        :param folder: root folder (training/generation/guid/data will be created here)
        :param num_batches: number of batches (each batch contains n games)
        :param batch_size: number of games to play in a batch
        :param num_workers: number of workers to launch
        :return:
        """
        self.gen = gen
        self.iteration = iteration
        self.tau = tau
        self.folder = folder
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.workers = [SelfPlay.remote(gen, iteration, folder) for _ in range(num_workers)]

    def start(self):
        guid = uuid.uuid4()
        training_folder = os.path.join(self.folder, 'training', self.gen, str(guid))
        os.makedirs(training_folder)

        for batch_no in range(self.num_batches):
            game_ids = self._start_workers()
            self._show_progress(game_ids, 120.0)

            batch = ray.get(game_ids)
            flattened = [step for game in batch for step in game]
            data_name = f'{batch_no}_{self.batch_size}_self_play.pkl'
            full_path = os.path.join(training_folder, data_name)

            _logger.debug(f'saving batch: {data_name}')
            with open(full_path, 'wb') as f:
                pickle.dump(flattened, f, pickle.HIGHEST_PROTOCOL)

    def _start_workers(self):
        i = 0
        game_ids = []
        while True:
            for worker in self.workers:
                if i == self.batch_size:
                    return game_ids
                game_ids.append(worker.self_play.remote(self.tau))
                i += 1

    def _show_progress(self, game_ids, freq):
        i = 0
        games = len(game_ids)
        while i < self.batch_size:
            i += self.num_workers
            num_returns = min(i, games)
            ready_ids, remaining_ids = ray.wait(game_ids, num_returns=num_returns, timeout=freq)
            _logger.debug(f'remaining {len(remaining_ids)} / {len(game_ids)}')


@ray.remote
class SelfPlay:
    def __init__(self, gen, iteration, folder=''):
        self.gen = gen
        self.iteration = iteration
        frozen_model_path = os.path.join(folder, 'model', gen, 'frozen_model.pb')

        # here tree and model are shared between players
        self.engine = connect4.GameEngine()
        self.model = inference.FrozenModel(frozen_model_path)
        self.mcts = search.Mcts(iteration, self.model, self.engine)

    def self_play(self, tau):
        s = self.engine.empty_board()
        root = search.Node(0, 0, s, 2)  # first move is the child node of root, so root belongs to 2nd player
        game_steps = []

        while True:
            pi, best_node = self.mcts.search(root, tau)
            game_steps.append([best_node.state, pi, best_node.Q, best_node.player])

            if best_node.terminal is not None:
                break

            root = best_node

        winner = best_node.player if best_node.terminal == 1 else 0
        self._update_match(winner, game_steps)

        return game_steps

    @staticmethod
    def _update_match(winner, game):
        if winner == 0:
            z = 0
        else:
            z = 1 if winner == 1 else -1

        for step in game:
            step.append(z)


def main(generation, iteration, tau, folder, num_batches, batch_size, num_workers, redis_address):

    @ray.remote
    def get_node_ip():
        time.sleep(0.01)
        return ray.services.get_node_ip_address()

    if redis_address is not None:
        ray.init(redis_address=redis_address)
    else:
        ray.init()

    nodes = set(ray.get([get_node_ip.remote() for _ in range(1000)]))
    num_machines = len(nodes)
    _logger.debug(f'Nodes: {num_machines}, {random.sample(nodes, min(5, num_workers, num_machines))}')
    _logger.debug(f'CPU count: {ray.cluster_resources()["CPU"]}')

    process = Process(
        generation,
        iteration,
        tau,
        folder,
        num_batches,
        batch_size,
        num_workers)
    process.start()
    ray.shutdown()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--batches', '-b',
        type=int,
        default=64,
        help='Number of batches. Each batch of games will be saved separately.'
    )
    parser.add_argument(
        '--batch_size', '-n',
        type=int,
        default=112,
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
    parser.add_argument(
        '--redis_address', '-r',
        type=str,
        default=None,
        help='Head node\'s redis address "host:port". If not set workers will be spawned on the host machine.'
    )
    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=4,
        help='The number of workers.'
    )

    flags, _ = parser.parse_known_args()

    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh = logging.FileHandler(os.path.join(flags.folder, 'selfplay.log'))
    fh.setFormatter(formatter)

    _logger.setLevel(logging.DEBUG)
    _logger.addHandler(fh)

    main(flags.generation,
         flags.iter,
         flags.tau,
         flags.folder,
         flags.batches,
         flags.batch_size,
         flags.workers,
         flags.redis_address)
