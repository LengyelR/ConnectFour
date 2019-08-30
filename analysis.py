import os
import glob
import pickle
import numpy as np
import connect4
from collections import Counter

# todo: no need for the class... doesn't hold state anymore
_engine = connect4.GameEngine()

np.set_printoptions(
    edgeitems=12,
    linewidth=120,
    formatter={'float': lambda x: f"{x:0.3f}"})


class Step:
    def __init__(self, step):
        self.board = step[0]
        self.pi = step[1]
        self.q = step[2]
        self.player = step[3]
        self.z = step[4]

        self.is_terminal = self.q == 0 or self.q == 1 or self.q == -1

    def __hash__(self):
        return hash(self.board.tostring())

    def __eq__(self, other):
        return np.allclose(self.board, other.board)

    def __str__(self):
        return str(self.board)


class GameCollection:
    def __init__(self, steps):
        self.counter = Counter(steps)
        self.iterable = iter(steps)
        self.terminal_steps = [step for step in steps if step.is_terminal]
        self.num_steps = len(steps)

    def __len__(self):
        return len(self.terminal_steps)

    def __iter__(self):
        return self

    def __next__(self):
        game = []
        step = next(self.iterable)
        while not step.is_terminal:
            game.append(step)
            step = next(self.iterable)
        game.append(step)
        return game


def load_data(folder):
    data = []
    search_pattern = os.path.join(folder, '**', '*.pkl')
    for match in glob.iglob(search_pattern, recursive=True):
        with open(match, 'rb') as f:
            batch = pickle.load(f)
            data.extend(batch)
    return data


def print_stats(steps, top=5):
    games = GameCollection(steps)
    print('steps', games.num_steps)
    print('games', len(games))
    print(f'most common states:')
    for d in games.counter.most_common(top):
        _engine.show_board(d[0].board)
        print('freq', d[1])
        print()
    print('='*32)

    for _ in range(3):
        game = next(games)
        for step in game:
            _engine.show_board(step.board)
            print(step.pi)
            print(f'player {step.player}\tq {step.q:0.3f} z {step.z}')
            print()
        print('-'*32)


def main():
    path = os.path.join('training', 'gen-0')
    data = load_data(path)
    print_stats([Step(step) for step in data], 10)


if __name__ == '__main__':
    main()
