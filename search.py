import math
import numpy as np
import network
import connect4
import inference

np.set_printoptions(edgeitems=12, linewidth=120)
C_PUCT = 5


class Node:
    def __init__(self, p, action, state, player):
        self.N = 0    # total visit count
        self.W = 0    # total action value
        self.P = p    # prior probability of choosing

        self.state = state    # np.mtx, the board state
        self.player = player  # the last player
        self.action = action  # the action last player took

        self.children = []
        self.parent = None
        self.terminal = None
        self.depth = 0

    def max_puct_node(self):
        return max(self.children, key=lambda node: node.Q + node.U if node is not None else 0)

    def get_policy(self, tau):
        # fairly ugly... fill in the "missing" illegal moves ...
        legal_moves, visits, q_values = set(), [], []
        for node in self.children:
            legal_moves.add(node.action)
            visits.append(node.N)
            q_values.append(node.Q)

        for move in range(7):
            if move not in legal_moves:
                visits.insert(move, 0)
                q_values.insert(move, -1)

        visits = np.asarray(visits)**(1/tau)
        pi = visits / visits.sum()
        return pi, q_values

    @property
    def U(self):
        # the numerator is the: total visit of potential actions
        # sum([other_action.N for other_action in self.parent.children]) i.e. self.parent.N
        numerator = self.parent.N
        denominator = (1 + self.N)
        return C_PUCT * self.P * math.sqrt(numerator) / denominator

    @property
    def Q(self):
        return 0 if self.W == 0 else self.W / self.N

    def select(self):
        curr = self
        while any(curr.children):
            curr = curr.max_puct_node()
        return curr

    def expand(self, engine, board_state, prior_probabilities):
        if self.terminal is not None:
            return

        # children belongs to the other player
        player = 3 - self.player

        illegal_mask = engine.illegal_moves_mask(board_state)
        prior_probabilities[0][illegal_mask] = 0
        norm = np.linalg.norm(prior_probabilities[0])

        for move, p in enumerate(prior_probabilities[0] / norm):
            # illegal move
            if p < 1e-5:
                continue

            r, c, next_state = engine.move(move, board_state, player)

            child_node = Node(p, move, next_state, player)
            child_node.parent = self
            child_node.depth = self.depth + 1

            # mark the leaf, so if we were to select it, we propagate the real values
            # (maybe even skip neural evaluation if we select this)
            winner = engine.has_player_won(r, c, next_state, player)
            if winner is not None:
                child_node.terminal = 1

            if child_node.depth == 42 and winner is None:
                child_node.terminal = 0

            self.children.append(child_node)

    def backup(self, v):
        v = v[0][0]  # todo: tf.squeeze
        curr = self
        # override v, when it's obvious to avoid or choose this state
        if self.terminal is not None:
            v = self.terminal

        turn = 1
        while curr is not None:
            curr.N += 1
            curr.W += (v * turn)

            curr = curr.parent
            turn *= -1

    def __str__(self):
        return f'{self.state}, N:{self.N}, W:{self.W:.2f}, P:{self.P:.2f}, depth:{self.depth}'


class Mcts:
    def __init__(self, iter_count, net, game_engine):
        self.iter_count = iter_count
        self.net = net
        self.game_engine = game_engine

    def search(self, board_state, player, tau=0.01):
        # create root (next level starts the game, so root should be the other player)
        root = Node(0, 0, board_state, 3-player)
        root.depth = np.count_nonzero(board_state)
        root.N = -1  # root counted twice...

        for _ in range(self.iter_count):
            node = root

            leaf = node.select()
            p, v = self.net.predict(network.to_network_state(leaf.state))
            leaf.expand(self.game_engine, leaf.state, p)
            leaf.backup(v)

        return root.get_policy(tau)


def main():
    from collections import Counter
    import datetime
    import os

    engine = connect4.GameEngine()
    model = inference.FrozenModel(os.path.join('model', 'frozen_model.pb'))
    mcts = Mcts(800, model, engine)

    end_results = []

    def step(s, player, use_mcts=True, log=False):
        if use_mcts:
            pi, q_values = mcts.search(s, player, 0.01)
            a = np.random.choice(7, p=pi)
            q = q_values[a]
            r, c, s = engine.move(a, s, player)
        else:
            r, c, s, a = engine.do_random_move(s, player)
            q = None
        if log:
            print(f'player{player} -> move:{a}, q:{q}')
            print(s)
        end_result = engine.has_player_won(r, c, s, player)
        if end_result is not None:
            end_results.append(end_result)
            return end_result, s

        if end_result is None and np.count_nonzero(s) == 42:
            end_results.append(0)
            return 0, s

        return None, s

    for i in range(100):
        state = np.zeros((6, 7)).astype(int)
        winner = None
        while winner is None:
            winner, state = step(state, 1, use_mcts=True, log=False)
            if winner is None:
                winner, state = step(state, 2, use_mcts=False, log=False)
        print(f'{datetime.datetime.now()}: ({i}) game over: {winner}')
        print('-'*42)

    print(Counter(end_results))


if __name__ == "__main__":
    main()
