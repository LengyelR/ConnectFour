import math
import numpy as np
import network
import connect4

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
        return max(self.children, key=lambda node: node.Q + node.U if node.action != -1 else -1)

    def get_policy(self, tau):
        """
        :param tau: temperature parameter
        :return: probability distribution from the exponentiated visit counts
        """
        visits = [node.N for node in self.children]
        visits = np.asarray(visits)**(1/tau)
        pi = visits / visits.sum()
        return pi

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

        # children belong to the other player
        player = 3 - self.player

        illegal_mask = engine.illegal_moves_mask(board_state)
        prior_probabilities[0][illegal_mask] = 0
        normed_priors = prior_probabilities[0] / sum(prior_probabilities[0])
        normed_priors[illegal_mask] = -1

        for move, p in enumerate(normed_priors):
            # illegal move
            if p < 0:
                illegal_move = Node(0, -1,  None, -1)
                illegal_move.parent = self
                illegal_move.depth = self.depth + 1
                self.children.append(illegal_move)
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
        v = v[0][0]
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


class MctsBase:
    def __init__(self, iter_count, net, game_engine, name='mcts'):
        self.iter_count = iter_count
        self.net = net
        self.game_engine = game_engine
        self.name = name

    def search(self, root, tau=1.0):
        for _ in range(self.iter_count):
            node = root

            leaf = node.select()
            p, v = self._predict(network.to_network_feature_planes(leaf.state))
            leaf.expand(self.game_engine, leaf.state, p)
            leaf.backup(v)

        pi = root.get_policy(tau)
        a = np.random.choice(7, p=pi)
        best_node = root.children[a]

        return pi, best_node

    def _predict(self, s):
        raise NotImplementedError


class Mcts(MctsBase):
    def _predict(self, s):
        return self.net.predict(s)


class MctsRnd(MctsBase):
    """
    Uses a uniform random distribution instead of a neural network.
    This class should be similar to a mcts guided by a gen-0 network, with "pure" exploration.
    (But with lower latency.)
    """
    def __init__(self, iter_count, game_engine):
        super().__init__(iter_count, None, game_engine)

    def _predict(self, s):
        p = np.random.uniform(0, 1, 7)
        p = p / p.sum()
        p = np.asarray([p])

        v = np.random.uniform(-1, 1, (1, 1, 1))
        return p, v[0]


def main():
    from collections import Counter
    import datetime
    import time

    engine = connect4.GameEngine()
    mcts = MctsRnd(800, engine)
    np.random.seed(0)

    def play(tau, log=True):
        state = engine.empty_board()
        root = Node(0, 0, state, 2)
        while True:
            pi, best_node = mcts.search(root, tau)
            if log:
                print('  '*best_node.action + ' v')
                engine.show_board(best_node.state)
                print()

            if best_node.terminal is not None:
                break
            root = best_node
        winner = best_node.player if best_node.terminal == 1 else 0
        return winner, best_node.state

    start = time.time()
    end_results = []
    n = 10
    for i in range(n):
        w, s = play(1.0)
        print(f'{datetime.datetime.now()}: ({i}) game over: {w}')
        print('-'*42)
        end_results.append(w)
    print(time.time()-start, f'seconds for {n} games.')
    print(Counter(end_results))


if __name__ == "__main__":
    main()
