import numpy as np
import copy


class GameEngine:
    def __init__(self):
        self.columns = 7
        self.rows = 6
        self.move_count = 0

    def empty_board(self):
        return np.zeros((self.rows, self.columns)).astype(int)

    def move(self, action, state, player):
        # todo: maybe not all the time...?
        state = copy.copy(state)

        for row in range(self.rows):
            if state[row][action] == 0:  # found an empty cell
                state[row][action] = player
                self.move_count += 1
                return row, action, state
        else:
            return None, None, None

    def do_random_move(self, state, player):
        """
        :param state: board as numpy mtx
            state[0] is the bottom of the board
            state[self.rows-1] is the top of the board
        :param player: current player (1 or 2)
        :return: row, column, the new state, the executed move
        """
        possible_moves = np.where(state[self.rows-1] == 0)[0]
        if len(possible_moves) == 0:
            return None, None, None, None

        move = np.random.choice(possible_moves)
        r, c, s = self.move(move, state, player)
        return r, c, s, move

    def is_game_over(self, row_idx, col_idx, state, current_player):
        """
         :param row_idx: position of last piece
         :param col_idx: position of last piece
         :param state: board as numpy mtx
            state[0] is the bottom of the board
            state[self.rows-1] is the top of the board
         :param current_player: current player (1 or 2)

         If the game has ended it returns the winner's id (current_player) or 0 if it's a draw,
         otherwise it returns None
         """
        row = state[row_idx]
        column = state[:, col_idx]

        def _check_col(height, depth=0):
            if depth == 4:
                return current_player

            if column[height] == current_player:
                return _check_col(height - 1, depth + 1)
            else:
                return None

        def _check_row():
            counter = 0
            for cell in row:
                if cell != current_player:
                    counter = 0
                else:
                    counter += 1
                    if counter == 4:
                        return current_player
            return None

        def _check_anti_diag(r, c):
            # go to left upper side
            while r != 0 and c != 0:
                r -= 1
                c -= 1

            # descend and count
            counter = 0
            while r != (self.rows - 1) and c != (self.columns - 1):
                if state[r, c] != current_player:
                    counter = 0
                else:
                    counter += 1
                    if counter == 4:
                        return current_player
                r += 1
                c += 1

            return None

        def _check_main_diag(r, c):
            # go to right upper side
            while r != 0 and c != 6:
                r -= 1
                c += 1

            # descend and count
            counter = 0
            while r != (self.rows - 1) and c != 0:
                if state[r, c] != current_player:
                    counter = 0
                else:
                    counter += 1
                    if counter == 4:
                        return current_player
                r += 1
                c -= 1

            return None

        # todo: user needs to be very careful...
        if self.move_count == 42:
            return 0

        return _check_col(row_idx)\
               or _check_row() \
               or _check_anti_diag(row_idx, col_idx) \
               or _check_main_diag(row_idx, col_idx)


def _example_games():
    game = GameEngine()
    board = game.empty_board()
    for _ in range(3):
        _, _, board = game.move(1, board, 1)
        _, _, board = game.move(2, board, 2)
    _, _, board = game.move(1, board, 1)

    game2 = GameEngine()
    board = game.empty_board()
    _, _, board = game2.move(0, board, 1)
    _, _, board = game2.move(1, board, 2)
    _, _, board = game2.move(0, board, 1)
    _, _, board = game2.move(2, board, 2)
    _, _, board = game2.move(0, board, 1)
    _, _, board = game2.move(3, board, 2)
    _, _, board = game2.move(1, board, 1)
    _, _, board = game2.move(4, board, 2)

    game3 = GameEngine()
    board = game.empty_board()
    _, _, board = game3.move(0, board, 1)
    _, _, board = game3.move(1, board, 2)

    _, _, board = game3.move(1, board, 1)
    _, _, board = game3.move(2, board, 2)

    _, _, board = game3.move(3, board, 1)
    _, _, board = game3.move(2, board, 2)

    _, _, board = game3.move(2, board, 1)
    _, _, board = game3.move(3, board, 2)

    _, _, board = game3.move(4, board, 1)
    _, _, board = game3.move(3, board, 2)

    _, _, board = game3.move(3, board, 1)


def _test():
    import time
    from collections import Counter

    results = []

    start = time.time()
    for _ in range(1000):
        player = 2
        engine = GameEngine()
        state = engine.empty_board()
        winner = None

        while winner is None:
            player = 3 - player
            r, c, next_state, move = engine.do_random_move(state, player)
            winner = engine.is_game_over(r, c, next_state, player)
            state = next_state

        results.append(winner)
    print(time.time() - start, 's')
    print(Counter(results))


if __name__ == "__main__":
    _example_games()
    _test()
