import numpy as np


class Game:
    def __init__(self):
        self.columns = 7
        self.rows = 6
        self.current_player = 2
        self.move_count = 0

        self.board = np.zeros((self.rows, self.columns)).astype(int)

    def place(self, column):
        """
        Returns the final position of the player's piece (i.e. where it landed)
        """
        self.move_count += 1
        self.current_player = 2 if self.current_player == 1 else 1

        for row in range(self.rows):
            if self.board[row][column] == 0:  # found an empty cell
                self.board[row][column] = self.current_player
                return row, column
        else:
            self.move_count -= 1
            self.current_player = 2 if self.current_player == 1 else 1
            raise ValueError(f'Column ({column}) is full')

    def is_game_over(self, row_idx, col_idx):
        """
        :param row_idx: position of last piece
        :param col_idx: position of last piece
        If game has ended it returns the winner's id (self.current_player) or 0 if it's a draw,
        otherwise it returns None
        """
        row = self.board[row_idx]
        column = self.board[:, col_idx]

        def _check_col(height, depth=0):
            if depth == 4:
                return self.current_player

            if column[height] == self.current_player:
                return _check_col(height - 1, depth + 1)
            else:
                return None

        def _check_row():
            counter = 0
            for cell in row:
                if cell != self.current_player:
                    counter = 0
                else:
                    counter += 1
                    if counter == 4:
                        return self.current_player
            return None

        def _check_anti_diag(r, c):
            # go to left upper side
            while r != 0 and c != 0:
                r -= 1
                c -= 1

            # descend and count
            counter = 0
            while r != (self.rows - 1) and c != (self.columns - 1):
                if self.board[r, c] != self.current_player:
                    counter = 0
                else:
                    counter += 1
                    if counter == 4:
                        return self.current_player
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
                if self.board[r, c] != self.current_player:
                    counter = 0
                else:
                    counter += 1
                    if counter == 4:
                        return self.current_player
                r += 1
                c -= 1

            return None

        if self.move_count == 42:
            return 0

        return _check_col(row_idx)\
               or _check_row() \
               or _check_anti_diag(row_idx, col_idx) \
               or _check_main_diag(row_idx, col_idx)


def example_games():
    game = Game()
    for _ in range(3):
        game.place(1)
        game.place(2)
    game.place(1)

    game2 = Game()
    game2.place(0)
    game2.place(1)
    game2.place(0)
    game2.place(2)
    game2.place(0)
    game2.place(3)
    game2.place(1)
    game2.place(4)

    game3 = Game()
    game3.place(0)
    game3.place(1)

    game3.place(1)
    game3.place(2)

    game3.place(3)
    game3.place(2)

    game3.place(2)
    game3.place(3)

    game3.place(4)
    game3.place(3)

    game3.place(3)


if __name__ == "__main__":
    import random
    import time
    from collections import Counter

    results = []

    start = time.time()
    for _ in range(1000):
        new_game = Game()
        state = None
        while state is None:
            try:
                col = random.randint(0, 6)
                r, c = new_game.place(col)
                state = new_game.is_game_over(r, c)
            except ValueError:
                pass
        results.append(state)
    print(time.time()-start, 's')
    print(Counter(results))
