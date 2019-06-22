import numpy as np


class Game:
    def __init__(self):
        self.columns = 7
        self.rows = 6
        self.current_player = 2
        self.move_count = 0

        self.board = np.zeros((self.rows, self.columns)).astype(int)

    def place(self, column):
        self.move_count += 1
        self.current_player = 2 if self.current_player == 1 else 1

        for row in range(self.rows):
            if self.board[row][column] == 0:  # found an empty cell
                self.board[row][column] = self.current_player

                if self.has_current_player_won(row, column):
                    print()
                    print(f'player {self.current_player} has won. move ({column})')
                    print(self.board)
                    print('-' * 16)
                    return self.current_player

                break
        else:
            self.move_count -= 1
            self.current_player = 2 if self.current_player == 1 else 1
            raise ValueError(f'Column ({column}) is full')

        # has not won, but was last move --> draw
        if self.move_count == self.columns * self.rows:
            print()
            print(f'draw')
            print(self.board)
            print('-' * 16)
            return 0

    def has_current_player_won(self, row_idx, col_idx):
        row = self.board[row_idx]
        column = self.board[:, col_idx]

        def _check_col(height, depth=0):
            if depth == 4:
                return True

            if column[height] == self.current_player:
                return _check_col(height - 1, depth + 1)
            else:
                return False

        def _check_row():
            counter = 0
            for cell in row:
                if cell != self.current_player:
                    counter = 0
                else:
                    counter += 1
                    if counter == 4:
                        return True
            return False

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
                        return True
                r += 1
                c += 1

            return False

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
                        return True
                r += 1
                c -= 1

            return False

        return _check_col(row_idx)\
               or _check_row() \
               or _check_anti_diag(row_idx, col_idx) \
               or _check_main_diag(row_idx,col_idx)


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
    for _ in range(10):
        new_game = Game()
        state = None
        while state is None:
            try:
                state = new_game.place(random.randint(0, 6))
            except ValueError:
                pass
