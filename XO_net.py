import numpy as np
import random as r
from tqdm import tqdm
from copy import deepcopy



def convert(x: int) -> list:
    return [x // 3, x % 3]

def play_game(net, eps: float, self_play=True, your_turn=0) -> 'Game':
    game = Game()
    players_moves = [1, -1]
    moves_list = [1, 0]
    which_move = 0
    if not self_play:
        print(game)
    while True:
        game.boards_in_game.append(deepcopy(game.board))
        if self_play and eps <= r.randint(1, 100):
            move = r.randint(0, 8)
        elif not self_play and your_turn == which_move:
            move = int(input())
        else:
            move = net.predict(np.array(game.board).reshape(1, 3, 3))
        game.moves_in_game.append(move)
        conv = convert(move)
        if game.board[conv[0]][conv[1]] != 0:
            game.game_result[moves_list[which_move]] = 1
            game.was_breaked = True
            break
        game.move(move_to=move, move_type=players_moves[which_move])
        if game.check_win():
            game.game_result[which_move] = 1
            break
        if game.check_draw():
            game.game_result = [0, 0]
            break
        which_move = moves_list[which_move]
        if not self_play:
            print(game)
    return game

class Game:
    def __init__(self) -> None:
        self.board = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        self.game_result = [0, 0]
        self.boards_in_game = []
        self.moves_in_game = []
        self.was_breaked = False
        self.weights = []

    def move(self, move_type, move_to):
        move_to = convert(move_to)
        if self.board[move_to[0]][move_to[1]] == 0:
            self.board[move_to[0]][move_to[1]] = move_type

    def check_win(self):
        for i in range(3):
            if self.board[i][0] == self.board[i][1] == self.board[i][2] != 0:
                return True
        for i in range(3):
            if self.board[0][i] == self.board[1][i] == self.board[2][i] != 0:
                return True
        if (self.board[0][0] == self.board[1][1] == self.board[2][2] != 0) or (self.board[0][2] == self.board[1][1] == self.board[2][0] != 0):
            return True
        return False

    def check_draw(self):
        for row in self.board:
            if 0 in row:
                return False
        return True

    def __repr__(self) -> str:
        buf = ""
        for i in range(3):
            buf += "+ - + - + - +\n|"
            for j in range(3):
                if self.board[i][j] == 1:
                    x = "X"
                elif self.board[i][j] == -1:
                    x = "O"
                elif self.board[i][j] == 0:
                    x = " "
                buf += f" {x} |"
            buf += "\n"
        buf += "+ - + - + - +\n"
        return buf
    
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)
    
class Nn():
    def __init__(self, neurons: list[int]) -> None:
        self.weights = [np.random.uniform(-1, 1, (neurons[i + 1], neurons[i])) for i in range(len(neurons) - 1)]
        self.biases = [np.random.uniform(-0.01, 0.01, (neurons[i + 1],)) for i in range(len(neurons) - 1)]

    def __call__(self, X: np.ndarray) -> list[np.ndarray]:
        # X: [batch_size, 9]
        Y = [X]

        for w, b in zip(self.weights, self.biases):
            # x: [batch_size, i]
            # w: [j, i]
            # y: [batch_size, j]

            x = Y[-1]
            y = sigmoid(x @ w.T + b)
            Y.append(y)
        Y[0] = np.array(Y[0])
        return Y

    def predict(self, x: list[list[int]] | np.ndarray) -> np.ndarray:
        return self(np.array(x).flatten())[-1]

    def train(self, boards: list[list[int]], labels: list[int], values: list[int], *, lr): 
        # Given i = layer index, Y[i]: [batch_size, j]
        
        batch_size = len(boards)
        assert batch_size == len(labels) == len(values), "batch_size mismatch"

        Y = self(np.array(boards).reshape(batch_size, -1))
        assert len(Y) == len(self.weights) + 1, "len(Y) mismatch"

        # errors: [batch_size, 9]
        errors = np.zeros(Y[0][-1].shape)[np.newaxis].repeat(batch_size, axis=0)
        for idx, (label, value) in enumerate(zip(labels, values)):
            assert -1 <= value <= 1, "value out of bounds"
            amplitude = np.abs(value)
            direction = max(0, value / amplitude)
            errors[idx, label] = (direction - Y[-1][idx, label]) * amplitude
        loss = np.mean(np.abs(errors))

        for i in reversed(range(len(self.weights))):
            gradients = errors * sigmoid_derivative(Y[i + 1]) # [batch_size, j]
            errors = gradients @ self.weights[i] # [batch_size, i]
            delta = np.einsum("bi, bj -> bij", gradients, Y[i]) # [batch_size, i, j]
            self.weights[i] += delta.sum(axis=0) * lr
            self.biases[i] += gradients.sum(axis=0) * lr
        
        return loss
    
net = Nn([9, 81, 729, 81, 9])
lr = 0.005

batch_size = 16
total_steps = 500000
logging_steps = 1000

history = []

loss_accum = 0.0
eps_accum = 0.0
for step in (pbar := tqdm(range(total_steps))):
    # eps - chance that the move was done by Nn.
    # not by random.randint().
    eps = (step / total_steps) * 0.6 + 0.2

    boards = []
    labels = []
    values = []

    done = False
    while len(boards) < batch_size:
        game = play_game(net, eps)

        # If the game ended by invalid move, we skip
        # all its moves and teach it on the last invalid move.
        if game.was_breaked:
            boards.append(game.boards_in_game[-1])
            labels.append(game.moves_in_game[-1])
            values.append(-1)
            continue

        for idx, (board, move) in enumerate(zip(game.boards_in_game, game.moves_in_game)):
            player = idx % 2

            if game.game_result[player]:
                # If the game ended by first Nn won.
                value = 0.5
            elif sum(game.game_result) == 0:
                # If it's draw.
                value = -0.25
            else:
                # If the game ended by first Nn lose.
                value = -0.5

            boards.append(board)
            labels.append(move)
            values.append(value)

    boards = boards[:batch_size]
    labels = labels[:batch_size]
    values = values[:batch_size]

    loss = net.train(boards, labels, values, lr=lr)

    loss_accum += loss
    eps_accum += eps
    
    if step > 0 and step % logging_steps == 0:
        record = dict(step=step, loss=loss_accum / logging_steps, eps=eps_accum / logging_steps)
        pbar.set_postfix(record)
        history.append(record)
        loss_accum = eps_accum = 0
np.savez("Weights_saved", x=net.weights[0], y=net.weights[1], z=net.weights[2], d=net.weights[3])
