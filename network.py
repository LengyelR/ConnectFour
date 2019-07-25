import tensorflow as tf
import numpy as np
np.set_printoptions(edgeitems=12, linewidth=120)

INPUT_SHAPE = (2, 6, 7)


class ConvolutionalBlock:
    def __init__(self, filters, shape, reg, name):
        self.conv = tf.keras.layers.Conv2D(filters, shape,
                                           padding='same',
                                           kernel_regularizer=reg,
                                           name=f'{name}_conv_block')
        self.bn = tf.keras.layers.BatchNormalization(name=f'{name}_bn_block')
        self.relu = tf.keras.layers.Activation('relu', name=f'{name}_relu_block')

    def __call__(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)


class ResidualBlock:
    def __init__(self, idx, reg):
        self.conv1 = ConvolutionalBlock(64, (3, 3), reg, f'resi1_{idx}')
        self.conv2 = ConvolutionalBlock(64, (3, 3), reg, f'resi2_{idx}')
        self.relu = tf.keras.layers.Activation('relu', name=f'resi_relu_{idx}')

    def __call__(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)

        out = tf.keras.layers.add([out, residual])
        out = self.relu(out)
        return out


class PolicyHead:
    def __init__(self, reg):
        self.conv = ConvolutionalBlock(16, (1, 1), reg, 'policy')
        self.flatten = tf.keras.layers.Flatten()
        self.fc = tf.keras.layers.Dense(7, kernel_regularizer=reg)   # column size of the board (i.e. moves)
        self.softmax = tf.keras.layers.Activation('softmax', name="policy_head")

    def __call__(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        x = self.fc(x)
        return self.softmax(x)


class ValueHead:
    def __init__(self, reg):
        self.conv = ConvolutionalBlock(8, (1, 1), reg, 'value')
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(256, kernel_regularizer=reg)
        self.relu = tf.keras.layers.Activation('relu', name='value_relu')
        self.fc2 = tf.keras.layers.Dense(1, kernel_regularizer=reg)
        self.tanh = tf.keras.layers.Activation('tanh', name='value_head')

    def __call__(self, x):
        x = self.conv(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        return self.tanh(x)


class Con4Zero:
    def __init__(self, input_shape):
        l2 = tf.keras.regularizers.l2(0.01)

        self.policy_head = PolicyHead(l2)
        self.value_head = ValueHead(l2)
        self.conv = ConvolutionalBlock(64, (3, 3), l2, 'first')
        self.input_shape = input_shape

    @staticmethod
    def _residual_tower(x, reg=tf.keras.regularizers.l2(0.01),  n=20):
        for i in range(n):
            x = ResidualBlock(i, reg)(x)
        return x

    def __call__(self):
        board = tf.keras.layers.Input(shape=self.input_shape, name="board_state_input")

        x = self.conv(board)
        x = self._residual_tower(x)
        policy = self.policy_head(x)
        value = self.value_head(x)

        model = tf.keras.models.Model(inputs=[board], outputs=[policy, value])
        return model

    @staticmethod
    def loss():
        """
        p, v = f_theta(s)
        loss := (z - v)**2 - pi * log(p)
        where
            p is the returned policy from the network
            v is the estimated outcome from the network
            z is the real outcome
            pi is MCTS outcome ("the improved p")

        :return: MSE + CROSS-ENTROPY
        """

        def _loss(y_true, y_pred):
            mse = tf.keras.losses.mean_squared_error(y_true[1], y_pred[1])
            ce = tf.keras.losses.categorical_crossentropy(y_true[0], y_pred[0])
            return mse + ce

        return _loss


def to_network_feature_planes(board):
    player1 = (board == 1).astype(int)
    player2 = (board == 2).astype(int)
    return np.asarray([[player1, player2]])


def to_training_feature_planes(board):
    player1 = (board == 1).astype(int)
    player2 = (board == 2).astype(int)
    return np.asarray([player1, player2])


if __name__ == "__main__":
    TRAIN = 500
    TEST = 100

    def run_model(m):
        t1 = (1,) + INPUT_SHAPE
        t2 = (10,) + INPUT_SHAPE
        print(m.predict(np.random.randn(*t1)))
        print()
        print(m.predict(np.random.randn(*t2)))


    def create_data(n):
        pi = np.zeros(7)
        pi[0] = 1.0
        return [np.tile(pi, (n, 1)), np.ones(n)]

    creator = Con4Zero(INPUT_SHAPE)
    neural = creator()
    print(neural.summary())

    sess = tf.keras.backend.get_session()
    writer = tf.summary.FileWriter("graph_output", sess.graph)
    writer.close()

    run_model(neural)

    neural.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, ),
        loss=Con4Zero.loss(),
        metrics=[Con4Zero.loss()]
    )

    test_shape = (TEST, ) + INPUT_SHAPE
    train_shape = (TRAIN, ) + INPUT_SHAPE
    neural.fit(
        np.random.randn(*train_shape), create_data(TRAIN),
        epochs=5,
        steps_per_epoch=10,
        validation_data=(np.random.randn(*test_shape), create_data(TEST)),
        validation_freq=10
    )
