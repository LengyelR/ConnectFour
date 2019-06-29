import tensorflow as tf
import numpy as np


class ConvolutionalBlock:
    def __init__(self):
        self.conv = tf.keras.layers.Conv2D(64, (3, 3), padding='same')
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.Activation('relu')

    def __call__(self, x):
        x = self.conv(x)
        x = self.bn(x)
        out = self.relu(x)
        return out


class ResidualBlock:
    def __init__(self):
        self.conv = tf.keras.layers.Conv2D(64, (3, 3), padding='same')
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.Activation('relu')

    def __call__(self, x):
        residual = x

        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)

        out = self.conv(out)
        out = self.bn(out)
        out = self.relu(out)

        out = out + residual

        out = self.relu(out)
        return out


class PolicyHead:
    def __init__(self):
        self.flatten = tf.keras.layers.Flatten()
        self.bn = tf.keras.layers.BatchNormalization()
        self.conv = tf.keras.layers.Conv2D(2, (1, 1), padding='same')
        self.fc = tf.keras.layers.Dense(7)   # column size of the board (i.e. moves)
        self.relu = tf.keras.layers.Activation('relu')
        self.softmax = tf.keras.layers.Activation('softmax')

    def __call__(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.flatten(x)
        x = self.fc(x)
        return self.softmax(x)


class ValueHead:
    def __init__(self):
        self.flatten = tf.keras.layers.Flatten()
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.Activation('relu')
        self.tanh = tf.keras.layers.Activation('tanh')
        self.fc1 = tf.keras.layers.Dense(256)
        self.fc2 = tf.keras.layers.Dense(1)
        self.conv = tf.keras.layers.Conv2D(1, (1, 1), padding='same')
        self.softmax = tf.keras.layers.Activation('softmax')

    def __call__(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        return self.tanh(x)


class Con4Zero:
    def __init__(self, input_shape, res_block):
        self.block = res_block()
        self.policy_head = PolicyHead()
        self.value_head = ValueHead()
        self.conv = ConvolutionalBlock()
        self.input_shape = input_shape

    def _residual_tower(self, x, n):
        for _ in range(n):
            x = self.block(x)
        return x

    def __call__(self):
        board = tf.keras.layers.Input(shape=self.input_shape)

        x = self.conv(board)
        x = self._residual_tower(x, 2)  # todo: higher tower
        policy = self.policy_head(x)
        value = self.value_head(x)

        model = tf.keras.models.Model(inputs=[board], outputs=[policy, value])
        return model

    @staticmethod
    def loss():
        """
        p, v = f_theta(s)
        loss := (z - v)**2 - pi * log(p) + l2reg(theta)
        where
            p is the returned policy from the network
            v is the estimated outcome from the network
            z is the real outcome
            pi is MCTS outcome ("the improved p")
            theta is the weight parameters of the network

        :return: MSE + CROSS-ENTROPY + l2reg
        """

        def _loss(y_true, y_pred):
            mse = tf.keras.losses.mean_squared_error(y_true[1], y_pred[1])
            ce = tf.keras.losses.categorical_crossentropy(y_true[0], y_pred[0])
            return mse + ce  # todo: + K.square(weights)*c

        return _loss


if __name__ == "__main__":
    INPUT_SHAPE = (32, 32, 1)
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

    creator = Con4Zero(INPUT_SHAPE, ResidualBlock)
    neural = creator()
    run_model(neural)

    neural.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, ),
        loss=Con4Zero.loss(),  # todo: pass weights?
        metrics=[Con4Zero.loss()]
    )

    test_shape = (TEST, ) + INPUT_SHAPE
    train_shape = (TRAIN, ) + INPUT_SHAPE
    neural.fit(
        np.random.randn(*train_shape), create_data(TRAIN),
        epochs=3,
        steps_per_epoch=10,
        validation_data=(np.random.randn(*test_shape), create_data(TEST)),
        validation_freq=10
    )

    run_model(neural)
