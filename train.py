import glob
import os
import pickle
import random
from collections import defaultdict

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.python.platform import gfile
import numpy as np

import network


def _get_window_size(gen):
    if gen < 4:
        return gen
    elif gen < 35:
        return gen // 2 + 2
    else:
        return 20


def load_training_data(folder, generation):
    data = []

    _, current_gen_str = generation.split('-')
    curr_gen = int(current_gen_str)
    window_size = _get_window_size(curr_gen)

    for i in range(window_size+1):
        gen_idx = curr_gen - i
        search_pattern = os.path.join(folder, 'training', 'gen-' + str(gen_idx), '**', '*.pkl')
        print(search_pattern)
        for match in glob.iglob(search_pattern, recursive=True):
            with open(match, 'rb') as f:
                batch = pickle.load(f)
                data.extend(batch)
    print('combined length:', len(data))
    return data


def _sample_data(training_data, batch_size, steps):
    dedupe = defaultdict(list)
    x_data = []

    pi_arr = []
    q_arr = []

    for step_data in training_data:
        s = step_data[0]
        pi = step_data[1]
        q = step_data[2]
        # player = step_data[3]
        z = step_data[4]
        avg = (q + z)/2

        dedupe[s.tostring()].append((s, pi, avg))

    for step_data in dedupe.values():
        length = len(step_data)
        x_data.append(network.to_training_feature_planes(step_data[0][0]))
        pi_arr.append(sum(v[1] for v in step_data) / length)
        q_arr.append(sum(v[2] for v in step_data) / length)

    dataset_size = len(x_data)
    sample_size = min(batch_size*steps, dataset_size)
    idx = random.sample(range(dataset_size), sample_size)
    xs = np.array(x_data)[idx]
    ys = [np.array(pi_arr)[idx], np.array(q_arr)[idx]]
    return xs, ys


class CyclicLearningRate(tf.keras.callbacks.Callback):
    def __init__(self, start_lr, peak_lr, steps):
        """
        :param start_lr: cycle start. Overrides the optimizer's learning rate!
        :param peak_lr: cycle peak. Overrides the optimizer's learning rate!
        :param steps: must be equal to the number of training steps!
        """
        super().__init__()
        self.start_lr = start_lr
        self.peak_lr = peak_lr
        self.steps = steps

    @staticmethod
    def _draw_line(p0, p1):
        m = (p1[1] - p0[1]) / (p1[0] - p0[0])
        return lambda x: m * (x - p0[0]) + p0[1]

    def _increase(self, x):
        start = (0, self.start_lr)
        stop = (self.steps*0.45, self.peak_lr)
        line = self._draw_line(start, stop)
        return line(x)

    def _decrease(self, x):
        start = (self.steps*0.45, self.peak_lr)
        stop = (self.steps*0.9, self.start_lr)
        line = self._draw_line(start, stop)
        return line(x)

    def _1cycle(self, x):
        start = (self.steps*0.9, self.start_lr)
        stop = (self.steps, self.start_lr/200)
        line = self._draw_line(start, stop)
        return line(x)

    def on_batch_begin(self, batch, logs=None):
        if batch < self.steps*0.45:
            new_lr = self._increase(batch)
        elif batch < self.steps*0.9:
            new_lr = self._decrease(batch)
        else:
            new_lr = self._1cycle(batch)
        print(f' lr: {new_lr:.4f}')
        K.set_value(self.model.optimizer.lr, new_lr)


def train(data, previous_network_weights, batch_size, steps):
    model = network.Con4Zero(network.INPUT_SHAPE)()
    model.load_weights(previous_network_weights)
    model.compile(
        optimizer=tf.keras.optimizers.SGD(),
        loss=network.Con4Zero.loss()
    )

    xs, ys = _sample_data(data, batch_size, steps)
    cyclic_lr = CyclicLearningRate(0.02, steps)
    model.fit(
        xs, ys,
        epochs=2,
        batch_size=batch_size,
        callbacks=[cyclic_lr]
    )

    return model


def _save_keras_model(weight_src, destination):
    tf.keras.backend.set_learning_phase(0)
    creator = network.Con4Zero(network.INPUT_SHAPE)
    neural = creator()
    neural.load_weights(weight_src)
    tf.keras.models.save_model(neural, destination, overwrite=True)


def _save_keras_model_as_tf(keras_src, destination):
    tf.keras.backend.set_learning_phase(0)
    tf.keras.models.load_model(keras_src)
    saver = tf.train.Saver()
    sess = tf.keras.backend.get_session()
    saver.save(sess, destination)


def _freeze_graph(tf_model_src, destination, outputs):
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(tf_model_src + ".meta")
        saver.restore(sess, tf_model_src)

        graph = tf.get_default_graph().as_graph_def()
        frozen_graph = tf.graph_util.convert_variables_to_constants(sess, graph, output_node_names=outputs)

        with gfile.FastGFile(destination, 'wb') as f:
            f.write(frozen_graph.SerializeToString())


def save(weight_path, keras_model_path, tf_model_path, frozen_path):
    tf.reset_default_graph()

    _save_keras_model(weight_path, keras_model_path)
    _save_keras_model_as_tf(keras_model_path, tf_model_path)
    _freeze_graph(tf_model_path, frozen_path, ["policy_head/Softmax", "value_head/Tanh"])


def main(folder, current_gen, new_gen, batch_size, steps):
    import utils

    root = os.path.join(folder, 'model')

    model_folder_path = utils.mkdir(root, new_gen)
    tf_folder_path = utils.mkdir(root, new_gen, 'tf')

    prev_weights_path = os.path.join(root, current_gen, 'weights.h5')
    weight_path = os.path.join(model_folder_path, 'weights.h5')
    keras_path = os.path.join(model_folder_path, 'keras_model.h5')
    frozen_path = os.path.join(model_folder_path, 'frozen_model.pb')
    tf_path = os.path.join(tf_folder_path, 'connect4')

    training_data = load_training_data(folder, current_gen)
    trained_model = train(training_data, prev_weights_path, batch_size, steps)
    trained_model.save_weights(weight_path)

    save(weight_path, keras_path, tf_path, frozen_path)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--folder', '-o',
        type=str,
        default='',
        help='Root folder.'
    )
    parser.add_argument(
        '--prev_gen', '-pg',
        type=str,
        default='gen-0',
        help='The model will be loaded and trained on this generation\'s data.'
    )
    parser.add_argument(
        '--new_gen', '-ng',
        type=str,
        default='gen-1',
        help='The new, next generation. After the training is finished, the model is saved under this name.'
    )
    parser.add_argument(
        '--batch_size', '-b',
        type=int,
        default=128,
        help='Number of samples per gradient update.'
    )
    parser.add_argument(
        '--steps', '-n',
        type=int,
        default=4000,
        help='Number of steps in an epoch.'
    )

    flags, _ = parser.parse_known_args()
    main(flags.folder, flags.prev_gen, flags.new_gen, flags.batch_size, flags.steps)
