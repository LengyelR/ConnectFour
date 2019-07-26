import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np
import glob
import os
import pickle
import network


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


def load_training_data(generation):
    data = []
    search_pattern = os.path.join('training', generation, '**', '*.pkl')
    for match in glob.iglob(search_pattern, recursive=True):
        with open(match, 'rb') as f:
            batch = pickle.load(f)
            data.extend(batch)
    return data


def _format_data(training_data):
    x_data = []

    pi_arr = []
    q_arr = []

    for step_data in training_data:
        s = step_data[0]
        pi = step_data[1]
        q = step_data[2]
        z = step_data[3]
        avg = (q + z)/2

        x_data.append(network.to_training_feature_planes(s))
        pi_arr.append(pi)
        q_arr.append(avg)

    xs = np.asarray(x_data)
    ys = [pi_arr, q_arr]
    return xs, ys


def train(data, previous_network_weights):
    xs, ys = _format_data(data)

    creator = network.Con4Zero(network.INPUT_SHAPE)
    neural = creator()
    neural.load_weights(previous_network_weights)
    neural.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4, ),
        loss=network.Con4Zero.loss(),
        metrics=[network.Con4Zero.loss()]
    )

    neural.fit(
        xs, ys,
        epochs=2,
        steps_per_epoch=1000
    )

    return neural


def main(current_gen, new_gen):
    import utils

    model_folder_path = utils.mkdir('model', new_gen)
    tf_folder_path = utils.mkdir('model', new_gen, 'tf')

    prev_weights_path = os.path.join('model', current_gen, 'weights.h5')
    weight_path = os.path.join(model_folder_path, 'weights.h5')
    keras_path = os.path.join(model_folder_path, 'keras_model.h5')
    frozen_path = os.path.join(model_folder_path, 'frozen_model.pb')
    tf_path = os.path.join(tf_folder_path, 'connect4')

    training_data = load_training_data(current_gen)
    trained_model = train(training_data, prev_weights_path)
    trained_model.save_weights(weight_path)

    save(weight_path, keras_path, tf_path, frozen_path)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
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
        help='The new, next generation. After the training is finished, the model is saved here.'
    )

    flags, _ = parser.parse_known_args()
    main(flags.prev_gen, flags.new_gen)
