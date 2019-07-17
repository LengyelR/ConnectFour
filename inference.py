import tensorflow as tf
from tensorflow.python.platform import gfile
import network


def save_keras_model(weight_src, destination):
    tf.keras.backend.set_learning_phase(0)
    creator = network.Con4Zero(network.INPUT_SHAPE, network.ResidualBlock)
    neural = creator()
    neural.load_weights(weight_src)
    tf.keras.models.save_model(neural, destination, overwrite=True)


def save_keras_model_as_tf(keras_src, destination):
    tf.keras.backend.set_learning_phase(0)
    tf.keras.models.load_model(keras_src)
    saver = tf.train.Saver()
    sess = tf.keras.backend.get_session()
    saver.save(sess, destination)


def freeze_graph(tf_model_src, destination, outputs):
    with tf.Session() as sess:

        saver = tf.train.import_meta_graph(tf_model_src + ".meta")
        saver.restore(sess, tf_model_src)

        graph = tf.get_default_graph().as_graph_def()
        frozen_graph = tf.graph_util.convert_variables_to_constants(sess, graph, output_node_names=outputs)

        with gfile.FastGFile(destination, 'wb') as f:
            f.write(frozen_graph.SerializeToString())


def save(weight_path, keras_model_path, tf_model_path, frozen_path):
    save_keras_model(weight_path, keras_model_path)
    save_keras_model_as_tf(keras_model_path, tf_model_path)
    freeze_graph(tf_model_path, frozen_path, ["policy_head/Softmax", "value_head/Tanh"])


class FrozenModel:
    def __init__(self, frozen_path):
        self.graph = tf.Graph()

        with gfile.FastGFile(frozen_path, 'rb') as f:
            frozen_graph = tf.GraphDef()
            frozen_graph.ParseFromString(f.read())

        with self.graph.as_default():
            tf.import_graph_def(frozen_graph, name='')

        self.sess = tf.Session(graph=self.graph)
        self.state = self.sess.graph.get_tensor_by_name('board_state_input:0')
        self.policy = self.sess.graph.get_tensor_by_name('policy_head/Softmax:0')
        self.value = self.sess.graph.get_tensor_by_name('value_head/Tanh:0')

    def run(self, state):
        res = self.sess.run([self.policy, self.value], feed_dict={self.state: state})
        return res


if __name__ == "__main__":
    import os
    import time
    import numpy as np

    def _mkdir(*folder):
        cwd = os.getcwd()
        path = os.path.join(cwd, *folder)
        if not os.path.exists(path):
            os.makedirs(path)
        return path


    tf_path = _mkdir('model', 'tf')
    keras_path = os.path.join('model', 'keras_model.h5')
    model_path = os.path.join('model', 'frozen_model.pb')

    save(network.WEIGHT_PATH, keras_path, tf_path, model_path)
    model = FrozenModel(model_path)

    for _ in range(10):
        s = np.random.randn(1, *network.INPUT_SHAPE)
        start = time.time()
        p, v = model.run(s)
        print(time.time()-start, p, v)

    print('-'*64)
    nn = network.Con4Zero(network.INPUT_SHAPE, network.ResidualBlock)()
    nn.load_weights(network.WEIGHT_PATH)
    for _ in range(10):
        s = np.random.randn(1, *network.INPUT_SHAPE)
        start = time.time()
        p, v = nn.predict(s)
        print(time.time()-start, p, v)
