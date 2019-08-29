import os
import train
import network


def mkdir(*folder):
    cwd = os.getcwd()
    path = os.path.join(cwd, *folder)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def init_generation_zero(folder=''):
    """
    creates the folder structure for gen-0, which contains the untrained, random model
    """
    current_gen = 'gen-0'
    root = os.path.join(folder, 'model')
    model_folder_path = mkdir(root, current_gen)
    tf_folder_path = mkdir(root, current_gen, 'tf')

    weight_path = os.path.join(model_folder_path, 'weights.h5')
    keras_path = os.path.join(model_folder_path, 'keras_model.h5')
    frozen_path = os.path.join(model_folder_path, 'frozen_model.pb')
    tf_path = os.path.join(tf_folder_path, 'connect4')

    creator = network.Con4Zero(network.INPUT_SHAPE)
    neural = creator()
    neural.save_weights(weight_path)

    train.save(weight_path, keras_path, tf_path, frozen_path)


def try_init_colorama():
    """
    Enables ANSI Escape codes under MS Windows.
    (Doesn't change anything on Linux.)
    """
    import importlib
    colorama_spec = importlib.util.find_spec("colorama")
    if colorama_spec is not None:
        from colorama import init
        init()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--folder', '-o',
        type=str,
        default='',
        help='Root folder.'
    )

    flags, _ = parser.parse_known_args()
    init_generation_zero(flags.folder)
