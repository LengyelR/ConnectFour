import os
import train
import network


def mkdir(*folder):
    cwd = os.getcwd()
    path = os.path.join(cwd, *folder)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def init_generation_zero():
    """
    creates the folder structure for gen-0, which contains the untrained, random model
    """
    current_gen = 'gen-0'
    model_folder_path = mkdir('model', current_gen)
    tf_folder_path = mkdir('model', current_gen, 'tf')

    weight_path = os.path.join(model_folder_path, 'weights.h5')
    keras_path = os.path.join(model_folder_path, 'keras_model.h5')
    frozen_path = os.path.join(model_folder_path, 'frozen_model.pb')
    tf_path = os.path.join(tf_folder_path, 'connect4')

    creator = network.Con4Zero(network.INPUT_SHAPE, network.ResidualBlock)
    neural = creator()
    neural.save_weights(weight_path)

    train.save(weight_path, keras_path, tf_path, frozen_path)


if __name__ == '__main__':
    init_generation_zero()
