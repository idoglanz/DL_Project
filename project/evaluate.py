import os
import cv2
import numpy as np
from project import Ido_net
from project import read_folder
from keras.models import load_model


def predict(images):
    labels = []

    model = Ido_net.define_model()
    model.load_weights("Desktop/weights.hdf5")  # TODO: change directory
    # OR:
    # model = load_model('model.h5')

    model.predict(images)
    labels = Ido_net.parse_output()

    return labels


def evaluate(file_dir='example/'):
    read_pics = read_folder(directory=file_dir)
    images, tiles_per_dim = read_pics.generate_net_input()

    Y = predict(images, tiles_per_dim)

    return Y


