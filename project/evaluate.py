import os
import cv2
import numpy as np
from project import Ido_net
import Recovery
from project import read_folder
from keras.models import load_model


def predict(images, n_crops):
    labels = []

    model = Ido_net.define_model()

    model = load_model('Recovery_rev1.h5')

    # OR:
    # model = Ido_net.define_model()
    # model.load_weights("Desktop/weights.hdf5")  # TODO: change directory

    output = model.predict(images)
    labels = Ido_net.parse_output(output, n_crops)

    return labels


def evaluate(file_dir='example/'):
    t_max = 4  # max number of cuts supported (hence max of 6^2 crops + 6 OOD = 42)
    crop_size = 50  # size of each crop ("pixels")
    max_crops = t_max ** 2 + t_max
    read_pics = read_folder.ReadFolder(directory=file_dir)
    images, n_crops = read_pics.generate_net_input(net_input_size=[int(max_crops), crop_size, crop_size])                               net_input_size=


    Y = predict(images, n_crops)

    return Y


evaluate(file_dir='output1/')
