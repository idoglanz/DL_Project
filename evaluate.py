import read_folder
from keras.models import load_model
import Recovery_network
import shredder_public as shred
import numpy as np


def predict(images, n_crops):

    model = load_model('model-in_process.h5')

    output = model.predict(images)
    labels = Recovery_network.parse_output(output[0, :, :], n_crops)

    return labels


def evaluate(file_dir='example/', crop_size=40):

    read_pics = read_folder.ReadFolder(directory=file_dir, crop_size=crop_size)

    images, n_crops = read_pics.generate_net_input()

    Y = predict(images, n_crops)

    return Y


data_pic = shred.Data_shredder(directory="image1/",
                               output_directory="output1/",
                               num_of_duplication=1,
                               net_input_size=[30, 40, 40])


tiles = 4
x, y = data_pic.generate_data(tiles_per_dim=[tiles], save_crops=1)

y_true = Recovery_network.parse_output(y[0, :, :], int(tiles**2 + tiles))

y_pred = evaluate(file_dir='output1/', crop_size=40)

x = x[:, :, :, :, np.newaxis]
# Recovery_network.arrange_image(y_pred, x[0, :, :, :, :], t=tiles, pixels=40, size_wo_pad=int(tiles**2 + tiles), n='evaluate_test')
