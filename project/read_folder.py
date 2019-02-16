import cv2
import os
import numpy as np
import matplotlib.pyplot as plt


class ReadFolder():
    def __init__(self, directory="project/images/", output_directory="project/output1/",  net_input_size=[42, 100, 100]):
        self.IM_DIR = directory  # directory_doc="project/documents/"
        self.OUTPUT_DIR = output_directory
        self.files = os.listdir(self.IM_DIR)
        self.number_of_samples = np.size(self.files)
        self.tiles_per_dim = []
        self.n_data_size = net_input_size  # here we are defining the training set size, this dimension is depend on the net input

        self.X = np.zeros((self.n_data_size[0], self.n_data_size[1], self.n_data_size[2]))

    def generate_net_input(self):
        show_figure = 0  # change this ver. to "1" if you would like to watch the pictures
        i = 0
        if np.shape(self.files)[0] < 6:
            self.tiles_per_dim = 2
        if np.shape(self.files)[0] > 20:
            self.tiles_per_dim = 5
        else:
            self.tiles_per_dim = 4

        for f in self.files:
            im = cv2.imread(self.IM_DIR+f)
            im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)

            if show_figure:
                plt.imshow(im, cmap='gray')
                plt.show()

            reshape_im = cv2.resize(im, dsize=(self.n_data_size[1], self.n_data_size[2]), interpolation=cv2.INTER_CUBIC)  # resize picture size for equal sizing

            self.X[i] = reshape_im
            i += 1

        return self.X, self.tiles_per_dim

    # def see_the_pic(self):
    #     for f in range()
