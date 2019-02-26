import cv2
import os
import numpy as np
import matplotlib.pyplot as plt


class ReadFolder():
    def __init__(self, directory="project/images/", crop_size=40):
        self.IM_DIR = directory  # directory_doc="project/documents/"
        self.files = os.listdir(self.IM_DIR)
        self.n_crops = np.shape(self.files)[0]
        self.net_input_size = [self.n_crops, crop_size, crop_size]

        # self.X = np.zeros((self.net_input_size[0], self.net_input_size[1], self.net_input_size[2]))
        self.X = np.zeros((30, self.net_input_size[1], self.net_input_size[2]))

    def generate_net_input(self):
        show_figure = 0  # change this ver. to "1" if you would like to watch the pictures
        i = 0

        for f in self.files:
            im = cv2.imread(self.IM_DIR+f)
            im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)

            if show_figure:
                plt.imshow(im, cmap='gray')
                plt.show()

            # reshape_im = cv2.resize(im, dsize=(self.net_input_size[1], self.net_input_size[2]), interpolation=cv2.INTER_CUBIC)  # resize picture size for equal sizing
            reshape_im = cv2.resize(im, dsize=(self.net_input_size[1], self.net_input_size[2]), interpolation=cv2.INTER_CUBIC)  # resize picture size for equal sizing

            self.X[i] = reshape_im
            i += 1

        return self.X, self.n_crops
