import cv2
import os
import numpy as np
import random
import matplotlib.pyplot as plt


class Data_shredder():
    def __init__(self, directory="project/images/", output_directory="project/output1/", num_of_duplication=1, number_of_crops=4, net_input_size=[30, 100, 100]):
        self.X_training_documents = []
        self.IM_DIR = directory  # directory_doc="project/documents/"
        self.OUTPUT_DIR = output_directory
        self.files = os.listdir(self.IM_DIR)
        self.num_of_duplication = num_of_duplication
        self.number_of_samples = np.size(self.files) * num_of_duplication

        self.tiles_per_dim = number_of_crops      # update this number for 4X4 crop 2X2 or 5X5 crops.
        self.n_data_size = net_input_size  # here we are defining the training set size, this dimension is depend on the net input

        self.X_training = np.zeros((self.number_of_samples, self.n_data_size[0], self.n_data_size[1], self.n_data_size[2]))
        self.X_training_as_one = np.zeros((self.number_of_samples, self.n_data_size[1]* self.n_data_size[0], self.n_data_size[2]))
        # self.X_training_as_one = np.zeros((self.number_of_samples, self.n_data_size[2] * self.n_data_size[0], self.n_data_size[1]))
        # self.y_training = np.zeros((self.number_of_samples, self.n_data_size[0], int(np.floor(np.sqrt(self.n_data_size[0]))**2+2)))
        self.y_training = np.zeros((self.number_of_samples, 4000, 200))

    def generate_data(self, add_random_crops=1, tiles_per_dim=[2, 4, 5], save_crops=0):
        show_figure = 1  # change this ver. to "1" if you would like to watch the pictures
        j = 0
        for duplication in range(self.num_of_duplication):
            print("Duplication #", duplication, "out of", self.num_of_duplication)
            for f in self.files:
                self.tiles_per_dim = np.array(random.sample(tiles_per_dim, k=1))[0]
                # print(self.tiles_per_dim)

                im = cv2.imread(self.IM_DIR+f)
                im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
                height = im.shape[0]
                width = im.shape[1]

                frac_h = height//self.tiles_per_dim
                frac_w = width//self.tiles_per_dim
                i = 0

                im_reshape = cv2.resize(im, dsize=(200, 4000),
                                          interpolation=cv2.INTER_CUBIC)  # resize picture size for equal sizing

                self.y_training[j] = im_reshape

                if show_figure:
                    plt.imshow(self.y_training[j], cmap='gray')
                    plt.show()

                for h in range(self.tiles_per_dim):
                    for w in range(self.tiles_per_dim):

                        crop = im[h*frac_h:(h+1)*frac_h, w*frac_w:(w+1)*frac_w]  # create crop
                        if save_crops:
                            cv2.imwrite(self.OUTPUT_DIR+f[:-4]+"_{}.jpg".format(str(i).zfill(2)), crop)  # save the crops of picture

                        reshape_crop = cv2.resize(crop, dsize=(self.n_data_size[1], self.n_data_size[2]), interpolation=cv2.INTER_CUBIC)  # resize picture size for equal sizing

                        self.X_training[j, i] = reshape_crop
                        i = i + 1

                        # if show_figure:
                            # plt.imshow(reshape_crop, cmap='gray')
                            # plt.show()

                if add_random_crops:
                    for add in range(self.tiles_per_dim):
                        randome_pic = self.files[np.random.randint(1, self.number_of_samples/self.num_of_duplication)]  # todo: self.number_of_samples/3
                        h = np.random.randint(0, self.tiles_per_dim-1)  # TODO: change to "self.tiles_per_dim - 1"?
                        w = np.random.randint(0, self.tiles_per_dim-1)

                        im = cv2.imread(self.IM_DIR + randome_pic)
                        im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
                        height = im.shape[0]
                        width = im.shape[1]

                        frac_h = height // self.tiles_per_dim
                        frac_w = width // self.tiles_per_dim

                        crop = im[h*frac_h:(h+1)*frac_h, w*frac_w:(w+1)*frac_w]  # create crop
                        reshape_crop = cv2.resize(crop, dsize=(self.n_data_size[1], self.n_data_size[2]), interpolation=cv2.INTER_CUBIC)  # resize picture size for equal sizing

                        if random.uniform(0, 3) < 2:
                            self.X_training[j, i] = reshape_crop
                        # self.y_training[j, i, int(np.floor(np.sqrt(self.n_data_size[0]))**2)] = 1

                        # if show_figure:
                        #     plt.imshow(reshape_crop, cmap='gray', interpolation='bicubic')
                        #     plt.show()
                        i += 1

                # self.y_training[j, i:, int(np.floor(np.sqrt(self.n_data_size[0]))**2+1)] = 1  # TODO: check if i or i+1

                # print(j)
                self.X_training[j] = self.shuffle_pic(self.X_training[j], j, i)
                self.X_training_as_one[j] = np.reshape(self.X_training[j], (self.n_data_size[1]*i, self.n_data_size[2]))
                if show_figure:
                    plt.imshow(self.X_training_as_one[j], cmap='gray')
                    plt.show()
                j = j + 1
        return self.X_training_as_one, self.y_training  #, x_test, y_test

    def shuffle_pic(self, picture_matrix, j, i):
        # assert np.shape(picture_matrix)[1] == np.shape(tag_vector)[1]
        p = np.random.permutation(i)
        picture_matrix = picture_matrix[p]
        return picture_matrix


def shuffle_before_fit(x, y):
    i = np.shape(x)[0]
    p = np.random.permutation(i)
    x, y = x[p], y[p]
    return x, y

