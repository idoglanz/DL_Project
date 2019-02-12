import cv2
import os
import numpy as np
import random
import matplotlib.pyplot as plt


class Data_shredder():
    def __init__(self, directory="project/images/", output_directory="project/output/", is_picture = 1, number_of_crops=4, net_input_size=[42, 100, 100]):
        self.X_training_documents = []
        self.IM_DIR = directory  # directory_doc="project/documents/"
        self.OUTPUT_DIR = output_directory
        self.files = os.listdir(self.IM_DIR)
        self.number_of_samples = np.size(self.files) * 1  # TODO: change the number "1" to the number of times that the same pic will be shuffled

        self.tiles_per_dim = number_of_crops      # update this number for 4X4 crop 2X2 or 5X5 crops.
        self.n_data_size = net_input_size  # here we are defining the training set size, this dimension is depend on the net input

        self.X_training = np.zeros((self.number_of_samples, self.n_data_size[0], self.n_data_size[1], self.n_data_size[2]))
        self.y_training = np.zeros((self.number_of_samples, self.n_data_size[0], 38))
        # self.X_training_documents = np.zeros((np.shape(self.files_doc), self.n_data_size[0], self.n_data_size[1], self.n_data_size[2]))

        self.X_validation = np.zeros((self.number_of_samples, self.n_data_size[0], self.n_data_size[1], self.n_data_size[2]))
        self.y_validation = np.zeros((self.number_of_samples, self.n_data_size[0], 38))

    def generate_data(self, pic=1, tiles_per_dim=[2, 4, 5]):

        show_figure = 0  # change this ver. to "1" if you would like to watch the pictures
        j = 0

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

            if show_figure:
                plt.imshow(im, cmap='gray')
                plt.show()

            for h in range(self.tiles_per_dim):
                for w in range(self.tiles_per_dim):

                    crop = im[h*frac_h:(h+1)*frac_h, w*frac_w:(w+1)*frac_w]  # create crop
                    # cv2.imwrite(self.OUTPUT_DIR+f[:-4]+"_{}.jpg".format(str(i).zfill(2)), crop)  # save the crop of picture

                    reshape_crop = cv2.resize(crop, dsize=(self.n_data_size[1], self.n_data_size[2]), interpolation=cv2.INTER_CUBIC)  # resize picture size for equal sizing

                    self.X_training[j, i] = reshape_crop
                    self.y_training[j, i, i] = 1
                    i = i + 1

                    if show_figure:
                        plt.imshow(reshape_crop, cmap='gray', interpolation='bicubic')
                        plt.show()

            add_random_crops = 1
            if add_random_crops:
                for add in range(self.tiles_per_dim):
                    randome_pic = self.files[np.random.randint(1, self.number_of_samples)]  # todo: self.number_of_samples/3
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

                    self.X_training[j, i] = reshape_crop
                    self.y_training[j, i, 36] = 1

                    if show_figure:
                        plt.imshow(reshape_crop, cmap='gray', interpolation='bicubic')
                        plt.show()
                    i += 1

            self.y_training[j, i-1:, 37] = 1  # TODO: check if i or i+1

            # print(j)
            self.X_training, self.y_training = self.shuffle_pic(self.X_training, self.y_training, j, i)
            j = j + 1
        return self.X_training, self.y_training  #, x_test, y_test

    def shuffle_pic(self, picture_matrix, tag_vector, j, i):
        assert np.shape(picture_matrix)[1] == np.shape(tag_vector)[1]
        p = np.random.permutation(i-1)
        picture_matrix[j, :i-1], tag_vector[j, :i-1] = picture_matrix[j, p], tag_vector[j, p]
        return picture_matrix, tag_vector










