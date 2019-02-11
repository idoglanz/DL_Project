import cv2
import os
import numpy as np
import random
import matplotlib.pyplot as plt


class Data_shredder():
    def __init__(self, directory="project/images1/", output_directory="project/output1/", is_picture = 1, number_of_crops=4, net_input_size=[42, 100, 100]):
        self.X_training_documents = []
        self.IM_DIR = directory  # directory_doc="project/documents/"
        self.OUTPUT_DIR = output_directory
        self.files = os.listdir(self.IM_DIR)
        self.number_of_samples = np.size(self.files) * 3  # TODO: change the number "1" to the number of times that the same pic will be shuffled

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

            if add_random_crops:  # TODO: add randome crops
                for add in range():
                    randome_pic = np.array(random.sample(tiles_per_dim, k=1))[0]
                    h = np.array(random.sample(tiles_per_dim, k=1))[0]
                    w = np.array(random.sample(tiles_per_dim, k=1))[0]

                    im = cv2.imread(self.IM_DIR + randome_pic)
                    im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
                    height = im.shape[0]
                    width = im.shape[1]

                    frac_h = height // self.tiles_per_dim
                    frac_w = width // self.tiles_per_dim

                    crop = im[h*frac_h:(h+1)*frac_h, w*frac_w:(w+1)*frac_w]  # create crop
                    reshape_crop = cv2.resize(crop, dsize=(self.n_data_size[1], self.n_data_size[2]), interpolation=cv2.INTER_CUBIC)  # resize picture size for equal sizing
                    if pic:  # depends of the input date save in picture matrix or in document matrix
                        # self.X_training[j,i][j][i] = reshape_crop
                        self.X_training[j, i] = reshape_crop
                        self.y_training[j, i, 36] = 1  # TODO: add spam
                    if show_figure:
                        plt.imshow(reshape_crop, cmap='gray', interpolation='bicubic')
                        plt.show()
                    i += 1

            for h in range(self.tiles_per_dim):

                for w in range(self.tiles_per_dim):

                    crop = im[h*frac_h:(h+1)*frac_h, w*frac_w:(w+1)*frac_w]  # create crop
                    cv2.imwrite(self.OUTPUT_DIR+f[:-4]+"_{}.jpg".format(str(i).zfill(2)), crop)  # save the crop of picture
                    i = i+1
                    # print(j)
                    # print(w)
                    reshape_crop = cv2.resize(crop, dsize=(self.n_data_size[1], self.n_data_size[2]), interpolation=cv2.INTER_CUBIC)  # resize picture size for equal sizing
                    # print(np.shape(reshape_crop))
                    # print(j)
                    # print(w)
                    if pic:  # depends of the input date save in picture matrix or in document matrix
                        # self.X_training[j,i][j][i] = reshape_crop
                        self.X_training[j, i] = reshape_crop
                        self.y_training[j, i, i] = 1  # TODO: add spam

                    if show_figure:
                        plt.imshow(reshape_crop, cmap='gray', interpolation='bicubic')
                        plt.show()
            self.y_training[j, i + 1:, 37] = 1  # TODO: check if i or i+1

            j = j+1
            print(j)
            self.X_training, self.y_training = self.shuffle_pic(self.X_training, self.y_training, i)

        return self.X_training, self.y_training  #, x_test, y_test

    def shuffle_pic(self, picture_matrix, tag_vector, i):
        assert np.shape(picture_matrix)[1] == np.shape(tag_vector)[1]
        p = np.random.permutation(i-1)
        picture_matrix[:, :i-1], tag_vector[:, :i-1] = picture_matrix[:, p], tag_vector[:, p]
        return picture_matrix, tag_vector










