import cv2
import os
import numpy as np
import matplotlib.pyplot as plt


class Data_shredder():
    def __init__(self, directory="project/images/", output_directory="project/output/", is_picture = 1, number_of_crops=4, net_input_size=[32, 4, 4]):
        self.X_training_documents = []
        self.IM_DIR = directory  # directory_doc="project/documents/"
        self.OUTPUT_DIR = output_directory
        self.files_im = os.listdir(self.IM_DIR)
        # print(self.files)
        # self.number_of_samples = np.shape(self.files_im) * 5  # TODO: change the number "5" to the number of times that the same pic will be shuffled
        self.number_of_samples = 5  # TODO: change the number "5" to the number of times that the same pic will be shuffled

    # update this number for 4X4 crop 2X2 or 5X5 crops.
        self.tiles_per_dim = number_of_crops
        self.n_data_size = net_input_size  # here we are defining the training set size, this dimension is depend on the net input

        self.X_training = np.zeros((self.number_of_samples, self.n_data_size[0], self.n_data_size[1], self.n_data_size[2]))
        self.y_training = np.zeros((self.number_of_samples, self.n_data_size[0]))
        # self.X_training_documents = np.zeros((np.shape(self.files_doc), self.n_data_size[0], self.n_data_size[1], self.n_data_size[2]))

        self.X_validation = np.zeros((self.number_of_samples, self.n_data_size[0], self.n_data_size[1], self.n_data_size[2]))
        self.y_validation = np.zeros((self.number_of_samples, self.n_data_size[0]))

    def generate_data(self, pic=1):

        show_figure = 0  # change this ver. to "1" if you would like to watch the pictures
        j = 0

        for f in self.files_im:
            self.tiles_per_dim = 4  # sample([2, 4, 5], k=1)
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
                    cv2.imwrite(self.OUTPUT_DIR+f[:-4]+"_{}.jpg".format(str(i).zfill(2)), crop)  # save the crop of picture
                    i = i+1
                    print(f)
                    reshape_crop = cv2.resize(crop, dsize=(self.n_data_size[1], self.n_data_size[2]), interpolation=cv2.INTER_CUBIC)  # resize picture size for equal sizing

                    if pic:  # depends of the input date save in picture matrix or in document matrix
                        self.X_training[f][w] = reshape_crop

                    if show_figure:
                        plt.imshow(reshape_crop, cmap='gray', interpolation='bicubic')
                        plt.show()
                j = j+1
        return self.X_training_picture, self.X_training_documents, self.y, x_test, y_test

    def unison_shuffled_copies(self, picture_matrix, tag_vector):
        assert len(picture_matrix) == len(tag_vector)
        p = np.random.permutation(len(picture_matrix))
        return picture_matrix[p], tag_vector[p]










