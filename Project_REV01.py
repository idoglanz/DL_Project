import shredder_public as shred
import evaluate
import numpy as np


if __name__ == '__main__':
    data_pic = shred.Data_shredder(directory="images/",
                                   output_directory="output/",
                                   num_of_duplication=15,
                                   net_input_size=[30, 40, 40])

    data_doc = shred.Data_shredder(directory="documents/",
                                   output_directory="output/",
                                   num_of_duplication=1,
                                   net_input_size=[30, 40, 40])

    # data_doc = shred.Data_shredder(directory="project/documents/", output_directory="project/output/", net_input_size=[42, 100, 100])

    x, y = data_pic.generate_data(tiles_per_dim=[2, 4, 5])

    # x_new, y_new = data_doc.generate_data(tiles_per_dim=[2, 4, 5])
    #
    # x = np.append(x, x_new, axis=0)
    # y = np.append(y, y_new, axis=0)

    # x, y = shred.shuffle_before_fit(x, y)

    np.savez_compressed("output/train_data.npz", a=x, b=y)


    # x, y = data_pic.generate_data(tiles_per_dim=[2, 4, 5])
    # np.save("project/output1/x_training_doc.npy", x)