import shredder_public as shred
import evaluate
import numpy as np


if __name__ == '__main__':
    data_pic = shred.Data_shredder(directory="project/images1/", output_directory="project/output/", num_of_duplication=1, net_input_size=[42, 100, 100])
    data_doc = shred.Data_shredder(directory="project/documents/", output_directory="project/output/", net_input_size=[42, 100, 100])

    x, y = data_pic.generate_data(tiles_per_dim=[2, 4, 5])
    np.save("project/output1/x_training_pic.npy", x)
    np.save("project/output1/y_training_pic.npy", y)

    # x, y = data_pic.generate_data(tiles_per_dim=[2, 4, 5])
    # np.save("project/output1/x_training_doc.npy", x)
    # np.save("project/output1/y_training_doc.npy", y)
