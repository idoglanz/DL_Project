import shredder_public as shred
import evaluate
import numpy as np


if __name__ == '__main__':
    data_pic = shred.Data_shredder(directory="images/", output_directory="output/", num_of_duplication=6, net_input_size=[30, 25, 25])
    data_doc = shred.Data_shredder(directory="documents/", output_directory="output/", net_input_size=[30, 25, 25])

    x, y = data_pic.generate_data(tiles_per_dim=[2, 4, 5])
    # np.save("output/x_training_pic.npy", x)
    # np.save("output/y_training_pic.npy", y)
    np.savez_compressed("output/train_data.npz", a=x, b=y)

    # x, y = data_pic.generate_data(tiles_per_dim=[2, 4, 5])
    # np.save("project/output1/x_training_doc.npy", x)
    # np.save("project/output1/y_training_doc.npy", y)
