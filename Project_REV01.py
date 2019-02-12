import shredder_public as shred
import evaluate
import numpy as np


if __name__ == '__main__':
    data = shred.Data_shredder()
    x, y = data.generate_data(pic=1, tiles_per_dim=[2, 4, 5])
    np.save("project/output/x_training.npy", x)
    np.save("project/output/y_training.npy", y)
    # print(x)
    # [x_lern, y_lern, ] = data.generate_data()


