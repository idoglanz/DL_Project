import shredder_public as shred
import evaluate


if __name__ == '__main__':
    data = shred.Data_shredder()
    data.generate_data(pic=1)
    # [x_lern, y_lern, ] = data.generate_data()


