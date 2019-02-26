import read_folder
import Recovery
from keras.models import load_model


def predict(images, n_crops):

    model = load_model('tempmodel.h5')

    # OR:
    # model = Recovery.define_model()
    # model.load_weights("Desktop/weights.hdf5")  # TODO: define directory

    output = model.predict(images)
    labels = Recovery.parse_output(output, n_crops)

    return labels


def evaluate(file_dir='example/', crop_size=40):

    read_pics = read_folder.ReadFolder(directory=file_dir, crop_size=crop_size)

    images, n_crops = read_pics.generate_net_input()

    Y = predict(images, n_crops)

    return Y


evaluate(file_dir='output1/', crop_size=50)
