import numpy as np
from keras.models import Sequential
from keras.layers import BatchNormalization, Dense, LSTM, Dropout, TimeDistributed, Conv2D, \
    MaxPooling2D, Flatten, Bidirectional
from keras import regularizers
from scipy.special import softmax
from random import randint
import matplotlib.pyplot as plt

weight_decay = 0.0001
t_max = 6  # max number of cuts supported (hence max of 6^2 crops + 6 OOD = 42)
crop_size = 100  # size of each crop ("pixels")
max_crops = t_max**2 + t_max
output_dim = t_max**2 + 2  # added 2 for OOD and zeros (padding) marking


def define_model():

    # define the CNN part of the network as a TimeDistributed input (each input is a set of crops,
    # a batch will therefore include N sets of such crops)

    model = Sequential()
    model.add(TimeDistributed(Conv2D(15, (10, 10), kernel_initializer='random_uniform',
                                     activation='relu',
                                     padding='valid',
                                     kernel_regularizer=regularizers.l2(weight_decay)),
                              input_shape=(max_crops, crop_size, crop_size, 1)))

    model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(1, 1))))
    model.add(TimeDistributed(Dropout(0.5)))
    model.add(TimeDistributed(BatchNormalization()))

    model.add(TimeDistributed(Conv2D(15, (5, 5), kernel_initializer='random_uniform',
                                     activation='relu',
                                     padding='valid',
                                     kernel_regularizer=regularizers.l2(weight_decay)))),

    model.add(TimeDistributed(MaxPooling2D((5, 5), strides=(2, 2))))
    model.add(TimeDistributed(Dropout(0.75)))

    model.add(TimeDistributed(Conv2D(15, (3, 3), kernel_initializer='random_uniform',
                                     activation='relu',
                                     padding='valid',
                                     kernel_regularizer=regularizers.l2(weight_decay)))),

    model.add(TimeDistributed(MaxPooling2D((5, 5), strides=(2, 2))))
    model.add(TimeDistributed(Dropout(0.75)))

    # Flatten model and feed to bidirectional LSTM
    model.add(TimeDistributed(Flatten()))
    model.add(Bidirectional(LSTM(output_dim, return_sequences=True), merge_mode='sum'))
    model.add(Dropout(0.4))
    model.add(TimeDistributed(Dense(output_dim, activation='softmax')))

    model.compile(loss='categorical_crossentropy', optimizer='adam')

    print(model.summary())
    return model


# test_model = define_model()


def parse_output(data, n_crops):
    # output from matrix is a (t^2+t) by t^2 matrix as so rows are crops and columns are positions
    # (last two are OOD and padding)

    t = np.floor(np.sqrt(n_crops))
    n_OODs = int(n_crops - t**2)
    n_pic_crops = int(t**2)
    n_padded = int(len(data) - (n_pic_crops+n_OODs))
    print('t = ' + str(t))

    output_vector = np.zeros(int(n_OODs+n_pic_crops))

    # First we clear out (t^2 + t - true_size) zeros padded crops
    data = data[:-n_padded, :]
    # [data[int(np.argmax(data[:, -1], axis=0)), :].fill(0) for i in range(n_padded)]

    data = softmax(data, axis=1)

    # Second we clear OOD elements
    for i in range(n_OODs):
        current_max = np.argmax(data[:, -2], axis=0)
        output_vector[current_max] = -1
        data[current_max, :].fill(0)

    # Remove OODs (and remember their position)
    OOD_locations = [np.sum(data[i, :]) != 0 for i in range(len(data))]
    print(OOD_locations)
    augmented_data = data[OOD_locations, :n_pic_crops]

    # Run Sinkhorn softmax rows and cols (n iterations)
    for k in range(4):
        augmented_data = softmax(augmented_data, axis=1)
        augmented_data = softmax(augmented_data, axis=0)

    position = 0
    for j in range(len(output_vector)):
        if output_vector[j] != -1:
            output_vector[j] = int(np.argmax(augmented_data[position, :]))
            augmented_data[:, int(output_vector[j])].fill(0)
            position += 1

    check_repeated(output_vector)

    print(output_vector)

    return output_vector


def check_repeated(vector):
    histogram = np.zeros(len(vector))
    for i in range(len(vector)):
        histogram[i] = sum(vector == i)

    return histogram


def arrange_image(output, crops_set, t, pixels):
    stacked_image = np.zeros((t**2, pixels, pixels))
    print(output.shape, crops_set.shape)

    for i in range(len(output)):
        if output[i] != -1:
            stacked_image[i, :, :] = crops_set[int(output[i]), :, :]

    image = np.zeros((t*pixels, t*pixels))
    for row in range(int(t)):
        for col in range(int(t)):
            image[(row*pixels):((row+1)*pixels), (col*pixels):((col+1)*pixels)] = \
                stacked_image[(t*row+col), :, :]

    plt.imshow(image)
    plt.show()

    print(image.shape)
    return image


x_train = np.load("output1/x_training.npy")
y_train = np.load("output1/y_training.npy")

# test_data = np.random.randint(10, size=(t_max**2 + t_max, t_max**2 + 2))
# crop_set = np.random.randint(10, size=(42, pix, pix))

# x_train = x_train[:, :, :, :, np.newaxis]
# print(x_train.shape)
# Model = define_model()

# Model.fit(x_train, y_train, epochs=1, verbose=1, batch_size=30)

# prediction = Model.predict(x_train[:, :, :, :, :])
# print(prediction.shape)
output = parse_output(y_train[0, :, :], n_crops=25)

# print(y_train[6])
# for i in range(42):
#     image = x_train[6, i, :, :]

    # plt.imshow(image)
    # plt.show()

arrange_image(output, x_train[0, :, :, :], t=5, pixels=100)
