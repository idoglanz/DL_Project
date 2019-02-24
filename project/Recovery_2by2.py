import numpy as np
import keras as keras
from keras.models import Sequential
from keras.layers import BatchNormalization, Dense, LSTM, Dropout, TimeDistributed, Conv2D, \
    MaxPooling2D, Flatten, Bidirectional
from keras import regularizers
from scipy.special import softmax
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import shredder_public as shred


weight_decay = 0.005
t_max = 3  # max number of cuts supported (hence max of 6^2 crops + 6 OOD = 42)
crop_size = 75  # size of each crop ("pixels")
max_crops = t_max**2 + t_max
output_dim = t_max**2 + 2  # added 2 for OOD and zeros (padding) marking


def plot_history(history, baseline=None):
    his = history.history
    # val_acc = his['val_acc']
    train_acc = his['acc']
    # plt.plot(np.arange(len(val_acc)), val_acc, label='val_acc')
    plt.plot(np.arange(len(train_acc)), train_acc, label='acc')
    plt.legend()
    plt.savefig('testplot.png')
    plt.show(block=True)


def define_model():

    # define the CNN part of the network as a TimeDistributed input (each input is a set of crops,
    # a batch will therefore include N sets of such crops)

    # ---------------------------------------- CNN part ------------------------------------------
    model = Sequential()
    model.add(TimeDistributed(Conv2D(16, (3, 3), kernel_initializer='random_uniform',
                                     activation='relu',
                                     padding='same',
                                     kernel_regularizer=regularizers.l2(weight_decay)),
                              input_shape=(max_crops, crop_size, crop_size, 1)))

    model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(1, 1))))
    model.add(TimeDistributed(Dropout(0.3)))
    model.add(TimeDistributed(BatchNormalization()))

    model.add(TimeDistributed(Conv2D(32, (3, 3), kernel_initializer='random_uniform',
                                     activation='relu',
                                     padding='same',
                                     kernel_regularizer=regularizers.l2(weight_decay)),
                              input_shape=(max_crops, crop_size, crop_size, 1)))

    model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(1, 1))))
    # model.add(TimeDistributed(Dropout(0.5)))
    model.add(TimeDistributed(BatchNormalization()))

    model.add(TimeDistributed(Conv2D(64, (3, 3), kernel_initializer='random_uniform',
                                     activation='relu',
                                     padding='same',
                                     kernel_regularizer=regularizers.l2(weight_decay)),
                              input_shape=(max_crops, crop_size, crop_size, 1)))

    model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(1, 1))))
    model.add(TimeDistributed(Dropout(0.3)))
    model.add(TimeDistributed(BatchNormalization()))

    model.add(TimeDistributed(Conv2D(128, (5, 5), kernel_initializer='random_uniform',
                                     activation='relu',
                                     padding='same',
                                     kernel_regularizer=regularizers.l2(weight_decay)))),

    model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))
    # model.add(TimeDistributed(Dropout(0.3)))
    model.add(TimeDistributed(BatchNormalization()))

    model.add(TimeDistributed(Conv2D(256, (3, 3), kernel_initializer='random_uniform',
                                     activation='relu',
                                     padding='valid',
                                     kernel_regularizer=regularizers.l2(weight_decay)))),

    model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))
    model.add(TimeDistributed(Dropout(0.3)))
    model.add(TimeDistributed(BatchNormalization()))

    model.add(TimeDistributed(Conv2D(256, (3, 3), kernel_initializer='random_uniform',
                                     activation='relu',
                                     padding='valid',
                                     kernel_regularizer=regularizers.l2(weight_decay)))),

    model.add(TimeDistributed(MaxPooling2D((5, 5), strides=(5, 5))))
    # model.add(TimeDistributed(Dropout(0.6)))


    # ---------------------------------------- LSTM part ------------------------------------------

    # Flatten model and feed to bidirectional LSTM
    model.add(TimeDistributed(Flatten()))
    model.add(Bidirectional(LSTM(output_dim, return_sequences=True), merge_mode='sum'))
    # model.add(Dropout(0.5))

    model.add(Bidirectional(LSTM(output_dim, return_sequences=True), merge_mode='sum'))
    model.add(Dropout(0.3))

    model.add(Bidirectional(LSTM(output_dim, return_sequences=True), merge_mode='sum'))
    # model.add(Dropout(0.2))

    model.add(TimeDistributed(Dense(output_dim, activation='softmax')))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])

    print(model.summary())
    return model


def parse_output(data, n_crops):
    # output from matrix is a (t^2+t) by t^2 matrix as so rows are crops and columns are positions
    # (last two are OOD and padding)

    t = np.floor(np.sqrt(n_crops))
    n_OODs = int(n_crops - t**2)
    n_pic_crops = int(t**2)
    print('t = ' + str(t))

    output_vector = np.zeros(int(n_OODs+n_pic_crops))

    # First we clear out (t^2 + t - true_size) zeros padded crops
    data = data[0:(n_pic_crops+n_OODs), :]
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

    # Run Sinkhorn Softmax rows and cols (n iterations)
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


def arrange_image(output, crops_set, t, pixels, size_wo_pad, n):
    t = int(t)
    stacked_image = np.zeros((int(t**2), pixels, pixels))
    print(output.shape, crops_set.shape)

    for i in range(int(size_wo_pad)):
        if output[i] != -1:
            stacked_image[int(output[i]), :, :] = crops_set[i, :, :, 0]

    image = np.zeros((int(t*pixels), int(t*pixels)))
    for row in range(int(t)):
        for col in range(int(t)):
            image[(row*pixels):((row+1)*pixels), (col*pixels):((col+1)*pixels)] = \
                stacked_image[(t*row+col), :, :]

    plt.imshow(image, cmap='gray')
    # plt.show()
    file_name = 'test_picture_' + str(n) + '.png'
    plt.savefig(file_name)
    print(image.shape)

    return image


def extract_crops(sample):
    n_crops = 0
    while np.sum(sample[n_crops, :, :, 0]) != 0:
        n_crops += 1
        if n_crops >= len(sample):
            break
    return n_crops


# x = np.load("output/x_training_pic.npy")
# y = np.load("output/y_training_pic.npy")

# training_data = np.load('output/train_data.npz')
# x = training_data['a']
# y = training_data['b']

print("Generating data")
data_pic = shred.Data_shredder(directory="images/",
                               output_directory="output/",
                               num_of_duplication=20,
                               net_input_size=[int(max_crops), crop_size, crop_size])

data_doc = shred.Data_shredder(directory="documents/",
                               output_directory="output/",
                               num_of_duplication=1,
                               net_input_size=[int(max_crops), crop_size, crop_size])

x, y = data_pic.generate_data(tiles_per_dim=[3])

print("Finished generating data. Data shapes:")
print(x.shape, y.shape)

print('Shuffling data')
x, y = shred.shuffle_before_fit(x, y)

x = x[:, :, :, :, np.newaxis]


# change value of PAD and OOD labeling to lower value
# y[:, :, -2:] *= 0.01

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42)

Model = define_model()

history = Model.fit(x_train, y_train, epochs=20, verbose=1, batch_size=32, validation_data=(x_test, y_test))

plot_history(history)

Model.save('Recovery_rev1.h5')

predict_sample = x_train[0:10, :, :, :, :]
predict_sample_tag = y_train[0:10, :, :]


prediction = Model.predict(x_train[0:10, :, :, :, :])

# predict_test = prediction[2, :, :]

# print(predict_test)

print(prediction.shape)

for test in range(10):

    print('Test sample number:', test)

    print(np.argmax(prediction[test, :, :], axis=1))

    crops = extract_crops(x_train[test, :, :, :, :])

    output = parse_output(prediction[test, :, :], n_crops=crops)

arrange_image(output, x_train[test, :, :, :, :], t=np.floor(np.sqrt(crops)), pixels=crop_size, size_wo_pad=crops, n=test)