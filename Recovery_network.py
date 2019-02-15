import numpy as np
import keras as keras
from keras.models import Sequential
from keras.layers import BatchNormalization, Dense, LSTM, Dropout, TimeDistributed, Conv2D, \
    MaxPooling2D, Flatten, Bidirectional
from keras import regularizers
from scipy.special import softmax
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


weight_decay = 0.0001
t_max = 5  # max number of cuts supported (hence max of 6^2 crops + 6 OOD = 42)
crop_size = 25  # size of each crop ("pixels")
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

    model = Sequential()
    model.add(TimeDistributed(Conv2D(15, (3, 3), kernel_initializer='random_uniform',
                                     activation='relu',
                                     padding='valid',
                                     kernel_regularizer=regularizers.l2(weight_decay)),
                              input_shape=(max_crops, crop_size, crop_size, 1)))

    # model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(1, 1))))
    model.add(TimeDistributed(Dropout(0.5)))
    model.add(TimeDistributed(BatchNormalization()))

    model.add(TimeDistributed(Conv2D(15, (3, 3), kernel_initializer='random_uniform',
                                     activation='relu',
                                     padding='valid',
                                     kernel_regularizer=regularizers.l2(weight_decay)))),

    # model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))
    model.add(TimeDistributed(Dropout(0.75)))

    model.add(TimeDistributed(Conv2D(10, (2, 2), kernel_initializer='random_uniform',
                                     activation='relu',
                                     padding='valid',
                                     kernel_regularizer=regularizers.l2(weight_decay)))),

    # model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))
    model.add(TimeDistributed(Dropout(0.75)))

    # Flatten model and feed to bidirectional LSTM
    model.add(TimeDistributed(Flatten()))
    model.add(Bidirectional(LSTM(output_dim, return_sequences=True), merge_mode='sum'))
    model.add(Dropout(0.4))
    model.add(TimeDistributed(Dense(output_dim, activation='softmax')))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

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
    t = int(t)
    stacked_image = np.zeros((int(t**2), pixels, pixels))
    print(output.shape, crops_set.shape)

    for i in range(int(t**2)):
        if output[i] != -1:
            stacked_image[i, :, :] = crops_set[int(output[i]), :, :]

    image = np.zeros((int(t*pixels), int(t*pixels)))
    for row in range(int(t)):
        for col in range(int(t)):
            image[(row*pixels):((row+1)*pixels), (col*pixels):((col+1)*pixels)] = \
                stacked_image[(t*row+col), :, :]

    plt.imshow(image)
    plt.savefig('test_picture.png')
    # print(image.shape)

    return image


def extract_crops(sample):
    crops = 0
    while sample[crops, 1, 1, 0] != 0:
        crops += 1

    return crops


# x = np.load("output/x_training_pic.npy")
# y = np.load("output/y_training_pic.npy")

training_data = np.load('output/train_data.npz')
x = training_data['a']
y = training_data['b']

print("Data Shapes:")
print(x.shape, y.shape)

x = x[:, :, :, :, np.newaxis]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

Model = define_model()

history = Model.fit(x_train, y_train, epochs=1, verbose=1, batch_size=128, validation_data=(x_test, y_test))

plot_history(history)

Model.save('Recovery_rev1.h5')

predict_sample = x_train[1:10, :, :, :, :]
predict_sample_tag = y_train[1:10, ]

prediction = Model.predict(x_train[1:10, :, :, :, :])
print(prediction.shape)

crops = extract_crops(x_train[5, :, :, :, :])

output = parse_output(y_train[5, :, :], n_crops=crops)

arrange_image(output, x_train[5, :, :, :, 0], t=np.floor(np.sqrt(crops)), pixels=crop_size)
