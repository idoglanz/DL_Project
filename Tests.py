import numpy as np
from scipy.special import softmax
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


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
    print(data)

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


def arrange_image(output, crops_set, t, pixels):
    t = int(t)
    stacked_image = np.zeros((int(t**2), pixels, pixels))
    print(output.shape, crops_set.shape)

    for i in range(len(crops_set)):
        if output[i] != -1:
            stacked_image[int(output[i]), :, :] = crops_set[i, :, :, 0]

    image = np.zeros((int(t*pixels), int(t*pixels)))
    for row in range(int(t)):
        for col in range(int(t)):
            image[(row*pixels):((row+1)*pixels), (col*pixels):((col+1)*pixels)] = \
                stacked_image[(t*row+col), :, :]

    plt.imshow(image, cmap='gray')
    plt.show()
    # plt.savefig('test_picture.png')
    print(image.shape)

    return image


def extract_crops(sample):
    crops = 0
    while np.sum(sample[crops, :, :, 0]) != 0:
        crops += 1
        if crops >= len(sample):
            break
    return crops


training_data = np.load('output/train_data.npz')
x = training_data['a']
y = training_data['b']

print("Data Shapes:")
print(x.shape, y.shape)

x = x[:, :, :, :, np.newaxis]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# x_train = x
# y_train = y

n = 10
crop_size = 25
test_sample = x_train[n, :, :, :, :]
test_sample_tag = y_train[n, :, :]

crops = extract_crops(test_sample)

print('Crops:' + str(crops))

output = parse_output(test_sample_tag, n_crops=crops)

arrange_image(output, test_sample, t=np.floor(np.sqrt(crops)), pixels=crop_size)
