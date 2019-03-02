import AutoEncoder_shredder as shred
import keras
from keras.models import load_model
from keras.datasets import cifar10
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

from keras.optimizers import Adam
import os
import pickle
import numpy as np


def define_model():
    input_img = Input(shape=(4000, 200, 1))
    x = Conv2D(64, (3, 3), padding='same')(input_img)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(16, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(16, (3, 3), padding='same')(encoded)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(1, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    decoded = Activation('sigmoid')(x)

    model = Model(input_img, decoded)
    model.compile(optimizer='adam', loss='binary_crossentropy')

    return model


def main():
    print("Generating data")
    data_pic = shred.Data_shredder(directory="project/images/",
                                   output_directory="project/output/",
                                   num_of_duplication=1,
                                   net_input_size=[20, 200, 200])

    # data_doc = shred.Data_shredder(directory="project/documents/",
    #                                output_directory="project/output/",
    #                                num_of_duplication=9,
    #                                net_input_size=[int(max_crops), crop_size, crop_size])

    x, y = data_pic.generate_data(tiles_per_dim=[4])

    x = x[:, :, :, np.newaxis]
    y = y[:, :, :, np.newaxis]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42)


    Model = define_model()

    es_cb = EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='auto')
    chkpt = 'AutoEncoder_Deep_weights.{epoch:02d}-{loss:.2f}-{val_loss:.2f}.hdf5'
    cp_cb = ModelCheckpoint(filepath=chkpt, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

    Model.fit(x_train, y_train, batch_size=32, epochs=2, verbose=1, validation_data=(x_test, y_test),
              callbacks=[es_cb, cp_cb], shuffle=True)

    Model.save('Model.h5')
    
    # x1, y1 = data_doc.generate_data(tiles_per_dim=[5])



main()
