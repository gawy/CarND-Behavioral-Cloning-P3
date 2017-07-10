import keras.models
from keras.models import Sequential
from keras.layers import Activation, Dense, Flatten, Convolution2D, MaxPool2D, Cropping2D, Lambda, Input, Dropout
import csv
import scipy.ndimage as ndimage
import numpy as np
import cv2
import logging
from sklearn.utils import shuffle as sk_shuffle
from sklearn.model_selection import train_test_split
from keras.applications.vgg16 import VGG16
from keras.models import Model

image_size = (160, 320, 3)

# Creates DL model inpired by LeNet
# returns: model object
def drive_model_create():
    log = logging.getLogger(__name__)

    log.info('Creating model...')
    #    define model here
    k_size = (5,5)

    model = Sequential()
    model.add(Cropping2D(((60, 0), (0, 0)), input_shape=image_size))
    model.add(Lambda(lambda x: x / 255.0 - 0.5))

    model.add(Convolution2D(6, k_size, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool2D((2,2)))

    model.add(Convolution2D(16, k_size, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool2D((2,2)))

    model.add(Convolution2D(24, k_size, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool2D((2,2)))

    model.add(Flatten())
    model.add(Dropout(0.4))

    model.add(Dense(120, activation='relu'))
    model.add(Dense(80, activation='relu'))

    model.add(Dense(1))

    log.info('Done creating model.')
    return model


# Creates model based on pretrained VGG16 (from Keras library)
# return: keras model
def create_vgg():
    log = logging.getLogger(__name__)

    log.info('Creating VGG based model...')

    input_layer = Input(image_size)
    layer = Cropping2D(((60, 0), (0, 0)))(input_layer)
    layer = Lambda(lambda x: x / 255.0 - 0.5)(layer)


    #224x224.

    base_model = VGG16(False, weights='imagenet', input_tensor=layer)
    base_output = base_model.output

    layer = Flatten()(base_output)
    layer = Dense(120, activation='relu')(layer)
    layer = Dense(80, activation='relu')(layer)

    layer = Dense(1)(layer)

    model = Model(inputs=input_layer, outputs=layer)

    # disable training on VGG layers
    for l in base_model.layers:
        l.trainable = False

    log.info('Done creating model.')
    return model

# Executes training code for any Keras model.
def train_in_driving(train_gen, valid_gen, train_data_len, valid_data_len, model):
    log = logging.getLogger(__name__)

    log.info('Start trainig...')

    #Compile and train model
    model.compile('adam', 'mean_squared_error')
    model.fit_generator(train_gen, train_data_len, epochs=3, validation_data=valid_gen, validation_steps=valid_data_len)

    log.info('Saving model to model.h5')
    model.save("model.h5")


# Read CSV file and create train/validation geneators that will be used while training.
# returns: tuple of (train_generator, valid_generator) - generators to load train and validation data when training
def create_data_generators():
    log = logging.getLogger(__name__)

    log.info('Reading csv file...')
    # read csv data file

    samples = []
    with open('./data/driving_log.csv') as f:
        csv_reader = csv.reader(f)
        next(csv_reader) #just skip the header line for test data set provided by Udacity

        for row in csv_reader:
            samples.append(row)

    train_samples, valid_samples = train_test_split(samples, test_size=0.2)
    log.info('Samples read. Sending data to generator...')

    generator_for_train = input_generator(train_samples, 32)
    generator_for_valid = input_generator(valid_samples, 32)

    return generator_for_train, generator_for_valid, len(train_samples), len(valid_samples)

def input_generator(data, batch_size=32):
    log = logging.getLogger(__name__)
    data_length = len(data)
    while 1:
        for offset in range(0, data_length, batch_size):
            samples = data[offset:offset + batch_size]
            log.debug('Batch offset:{}, len:{}'.format(offset, len(samples)))


            x_data = []
            y_data = []

            for row in samples:

                s_angle = float(row[3])
                im_path, l_im_path, r_im_path = row[0], row[1], row[2]

                load_image(im_path, s_angle, x_data, y_data)
                correction_angle = 0.21
                load_image(l_im_path, s_angle + correction_angle, x_data, y_data)
                load_image(r_im_path, s_angle - correction_angle, x_data, y_data)


            log.debug('Processing image files: converting X...')
            # Normalize them
            x_data = np.array(x_data)
            log.debug('X shape {}'.format(x_data.shape))

            log.debug('Processing image files: Y...')
            y_data = np.array(y_data)

            x_data, y_data = sk_shuffle(x_data, y_data)

            log.debug(x_data[10])
            log.debug(y_data[:20])

            log.debug('Data batch ready')
            yield x_data, y_data


# Read image file via provided path
# im_path - path to a file to load
# s_angle - strearing angle adjustment for image. Usage for side camera images to correct the steering.
# x_data - image data
# y_data - steering angles for corresponding images
def load_image(im_path, s_angle, x_data, y_data):

    path = im_path[im_path.rfind('/'):]
    im = cv2.imread('./data/IMG' + path)
    # im = cv2.resize(im, (224,224,3), interpolation=cv2.INTER_AREA)
    # process image
    x_data.append(im)
    y_data.append(s_angle)

    # add flipped images to avoid one side bias
    y_data.append(-(s_angle))
    x_data.append(np.fliplr(im))


# Execute when ran from terminal

if __name__ == '__main__' :
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)

    train_gen, valid_gen, train_len, valid_len = create_data_generators()

    # x_data, y_data = next(train_gen)
    # log.debug('X_size {}'.format(len(x_data)))
    # log.debug('Y_sample: {}'.format(y_data[:10]))

    model = drive_model_create()# create_vgg()
    train_in_driving(train_gen, valid_gen, train_len, valid_len, model)
