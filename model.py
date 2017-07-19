import argparse
import logging

from keras.models import Sequential
from keras.layers import Activation, Dense, Flatten, Convolution2D, MaxPool2D, Cropping2D, Lambda, Input, Dropout
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.utils import plot_model
from keras.models import load_model

import csv
import numpy as np
import random as rnd
import cv2
import math
import time
import pydot_ng
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.utils import shuffle as sk_shuffle
from sklearn.model_selection import train_test_split

image_size = (160, 320, 3)


# Creates DL model inpired by LeNet
# returns: model object
def drive_model_create():
    log = logging.getLogger(__name__)

    log.info('Creating model...')
    #    define model here
    k_size = (5,5)

    model = Sequential()
    model.add(Cropping2D(((60, 20), (0, 0)), input_shape=image_size)) #80x320x3
    model.add(Lambda(lambda x: x / 255.0 - 0.5))

    model.add(Convolution2D(24, k_size, padding='valid', strides=(2,2), activation='relu')) #38x158x24
    model.add(MaxPool2D((4,4), strides=(1,1), padding='same'))

    model.add(Convolution2D(36, k_size, padding='valid', strides=(2,2), activation='relu')) #17x37x36
    model.add(MaxPool2D((2,2), strides=(1,1), padding='same'))

    model.add(Convolution2D(48, k_size, padding='valid', strides=(2,2), activation='relu')) #6x16x48
    model.add(MaxPool2D((2,2), strides=(1,1), padding='same'))

    model.add(Convolution2D(64, (3,3), padding='valid', activation='relu')) # 5x15x64

    model.add(Convolution2D(64, (3,3), padding='valid', activation='relu')) #4x14x64

    model.add(Flatten())  #3584
    model.add(Dropout(0.4))

    model.add(Dense(1164, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))

    model.add(Dense(1))

    log.info('Done creating model.')
    return model



# Executes training code for any Keras model.
def train_in_driving_with_generator(train_gen, valid_gen, train_data_len, valid_data_len, model, epochs_count=3,
                                    label_name=''):
    log = logging.getLogger(__name__)

    log.info('Start trainig...')

    lbl = ''
    if len(label_name):
        lbl = '-' + label_name

    #Compile and train model
    model.compile('adam', 'mean_squared_error')
    history_object = model.fit_generator(train_gen, train_data_len, epochs=epochs_count, validation_data=valid_gen,
                        validation_steps=valid_data_len)

    log.info('Saving model to model.h5')
    model.save("model{}.h5".format(lbl))

    log.info(history_object.history['loss'])
    log.info(history_object.history['val_loss'])

    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')


    plt.savefig('training_loss{}.png'.format(lbl), bbox_inches='tight')


# To spead up training and in case we have enough memory and small enough model, all data is loaded into memory
# returns: tuple of images and steering angles
def read_all_data_to_memory():

    log = logging.getLogger(__name__)

    log.info('Reading csv file...')
    # read csv data file

    samples = []
    with open('./data/driving_log.csv') as f:
        csv_reader = csv.reader(f)
        next(csv_reader)  # just skip the header line for test data set provided by Udacity

        x_data, y_data = load_samples(csv_reader)

    log.info('All data read to memory')
    return x_data, y_data


# Read CSV file and create train/validation geneators that will be used while training.
# returns: tuple of (train_generator, valid_generator) - generators to load train and validation data when training
def create_data_generators(batch_size=32):
    log = logging.getLogger(__name__)

    log.info('Reading csv file...')
    # read csv data file

    samples = []
    with open('./data/driving_log.csv') as f:
        csv_reader = csv.reader(f)
        next(csv_reader) #just skip the header line for test data set provided by Udacity

        for row in csv_reader:
            s_angle = float(row[3])
            # if abs(s_angle) < 0.05 and rnd.random() <= 0.85: continue
            # if abs(s_angle) < 0.2 and rnd.random() <= 0.85: continue
            samples.append(row)

    # samples = samples[:500]
    train_samples, valid_samples = train_test_split(samples, test_size=0.2)
    log.info('Samples read, {} lines. Sending data to generator...'.format(len(samples)))

    generator_for_train = input_generator(train_samples, batch_size)
    generator_for_valid = input_generator(valid_samples, batch_size, is_for_validation=True)

    return generator_for_train, generator_for_valid, \
           math.ceil(len(train_samples) / batch_size), math.ceil(len(valid_samples) / batch_size)


# Generator to load data for model training
def input_generator(data, batch_size=32, is_for_validation=False):
    log = logging.getLogger(__name__)
    data_length = len(data)
    while 1:
        for offset in range(0, data_length, batch_size):
            samples = data[offset:offset + batch_size]
            # log.debug('Batch offset:{}, len:{}'.format(offset, len(samples)))
            t = time.time()

            x_data, y_data = load_samples(samples, is_for_validation)

            log.debug('Data batch ready in {}, actual len={}, offset={}'.format(time.time() - t, y_data.shape[0], offset))
            yield x_data, y_data


# Convenience method used to load batch of data either by generator or as a whole into memory.
# samples - rows from csv file containing training information.
# returns: tuple of (x_data, y_data) - image data and steering angles
def load_samples(samples, only_main_image=False):
    log = logging.getLogger(__name__)
    x_data = []
    y_data = []
    for row in samples:
        s_angle = float(row[3])
        im_path, l_im_path, r_im_path = row[0], row[1], row[2]

        if (abs(s_angle) < 0.05 and rnd.random() <= 0.98):
            # or (abs(s_angle) < 0.2 and rnd.random() <= 0.85):

            continue

        load_image(im_path, s_angle, x_data, y_data, main_only=only_main_image, flip_probability=1.0)

        if only_main_image: continue #skip side images

        load_image(l_im_path, f_angle(s_angle, 1), x_data, y_data, flip_probability=0.7)
        load_image(r_im_path, f_angle(s_angle, -1), x_data, y_data, flip_probability=0.7)
    log.debug('Processing image files: converting X...')
    # Normalize them
    x_data = np.array(x_data)
    log.debug('X shape {}'.format(x_data.shape))
    log.debug('Processing image files: Y...')
    y_data = np.array(y_data)
    x_data, y_data = sk_shuffle(x_data, y_data)
    # log.debug(x_data[10])
    # log.debug(y_data[:20])
    return x_data, y_data


def f_angle (angle, direction):
    a = angle + direction * 0.25
    if a > 1:
        return 1
    elif a < -1:
        return -1
    return a


# Read image file via provided path
# im_path - path to a file to load
# s_angle - streering angle adjustment for image. Usage for side camera images to correct the steering.
# x_data - image data
# y_data - steering angles for corresponding images
# main_only - load only main image without augmentations
def load_image(im_path, s_angle, x_data, y_data, flip_probability=1.0, main_only=True):

    path = im_path[im_path.rfind('/'):]
    im = cv2.imread('./data/IMG' + path)
    # im = cv2.resize(im, (224,224,3), interpolation=cv2.INTER_AREA)

    # process image
    x_data.append(im)
    y_data.append(s_angle)

    if main_only: return

    if rnd.random() < flip_probability:
        # add flipped images to avoid one side bias
        y_data.append(-s_angle)
        x_data.append(np.fliplr(im))


    # augment image with brightness

    if s_angle > 0.4:
        add_augmented_brightness(im, s_angle, x_data, y_data, 0.5)
        add_augmented_brightness(im, s_angle, x_data, y_data, 0.5)


def add_augmented_brightness(im, s_angle, x_data, y_data, flip_probability=0.5):
    """
    Augments im brightness.
    Also performs randomisation on whether to select original image or horizontaly flipped.
    Brightness is randomized between 40 to 90 along with random sign (+/-).

    im - image to augment
    s_angel - steering angle
    x_data - array of image data to append result to
    y_data - array of steering angles to add result to
    :param flip_probability: how often to use flipped image
    """
    # if rnd.random() < 0.1: return # skip in 20% cases

    b_low = 80
    b_high = 150

    aug_image = im
    if rnd.random() < flip_probability:
        aug_image = np.fliplr(im)
        y_data.append(-s_angle)
    else:
        y_data.append(s_angle)

    b = rnd.randint(b_low, b_high) * (rnd.randint(0, 1) * 2 - 1)
    x_data.append(cv2.convertScaleAbs(aug_image, alpha=1, beta=b))


# Execute when ran from terminal

if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('epochs', type=int, default=3, help='Amount of epochs to run training for')
    parser.add_argument('--model', nargs='?', const='', default='', help='Model file to use as a start')
    parser.add_argument('--label', nargs='?', const='', default='', help='Optional name to be added to output files')
    parser.add_argument('-d', action='store_true', help='Debug mode')

    args = parser.parse_args()
    epochs_count = args.epochs

    if args.d:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)

    # with generators
    train_gen, valid_gen, train_steps, valid_steps = create_data_generators(128)
    log.info('Train set len={}, valid_len={}'.format(train_steps, valid_steps))

    if len(args.model) > 0:
        log.info('Loading existing model from {}'.format(args.model))
        model = load_model(args.model, compile=False)
    else:
        log.info('Creating a new model configuration')
        model = drive_model_create()# create_vgg()
    train_in_driving_with_generator(train_gen, valid_gen, train_steps, valid_steps, model, epochs_count,
                                    label_name=args.label)
    # plot_model(model, to_file='model.png')


    # without generator
    # model = drive_model_create()
    # x_data, y_data = read_all_data_to_memory()
    # train_in_driving(x_data, y_data, model, epochs_count)
