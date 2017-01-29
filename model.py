#!/usr/bin/env ipython
import os
import cv2
import numpy as np
import pickle
import random

from sklearn.utils import shuffle
from keras.layers import Dense  # Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Dropout, Flatten
from keras.models import load_model, model_from_json, Sequential
from keras.optimizers import Adam, SGD
# from keras.regularizers import activity_l2,  l2
from keras.utils.visualize_util import plot

# Settings
DEBUG = True
DISPLAY_IMAGES = True
BATCH_SIZE = 1
NUM_EPOCHS = 2
TRAINING_PORTION = 1
HZFLIP = False

# Steering Miltiplier
STEERING_MULTIPLIER = 100

# Image Dimensions
WIDTH = 200
HEIGHT = 66
DEPTH = 1

# Image ROI Crop Percentage from [left, top, right, bottom]
ROI_bbox = [0.0, 0.40, 0.0, 0.13]

# Model Regularization
DROPOUT = 0.1

# Training Data
# DATA_DIR = "training/data/"               # Udacity Data
DATA_DIR = "training/minimal/"              # Left, Center, Right
DRIVING_LOG = DATA_DIR + "driving_log.csv"


def cut_ROI_bbox(image_data):
    w = image_data.shape[1]
    h = image_data.shape[0]
    x1 = int(w * ROI_bbox[0])
    x2 = int(w * (1 - ROI_bbox[2]))
    y1 = int(h * ROI_bbox[1])
    y2 = int(h * (1 - ROI_bbox[3]))
    ROI_data = image_data[y1:y2, x1:x2]
    return ROI_data


def rgb_to_grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def normalize_grayscale(imgray):
    a = -0.5
    b = 0.5
    grayscale_min = 0
    grayscale_max = 255
    return a + ( ( (imgray - grayscale_min)*(b - a) )/( grayscale_max - grayscale_min ) )


def preprocess_image(image):
    """ Resize and Crop Image """
    gray = rgb_to_grayscale(image)
    cropped = cut_ROI_bbox(gray)
    resized = cv2.resize(cropped, (WIDTH, HEIGHT))
    normalized = normalize_grayscale(resized)
    reshaped = normalized.reshape(HEIGHT, WIDTH, DEPTH)
    return reshaped


class Model(object):

    def __init__(self, filename):
        # Model Definition
        self.filename = filename
        if os.path.isfile(filename + '.h5'):
            if DEBUG: print ("Loading Model:", filename + '.h5')
            self.model = self.load()
        else:
            self.model = self._define_model()

    def load(self):
        return self.load_model_with_weights(self.filename)

    def _define_model(self):
        """
            nVidia End to End Learning Model
        """

        self.model = Sequential()
        self.model.add(Convolution2D(24, 5, 5, name='Conv1', init='glorot_normal', subsample=(2, 2), input_shape=(HEIGHT, WIDTH, DEPTH), activation='relu'))
        self.model.add(Convolution2D(36, 5, 5, name='Conv2', init='glorot_normal', subsample=(2, 2), activation='relu'))
        self.model.add(Convolution2D(48, 5, 5, name='Conv3', init='glorot_normal', subsample=(2, 2), activation='relu'))
        self.model.add(Convolution2D(64, 3, 3, name='Conv4', init='glorot_normal', subsample=(1, 1), activation='relu'))
        self.model.add(Convolution2D(64, 3, 3, name='Conv5', init='glorot_normal', subsample=(1, 1), activation='relu'))

        self.model.add(Flatten(name='Flatten'))
        self.model.add(Dense(1164, name='Dense1'))
        # self.model.add(Dropout(DROPOUT))
        self.model.add(Dense(100, name='Dense2', activation='relu'))
        # self.model.add(Dropout(DROPOUT))
        self.model.add(Dense(50, name='Dense3', activation='relu'))
        # self.model.add(Dropout(DROPOUT))
        self.model.add(Dense(10, name='Dense4', activation='relu'))
        self.model.add(Dense(1, name='Dense5'))

        return self.model

    # def create_random_data(batches):
    #     x = np.random.random((BATCH_SIZE * batches, HEIGHT, WIDTH, DEPTH))
    #     y = np.random.randint(2, size=(BATCH_SIZE * batches, 1))
    #     return x, y

    def read_csv(self, filename):
        self.lines = []
        with open(filename, 'r') as dbfile:
            self.lines = dbfile.readlines()
        return self.lines

    def rows_to_feature_labels(self, count, hzflip=False):
        # Allocate twice the size to accomodate
        # both original and horizontally flipped images
        size_multiple = 1
        if HZFLIP:
            size_multiple = 2

        x = np.empty((size_multiple * count, HEIGHT, WIDTH, DEPTH), dtype=np.float32)
        y = np.empty((size_multiple * count), dtype=np.float32)

        for idx, line in enumerate(self.lines[:count]):
            [center, left, right, steering, throttle, breaks, speed] = line.split(',')

            # Adding Random Perturbations between [-5, 5] to each input
            steering = float(steering) * STEERING_MULTIPLIER + random.randint(-50, 50) / 10.0
            if DEBUG: print(idx, steering)

            image_data = cv2.imread(DATA_DIR + center)
            message = 'Raw Input: {:.2}'.format(steering)
            display_images(image_data, message)

            image_data = preprocess_image(image_data)
            x[idx, :, :, :] = np.copy(image_data)
            y[idx] = np.copy(steering)

        return x, y

    def set_optimizer(self):
        optimizer = Adam()
        # optimizer = SGD(lr=0.0001)
        self.model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_squared_error'])

    def train(self, x, y):
        history = self.model.fit(x, y, nb_epoch=NUM_EPOCHS, batch_size=BATCH_SIZE, shuffle=True,
                                 validation_split=0.5)
        return history

    def train_generator(self, generator):
        history = self.model.fit(generator, nb_epoch=NUM_EPOCHS, batch_size=BATCH_SIZE, shuffle=True,
                                 validation_split=0.5)
        return history

    def save_model_to_json_file(self, filename):
        json_string = self.model.to_json()
        with open(filename + '.json', 'w') as jfile:
            jfile.write(json_string)

    def save_model_weights(self, filename):
        self.model.save(filename + '.h5')

    def load_model_from_json(self, filename):
        with open(filename + '.json', 'r') as jfile:
            json_string = jfile.read()

        self.model = model_from_json(json_string)
        return self.model

    def load_model_with_weights(self, filename):
        return load_model(filename + '.h5')

    def plot_model_to_file(self, filename):
        plot(self.model, show_shapes=True, to_file=filename + '.jpg')

    def show_model_from_image(self, filename):
        model_image = cv2.imread(filename + ".jpg")
        cv2.imshow("model", model_image)
        cv2.waitKey(0)


def display_images(image_features, message=None, delay=500):
    if not DISPLAY_IMAGES: return
    font = cv2.FONT_HERSHEY_SIMPLEX
    WHITE = (255, 255, 255)
    FONT_THICKNESS = 1
    # FONT_SCALE = 4

    if image_features.ndim == 3:
        image_features = [image_features]

    height, width, depth = image_features[0].shape

    for image in image_features:
        image = np.copy(image)  # Avoid Overwriting Original Image
        if message:
            text_position = (int(width * 0.05), int(height * 0.95))
            cv2.putText(image, message, text_position, font, FONT_THICKNESS, WHITE)
        cv2.imshow(message, image)
        cv2.waitKey(delay)


def main():
    model_filename = "save/model.json"
    model_filename = model_filename[:-5]
    model = Model(model_filename)

    # model.plot_model_to_file(model_filename)
    # model.show_model_from_image(model_filename)

    rows = model.read_csv(DRIVING_LOG)
    if DEBUG: print ("Database size: {}".format(len(rows)))

    n_train = int(len(rows) * TRAINING_PORTION)
    # x, y = model.rows_to_feature_labels(3, hzflip=HZFLIP)
    X_train, y_train = model.rows_to_feature_labels(n_train, hzflip=HZFLIP)
    X_train, y_train = shuffle(X_train, y_train, random_state=1)


    if HZFLIP:
        for i in range(n_train):
            X_train[n_train + i, :, :, :] = cv2.flip(X_train[i, :, :, :], 1)
            y_train[n_train + i] = -1.0 * y_train[i]

    display_images(X_train, "ROI Input")

    model.set_optimizer()
    history = model.train(X_train, y_train)
    model.save_model_to_json_file(model_filename)
    model.save_model_weights(model_filename)

    predictions = model.model.predict_on_batch(X_train)
    print("Predictions:", list(zip(predictions, y_train, predictions-y_train)))

    # Pickle Dump
    pickle.dump([history.history, X_train, y_train], open('save/hist_xy.p', 'wb'))


if __name__ == '__main__':
    main()
