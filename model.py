#!/usr/bin/env ipython
import os
import cv2
import numpy as np
import pickle
import random

from tqdm import tqdm
from sklearn.utils import shuffle
from keras.layers import Dense
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Dropout, Flatten
from keras.models import load_model, Sequential
from keras.optimizers import Adam
from keras.utils.visualize_util import plot
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import History


from utils import (display_images, print_predictions, preprocess_image)
from settings import (HEIGHT, WIDTH, DEPTH,
                      STEERING_MULTIPLIER,)

# Next Steps:
# 1. Use fit_generator (done); Manual fit_generator (done)
# 2. Horizontal Flipping (Done)
# 3. Data Augmentation (Done, via ImageDataGenerator)
# 4. Balancing Input Dataset using binning (Done)

# Settings
DEBUG = False
BATCH_SIZE = 1024
NUM_EPOCHS = 5
TRAINING_PORTION = 1
TRAINING_ENABLE = True
PREDICTION_ENABLE = False

# Features
FIT_GENERATOR_ENABLE = True
MANUAL_FIT_ENABLED = True
# Dataset Balancing
ZERO_PENALIZING = False
DESIRED_DATASET_SIZE = 1024 * 32

# Training Data
# DATA_DIR = "training/track1/sand-corner1/"               # Sand-Corner1
# VALIDATION_SPLIT = 0.5
DATA_DIR = "training/data/"               # Udacity Data
VALIDATION_SPLIT = 0.25
# DATA_DIR = "training/minimal/"            # Left, Center, Right
# VALIDATION_SPLIT = 0.5
DRIVING_LOG = DATA_DIR + "driving_log.csv"

# OpenCV Flip Type for Horizontal Flipping
CV_FLIPTYPE_HORIZONTAL = 1

# Model Regularization
DROPOUT = 0.1

# Data Augmentation
HZ_FLIP_ENABLE = True
HORIZONTAL_SHIFT_RANGE_PCT = 0.1
VERTICAL_SHIFT_RANGE_PCT = 0.00
SAVE_TO_DIR = DATA_DIR
SAVE_PREFIX = 'augmented_'


class Model(object):

    def __init__(self, filename):
        # Model Definition
        filename = filename
        self.filename = filename
        if os.path.isfile(filename):
            self.model = self.load_model(self.filename)
        else:
            self.model = self._define_model()

        # Zero-Penalized Filtered Labels
        if ZERO_PENALIZING: self.filtered_y = []

    def _define_model(self):
        """
            nVidia End to End Learning Model
        """

        self.model = Sequential()
        self.model.add(Convolution2D(24, 5, 5, name='Conv1', subsample=(2, 2), input_shape=(HEIGHT, WIDTH, DEPTH), activation='relu'))
        self.model.add(Convolution2D(36, 5, 5, name='Conv2', subsample=(2, 2), activation='relu'))
        self.model.add(Convolution2D(48, 5, 5, name='Conv3', subsample=(2, 2), activation='relu'))
        self.model.add(Convolution2D(64, 3, 3, name='Conv4', subsample=(1, 1), activation='relu'))
        self.model.add(Convolution2D(64, 3, 3, name='Conv5', subsample=(1, 1), activation='relu'))

        self.model.add(Flatten(name='Flatten'))
        self.model.add(Dense(1164, name='Dense1'))
        self.model.add(Dropout(DROPOUT))
        self.model.add(Dense(100, name='Dense2', activation='relu'))
        self.model.add(Dropout(DROPOUT))
        self.model.add(Dense(50, name='Dense3', activation='relu'))
        self.model.add(Dropout(DROPOUT))
        self.model.add(Dense(10, name='Dense4', activation='relu'))
        self.model.add(Dense(1, name='Dense5'))

        return self.model

    def read_csv(self, filename):
        self.lines = []
        with open(filename, 'r') as dbfile:
            self.lines = dbfile.readlines()
        return self.lines

    def rows_to_feature_labels(self, count, hzflip=False):
        # Allocate twice the size to accomodate
        # both original and horizontally flipped images
        size_multiple = 1
        if HZ_FLIP_ENABLE:
            size_multiple = 2

        x = np.empty((size_multiple * count, HEIGHT, WIDTH, DEPTH), dtype=np.float32)
        y = np.empty((size_multiple * count), dtype=np.float32)

        if DEBUG:
            print("Index: Steering")
        for idx, line in enumerate(self.lines[:count]):
            [center, left, right, steering, throttle, breaks, speed] = line.split(',')

            # Adding Random Perturbations between [-STEERING_MULTIPLIER/20, STEERING_MULTIPLIER/20] to each input
            steering = float(steering) * STEERING_MULTIPLIER + random.randint(-STEERING_MULTIPLIER / 20, STEERING_MULTIPLIER / 20)

            if DEBUG:
                print("{:^5d} -> {:>5}".format(idx, steering))

            image_data = cv2.imread(DATA_DIR + center)
            message = 'Angle: {:=+03d}'.format(int(steering))
            display_images(image_data, message)

            image_data = preprocess_image(image_data)
            x[idx, :, :, :] = np.copy(image_data)
            y[idx] = np.copy(steering)

        if hzflip:
            for idx in range(count):
                # Fill the empty end of the array
                hz_flipped_image = cv2.flip(x[idx, :, :, :],
                                            CV_FLIPTYPE_HORIZONTAL)
                x[count + idx, :, :, :] = hz_flipped_image.reshape(HEIGHT, WIDTH, DEPTH)
                y[count + idx] = -1.0 * y[idx]

        return x, y

    def set_optimizer(self):
        optimizer = Adam()
        # optimizer = SGD(lr=0.0001)
        self.model.compile(loss='mean_squared_error', optimizer=optimizer)

    def train(self, x, y):
        history = self.model.fit(x, y, nb_epoch=NUM_EPOCHS, batch_size=BATCH_SIZE, shuffle=True,
                                 validation_split=VALIDATION_SPLIT)
        return history

    def train_and_validate_with_generator(self,
                                          X_train,
                                          y_train,
                                          validation_split=VALIDATION_SPLIT,
                                          nb_epochs=NUM_EPOCHS,
                                          batch_size=BATCH_SIZE,
                                          manual=True):
        # Setup Split Parameters
        training_split = (1 - validation_split)
        samples_per_epoch_train = int(len(X_train) * training_split)
        samples_per_epoch_val = int(len(X_train) * validation_split)

        # Setup Image Generator
        data_gen_args = dict(width_shift_range=HORIZONTAL_SHIFT_RANGE_PCT,
                             height_shift_range=VERTICAL_SHIFT_RANGE_PCT)
        train_datagen = ImageDataGenerator(**data_gen_args)
        val_datagen = ImageDataGenerator(**data_gen_args)
        train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)
        val_generator = val_datagen.flow(X_train, y_train, batch_size=batch_size)

        # Fit
        if DEBUG: print("Running Fit Generator (Manual={})".format(manual))
        if not manual:
            history = self.model.fit_generator(train_generator,
                                               samples_per_epoch_train,
                                               nb_epochs,
                                               validation_data=val_generator,
                                               nb_val_samples=samples_per_epoch_val)
        else:
            history = self.fit_generator_manual(train_generator,
                                                samples_per_epoch_train,
                                                nb_epochs,
                                                val_datagen=val_generator,
                                                nb_val_samples=samples_per_epoch_val)

        return history

    def zero_penalize(self, current_epoch, datagen):
        """
        Zero-Penalizing: Killing Zeros as input labels
            Inspired by Mohan Karthik's Blog Post "Cloning a car to mimic human driving":
            https://medium.com/@mohankarthik/cloning-a-car-to-mimic-human-driving-5c2f7e8d8aff#.y5qsm32s4
        """

        if len(self.filtered_y) <= current_epoch:
            self.filtered_y.append([])

        # We start with a bias of 1.0 (allow all angles) and slowly as the epochs
        # continue, reduce the bias, thereby dropping low angles progressively
        bias = 1. / (current_epoch + 1.)

        # Define a random threshold for each image taken
        threshold = np.random.uniform()

        for X_batch, y_batch in datagen:
            # If the newly augmented angle + the bias falls below the threshold
            # then discard this angle / img combination and look again
            # FIXME(manav): This assumes BATCH_SIZE = 1
            normalized_y_value = y_batch[0] / STEERING_MULTIPLIER
            if DEBUG:
                print("normalized_y_value: {:>5.1f} <- {:>5.1f}".format(
                      normalized_y_value, y_batch[0], bias, threshold))
            if (abs(normalized_y_value) + bias) < threshold:
                if DEBUG:
                    print("Zero Penalizing: {:>5.1f} + {:>5.1f} <  {:>5.1f}".format(
                          y_batch[0], bias, threshold))
                continue
            else:
                if DEBUG:
                    print("Not  Penalizing: {:>5.1f} + {:>5.1f} >= {:>5.1f}".format(
                          y_batch[0], bias, threshold))
                self.filtered_y[current_epoch].append(y_batch[0])
                break

        return X_batch, y_batch

    def fit_generator_manual(self,
                             train_datagen,
                             nb_train_samples,
                             nb_epochs,
                             val_datagen,
                             nb_val_samples,
                             verbose=1):
        # Manual Mode
        loss, val_loss = [], []
        for e in range(nb_epochs):
            batch_loss, batch_val_loss = [], []
            train_batch_counter, val_batch_counter = 0, 0
            nb_training_batches = nb_train_samples / BATCH_SIZE
            nb_val_batches = nb_train_samples / BATCH_SIZE

            # Training
            with tqdm(total=nb_train_samples, desc='Training Samples') as pbar:
                for X_batch, y_batch in train_datagen:
                    if ZERO_PENALIZING:
                        X_batch, y_batch = self.zero_penalize(e, train_datagen)
                    batch_loss.append(self.model.train_on_batch(X_batch, y_batch))
                    pbar.update(BATCH_SIZE)
                    train_batch_counter += 1
                    if train_batch_counter >= nb_training_batches:
                        break

            # Validation
            for X_batch, y_batch in val_datagen:
                batch_val_loss.append(self.model.test_on_batch(X_batch, y_batch))
                val_batch_counter += 1
                if val_batch_counter >= nb_val_batches:
                    break

            loss.append(sum(batch_loss) / float(len(batch_loss)))
            val_loss.append(sum(batch_val_loss) / float(len(batch_val_loss)))

            if verbose == 1:
                print('Manual Fit. Epoch {:02d}/{:02d}: loss: {:>8.1f} - val_loss {:>8.1f}'.format(
                      e,
                      nb_epochs,
                      loss[e],
                      val_loss[e]))

        history = History()
        history.history = {'loss': loss, 'val_loss': val_loss}

        if ZERO_PENALIZING:
            history.history['filtered_y'] = self.filtered_y

        return history

    def save_model(self, filename):
        self.model.save(filename)

    def load_model(self, filename):
        print("Loading Model:", self.filename)
        return load_model(filename)

    def plot_model_to_file(self, filename):
        plot(self.model, show_shapes=True, to_file=filename + '.jpg')

    def show_model_from_image(self, filename):
        model_image = cv2.imread(filename + ".jpg")
        cv2.imshow("model", model_image)
        cv2.waitKey(0)


def bin_dataset(y_train):
    counts, bin_edges = np.histogram(y_train, bins='auto')
    bin_ids = np.digitize(y_train, bin_edges)
    nbins = len(bin_edges)

    # Initialize Bins
    bins = [[] for _ in range(nbins)]
    # Build Reverse Index
    for i, y in enumerate(y_train):
        rev_idx = bin_ids[i] - 1
        bins[rev_idx].append(i)

    return bins, counts, bin_edges


def construct_balanced_dataset_from_bins(X_train, y_train, bins, size=100):
    X_balanced = np.empty((size, HEIGHT, WIDTH, DEPTH), dtype=np.float32)
    y_balanced = np.empty((size,), dtype=np.float32)

    for i, idx in enumerate(random_uniform_sampling_from_bins(bins)):
        X_balanced[i, :, :, :] = X_train[idx]
        y_balanced[i] = y_train[idx]
        if i >= (size - 1):
            break

    return X_balanced, y_balanced


def random_uniform_sampling_from_bins(bins):
    while 1:
        bin_id = np.random.randint(len(bins))
        selected_bin_indices = bins[bin_id]
        bin_length = len(selected_bin_indices)
        if bin_length <= 0:
            # Pick another bin; this one being empty
            continue
        rev_idx = np.random.randint(bin_length)
        yield selected_bin_indices[rev_idx]


def balance_dataset(X_train, y_train, size=DESIRED_DATASET_SIZE):
    bins, _, bin_edges = bin_dataset(y_train)
    X_balanced, y_balanced = construct_balanced_dataset_from_bins(X_train, y_train, bins, size=size)

    for image, angle in zip(X_balanced, y_balanced):
        display_images(image, message=str(angle), delay=500)
    return X_balanced, y_balanced


def main():
    model_filename = "save/model.h5"
    model = Model(model_filename)

    # model.plot_model_to_file(model_filename)
    # model.show_model_from_image(model_filename)
    rows = model.read_csv(DRIVING_LOG)
    n_train = int(len(rows) * TRAINING_PORTION)

    if DEBUG:
        print("Using Training Dataset Size: {}".format(n_train))

    X_train, y_train = model.rows_to_feature_labels(n_train, hzflip=HZ_FLIP_ENABLE)
    X_train, y_train = shuffle(X_train, y_train, random_state=1)

    if TRAINING_ENABLE:
        X_balanced, y_balanced = balance_dataset(X_train, y_train)
        display_images(X_balanced, "ROI")

        model.set_optimizer()
        if FIT_GENERATOR_ENABLE:
            history = model.train_and_validate_with_generator(X_balanced,
                                                              y_balanced,
                                                              manual=MANUAL_FIT_ENABLED)
        else:
            history = model.train(X_balanced, y_balanced)

        model.save_model(model_filename)

    if PREDICTION_ENABLE:
        for i, image in enumerate(X_train):
            prediction = model.model.predict(image[None, :, :, :],
                                             batch_size=1)
            print("{};  {:>6.2f}".format(model.lines[i].split(',')[0],
                                         prediction[0][0] / STEERING_MULTIPLIER))
            display_images(image, str(prediction))

    # Pickle Dump
    pickle.dump([history.history, X_balanced, y_balanced, y_train], open('save/hist_xy.p', 'wb'))
    if DEBUG:
        print_predictions(model.model, X_balanced, y_balanced)

    import gc; gc.collect()  # Suppress a Keras Tensorflow Bug


if __name__ == '__main__':
    main()
