#!/usr/bin/env python
import os
import cv2
import matplotlib
import numpy as np

from keras.layers import Dense  # Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Dropout, Flatten
from keras.models import load_model, model_from_json, Sequential
from keras.optimizers import Adam  # SGD
# from keras.regularizers import activity_l2,  l2
from keras.utils.visualize_util import plot

matplotlib.use('TkAgg')  # MacOSX Compatibility
import matplotlib.pyplot as plt

# Settings
BATCH_SIZE = 32
NUM_EPOCHS = 1
LEARNING_RATE = 0.0001
MOMENTUM = 0.01
REGULARIZE_PARAM = 0.1
DECAY = 0.0

WIDTH = 200
HEIGHT = 66
DEPTH = 3

DROPOUT = 0.1

DRIVING_LOG = "training/minimal/driving_log.csv"

TRAINING_PORTION = 1.0
SHOW_DATA = False

# percentage of data to be cut from left, top, right, bottom
ROI_bbox = [0.0, 0.40, 0.0, 0.13]


def cut_ROI_bbox(image_data):
    w = image_data.shape[1]
    h = image_data.shape[0]
    x1 = int(w * ROI_bbox[0])
    x2 = int(w * (1 - ROI_bbox[2]))
    y1 = int(h * ROI_bbox[1])
    y2 = int(h * (1 - ROI_bbox[3]))
    ROI_data = image_data[y1:y2, x1:x2]
    return ROI_data


def preprocess_image(image):
    """ Resize and Crop Image """
    ROI_data = cut_ROI_bbox(image)
    processed_data = cv2.resize(ROI_data, (WIDTH, HEIGHT))
    return processed_data


class Model(object):

    def __init__(self, filename):
        # Model Definition
        self.filename = filename
        if os.path.isfile(filename + '.h5'):
            self.model = self.load(filename)
        else:
            self.model = self._define_model()

        # Data Definition
        self.y = list()
        self.x = np.empty((BATCH_SIZE, HEIGHT, WIDTH, DEPTH), dtype=np.uint8)

        # Bookkeeping?
        self.curr_id = 0

    def load(self):
        self.model = self.load_model_with_weights(self.filename)

    def _define_model(self):
        """
            nVidia End to End Learning Model
        """

        conv_kernal = 5
        model = Sequential()
        model.add(Convolution2D(24, 5, 5, name='Conv1', border_mode='same', subsample=(2, 2), input_shape=(HEIGHT, WIDTH, DEPTH), activation='relu'))
        model.add(Convolution2D(36, 5, 5, name='Conv2', border_mode='same', subsample=(2, 2), activation='relu'))
        model.add(Convolution2D(48, 5, 5, name='Conv3', border_mode='same', subsample=(2, 2), activation='relu'))
        model.add(Convolution2D(64, 3, 3, name='Conv4', border_mode='same', subsample=(1, 1), activation='relu'))
        model.add(Convolution2D(64, 3, 3, name='Conv5', border_mode='same', subsample=(1, 1), activation='relu'))

        model.add(Flatten(name='Flatten'))
        model.add(Dense(1164, name='Dense1'))
        # model.add(Dropout(DROPOUT))
        model.add(Dense(100, name='Dense2', activation='relu'))
        # model.add(Dropout(DROPOUT))
        model.add(Dense(50, name='Dense3', activation='relu'))
        # model.add(Dropout(DROPOUT))
        model.add(Dense(10, name='Dense4', activation='relu'))
        model.add(Dense(1, name='Dense5', activation='softmax'))

        return model

    def create_random_data(batches):
        x = np.random.random((BATCH_SIZE * batches, HEIGHT, WIDTH, DEPTH))
        y = np.random.randint(2, size=(BATCH_SIZE * batches, 1))
        return x, y

    def read_csv(self, filename):
        self.lines = []
        with open(filename, 'r') as dbfile:
            self.lines = dbfile.readlines()
            # lines.append(dbfile.readlines())
        return self.lines

    def rows_to_feature_labels(self, count, rows, horizontal_flip=True):
        # Allocate twice the size to accomodate
        # both original and horizontally flipped images
        size_multiple = 1
        if horizontal_flip:
            size_multiple = 2

        x = np.empty((size_multiple * count, HEIGHT, WIDTH, DEPTH), dtype=np.uint8)
        y = np.empty((size_multiple * count), dtype=np.float16)

        for idx, line in enumerate(self.lines[self.curr_id:self.curr_id + count]):
            [c_f, l_f, r_f, steering, throttle, breaks, speed] = line.split(',')
            print(steering, c_f)

            image_data = cv2.imread(c_f)
            display_images(image_data, "Raw Input")
            image_data = preprocess_image(image_data)

            self.curr_id += 1
            if(self.curr_id > len(self.lines)):
                self.curr_id = 0

            x[idx, :, :, :] = np.copy(image_data)
            y[idx] = np.copy(float(steering))
        return x, y

    def set_optimizer_params(self, learning_rate):
        Adam_optimizer = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=DECAY)
        self.model.compile(loss='mean_squared_error', optimizer=Adam_optimizer, metrics=['accuracy'])

    def start_training(self, x, y):
        # sgd = SGD(lr=LEARNING_RATE) #, momentum=MOMENTUM, decay=DECAY, nesterov=False)
        # self.model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])
        hist = self.model.fit(x, y, nb_epoch=NUM_EPOCHS, batch_size=BATCH_SIZE, shuffle=True,
                              validation_split=0.2)
        return hist

    # def train_model(self, X_train, y_train):
    #     h = self.model.fit(X_train, y_train,
    #                        nb_epoch = 1, verbose=0,
    #                        batch_size=training_batch_size)
    #     self.model.save_weights(checkpoint_filename)
    #     print('loss : ',h.history['loss'][-1])
    #     return model

    def train_with_input(self, x_in, y_in):
        curr_id = len(self.y)
        self.x[curr_id, :, :, :] = x_in
        self.y.append(y_in)
        if len(self.y) == BATCH_SIZE:
            y = np.array(self.y)
            # x = np.array(self.x)
            print("lr: ", self.model.optimizer.get_config()['lr'])
            hist = self.model.fit(self.x, y, nb_epoch=1, batch_size=BATCH_SIZE, verbose=1, validation_data=(self.x, self.y))
            self.y = []
            return hist
        else:
            return None

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

    def plot_metrics(self, history):
        # import pdb; pdb.set_trace()
        print(history.history.keys())
        # summarize history for accuracy
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper right')
        plt.show()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper right')
        plt.show()

    def plot_predictions(self, x, y):
        orig_steer = []
        pred_steering_angle = []
        orig_steer = y.tolist()

        for image in x:
            pred_steering_angle.append(float(self.model.predict(image[None, :, :, :], batch_size=1)))

        print(len(orig_steer), len(pred_steering_angle))
        plt.plot(orig_steer)
        plt.plot(pred_steering_angle)
        plt.xlabel('frame')
        plt.ylim([-15, 15])
        plt.ylabel('steering angle')
        plt.legend(['original', 'predicted'], loc='upper right')
        plt.show()


def display_images(image_features, message=None, delay=500):
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
    model_filename = "model.json"
    model_filename = model_filename[:-5]
    model = Model(model_filename)

    # model.plot_model_to_file(model_filename)
    # model.show_model_from_image(model_filename)
    # import ipdb; ipdb.set_trace()

    rows = model.read_csv(DRIVING_LOG)
    # print ("Database size: {}".format(len(rows)))

    horizontal_flip = False

    n_train = int(len(rows) * TRAINING_PORTION)
    x, y = model.rows_to_feature_labels(n_train, rows, horizontal_flip=horizontal_flip)

    if horizontal_flip:
        for i in range(n_train):
            x[n_train + i, :, :, :] = cv2.flip(x[i, :, :, :], 1)
            y[n_train + i] = -1.0 * y[i]

    # if SHOW_DATA:
    display_images(x, "ROI Input")

    # model.set_optimizer_params(LEARNING_RATE)
    # hist = model.start_training(x, y)
    # model.save_model_to_json_file(model_filename)
    # model.save_model_weights(model_filename)
    # #model.plot_metrics(hist)
    # model.plot_predictions(x, y)


if __name__ == '__main__':
    main()
