#!/usr/bin/env pythonw
# import matplotlib
import matplotlib.pyplot as plt
import pickle

from model import Model
from sklearn.utils import shuffle

import utils
# matplotlib.use('TkAgg')  # MacOSX Compatibility
# matplotlib.interactive(True)

ENABLE_HISTOGRAM = True


class ModelPlotter(object):

    @classmethod
    def plot_metrics(cls, history):
        # print(history.keys())
        # summarize history for MSE
        plt.plot(history['loss'])
        plt.plot(history['val_loss'])
        plt.title('Model Loss Function: Mean Squared Error')
        plt.ylabel('Loss (Mean Squared Error)')
        plt.xlabel('Epoch')
        plt.legend(['training', 'validation'])
        # plt.show()
        plt.grid(b=True, which='both')
        plt.savefig('save/loss')
        plt.close()

    @classmethod
    def plot_predictions(cls, model, x, y):
        orig_steer = []
        pred_steering_angle = []
        orig_steer = y.tolist()

        # import ipdb; ipdb.set_trace()
        for image in x:
            pred_steering_angle.append(float(model.predict(image[None, :, :, :], batch_size=1)))

        # print(len(orig_steer), len(pred_steering_angle))
        plt.title('Predictions')
        plt.plot(orig_steer)
        plt.plot(pred_steering_angle)
        plt.xlabel('Frame')
        # plt.ylim([-15, 15])
        plt.ylabel('Steering Angle')
        plt.legend(['Original', 'Predicted'])
        # plt.show()
        plt.grid(b=True, which='both')
        plt.savefig('save/predictions')
        plt.close()


def visualize_histogram(y_train, msg, filename_suffix, bins='auto'):
    # histogram, bin_edges =
    counts, bin_edges, _ = plt.hist(y_train, bins=bins)
    plt.title('Histogram of {} Labels'.format(msg))
    plt.grid(b=True, which='both')
    plt.savefig('save/histogram_' + filename_suffix)
    plt.close()
    return counts, bin_edges


def main():
    """ Plot Model Metrics & Predictions """
    model_filename = "save/model.h5"

    # Load Model & data
    model = Model(model_filename)
    model = model.model

    history, X_balanced, y_balanced, y_train = pickle.load(open('save/hist_xy.p', 'rb'))
    X_balanced, y_balanced = shuffle(X_balanced, y_balanced)

    if ENABLE_HISTOGRAM:
        _, bin_edges = visualize_histogram(y_train, 'Unfiltered', 'unfiltered')
        visualize_histogram(y_balanced, 'Balanced', 'Balanced', bin_edges)

        # Zero-Penalized Histogram Series
        # filterd_ys = history.get('filtered_y', None)
        # for i, filtered_y in enumerate(filterd_ys):
        #     msg = 'filtered_' + str(i)
        #     visualize_histogram(filtered_y, msg, msg)
        # return

    # Metrics
    ModelPlotter.plot_metrics(history)
    # Predictions
    # utils.print_predictions(model.model, X_balanced, y_balanced)
    ModelPlotter.plot_predictions(model, X_balanced, y_balanced)

    import gc; gc.collect()  # Suppress a Keras Tensorflow Bug


if __name__ == '__main__':
    main()
