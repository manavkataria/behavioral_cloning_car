#!/usr/bin/env pythonw
import matplotlib
import matplotlib.pyplot as plt
import pickle

from model import Model
from sklearn.utils import shuffle

# matplotlib.use('TkAgg')  # MacOSX Compatibility
# matplotlib.interactive(True)

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
        plt.xlabel('frame')
        # plt.ylim([-15, 15])
        plt.ylabel('steering angle')
        plt.legend(['original', 'predicted'])
        # plt.show()
        plt.grid(b=True, which='both')
        plt.savefig('save/predictions')
        plt.close()


def main():
    """ Plot Model Metrics & Predictions """
    model_filename = "save/model.json"
    model_filename = model_filename[:-5]

    # Load Model & data
    model = Model(model_filename)
    model = model.model

    history, X_train, y_train = pickle.load(open('save/hist_xy.p', 'rb'))
    X_train, y_train = shuffle(X_train, y_train)

    # Print Predictions:
    predictions = model.model.predict_on_batch(X_train)
    for i in range(len(predictions)):
        print("Prediction[{i}]: ({diff}) = ({pred}) - ({y_train})".format(
               i=i,
               diff=int(predictions[i][0] - y_train[i]),
               pred=int(predictions[i][0]),
               y_train=int(y_train[i])))

    # Plot
    ModelPlotter.plot_metrics(history)
    ModelPlotter.plot_predictions(model, X_train, y_train)

    import gc; gc.collect()  # Suppress a Keras Tensorflow Bug

if __name__ == '__main__':
    main()
