#!/usr/bin/env pythonw
import matplotlib
import matplotlib.pyplot as plt
import pickle

from model import Model

# matplotlib.use('TkAgg')  # MacOSX Compatibility
matplotlib.interactive(True)

class ModelPlotter(object):

    @classmethod
    def plot_metrics(cls, history):
        # print(history.keys())
        # summarize history for accuracy
        plt.plot(history['acc'])
        plt.plot(history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper right')
        plt.show()
        # summarize history for loss
        plt.plot(history['loss'])
        plt.plot(history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper right')
        # plt.show()
        plt.savefig('save/history')
        plt.close()

    @classmethod
    def plot_predictions(cls, model, x, y):
        orig_steer = []
        pred_steering_angle = []
        orig_steer = y.tolist()

        # import ipdb; ipdb.set_trace()
        for image in x:
            pred_steering_angle.append(float(model.predict(image[None, :, :, :], batch_size=1)))

        print(len(orig_steer), len(pred_steering_angle))
        plt.plot(orig_steer)
        plt.plot(pred_steering_angle)
        plt.xlabel('frame')
        plt.ylim([-15, 15])
        plt.ylabel('steering angle')
        plt.legend(['original', 'predicted'], loc='upper right')
        # plt.show()
        plt.savefig('save/predictions')



def main():
    """ Plot Model Metrics & Predictions """
    model_filename = "save/model.json"
    model_filename = model_filename[:-5]

    # Load Model & data
    model = Model(model_filename)

    model = model.model

    # Plot
    history, x, y = pickle.load(open('save/hist_xy.p', 'rb'))
    ModelPlotter.plot_metrics(history)
    ModelPlotter.plot_predictions(model, x, y)

    import gc; gc.collect()

if __name__ == '__main__':
    main()
