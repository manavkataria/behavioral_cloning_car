# Behavioral Cloning
Using Keras to train a deep neural network for predicting steering angles based on camera input. Trained on a Unity3D simulator.

### TLDR; Watch the Video - Gone in 60 Seconds!
[![Youtube Video](https://cloud.githubusercontent.com/assets/2206789/22684201/316bb47c-ecd0-11e6-92c7-66eb5790d286.jpg)](https://goo.gl/zhD2jV)

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

<!--
[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Sleazy Corner"
[image4]: ./examples/placeholder_small.png "Whiskey Lake" -->

---
### Files
My project includes the following files:
* `model.py` - the script to create and train the model
* `drive.py` - for driving the car in autonomous mode
* `model.h5` - a trained convolution neural network
* `utils.py` - shared utils across module
* `settings.py` - settings shared across module
* `plotter.py` - to plot histograms, predictions, loss function, etc.
* `run.sh` - cleanup, train, validate, plot/visualize
* `install.sh` - install dependencies
* `README.md` - description of the development process (this file)
* Udacity Dataset [Download here](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip) - Track1 Dataset Used for Training
* Unity3D Simulator - [Download for MacOS](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5894ecbd_beta-simulator-mac/beta-simulator-mac.zip)

Repository includes all required files and can be used to run the simulator in autonomous mode.

### Code Quality
#### Functional Code
Using the Udacity provided simulator and my `drive.py` file, the car can be driven autonomously around the track by executing
```
$ python drive.py model.h5
```

#### Comments inline with code

The `model.py` file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model. It contains **detailed comments** to explain how the code works.

### Model Architecture and Training Strategy

#### Architecture: nVidia End-to-End Deep Learning Network

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 [`model.py`](https://github.com/manavkataria/behavioral_cloning_car/blob/master/model.py#L83-L105)

The model includes `ReLU` layers to introduce nonlinearity.
![model](https://cloud.githubusercontent.com/assets/2206789/22705301/dcde791c-ed1f-11e6-9354-3fd2d59aeb80.jpg)

##### Objective, Loss function and Hyper-Parameter tuning

I used `Mean Squared Error` as a loss metric. It seems reasonably appropriate to train the model to follow the training steering angles to some close enough extent. It was important to not let the loss function go very close to zero. A very low loss indicates memorization and thus overfitting.

The optimization space was non-linear. Hence there were instances where the model training would not converge. Retraining reinitializes the weights randomly and provides another shot at convergence. The model used an [`Adam`](https://github.com/manavkataria/behavioral_cloning_car/blob/master/model.py#L152) optimizer; the learning rate was not tuned manually. I did use Stochastic Gradient Descent initially but Adam is proven to be a better choice for most cases.

#### Controlling Overfitting

The model contains [`Dropout`(https://github.com/manavkataria/behavioral_cloning_car/blob/master/model.py#L97) layers] in order to reduce overfitting. The Dropout was [set to 10%](https://github.com/manavkataria/behavioral_cloning_car/blob/master/model.py#L59)

The loss function and predictions were carefully monitored to ensure the loss doesn't go too low and the predictions aren't a perfect match. This ensures the model isn't overfitted. The model was ultimately tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### Image Preprocessing

The Image data is [preprocessed](https://github.com/manavkataria/behavioral_cloning_car/blob/master/utils.py#L49-L56) in the model using the following techniques:
* RGB to Grayscale
* Crop to ROI Quadilateral
* Resize to a smaller `WIDTH = 200`, `HEIGHT = 66`
* Dynamic Range Adjustment

#### Steering Angle Preprocessing
The steering angles were scaled up to [`STEERING_MULTIPLIER = 100`](https://github.com/manavkataria/behavioral_cloning_car/blob/master/settings.py#L10).



### Training Strategy

#### MVP Solution: Lean Design Approach

##### Building an Overfitted Model with Minimal Data
The overall strategy for deriving a model architecture was to initially overfit the model on three critical images before building a regualarized model for the entire track. This saves time and validates the approach.

I used the following three images, twice for each training (3x2 = 6 per set) and validation set (50% validation split). Thus making a total of 12 images in the initial dataset.

**Recovery: Extreme Left of Lane**

![center_2017_01_16_18_49_00_738](https://cloud.githubusercontent.com/assets/2206789/22707096/80e205fa-ed26-11e6-8f2d-114353c31d54.jpg)

**Drive Straight: Center of Lane**

![center_2017_01_16_18_49_02_100](https://cloud.githubusercontent.com/assets/2206789/22707103/885d4f4c-ed26-11e6-8525-5d1a78465959.jpg)

**Recovery: Extreme Right of Lane**

![center_2017_01_16_18_49_04_959](https://cloud.githubusercontent.com/assets/2206789/22707162/b5e5e5e6-ed26-11e6-829c-f119ac7ceafd.jpg)

**I ran this for 30 epochs to achieve satisfactory loss convergence.**

![Loss Function 30 Epochs](https://cloud.githubusercontent.com/assets/2206789/22679857/e079a576-ecb9-11e6-9e0a-f601946d25aa.png)

**The predictions came out close but not extremely overfitted, which is ideal!**

![Predictions for 12 Frames](https://cloud.githubusercontent.com/assets/2206789/22707512/da290d60-ed27-11e6-8b18-81d697aa34c7.png)

#### Building a Regularized Model
The next step was to run the model on the entire training dataset (full track). The provided Udacity dataset had 8k images. The label distribution was quite Asymmetric and Unbalanced [Histogram: Asymmetric and Unbalanced]. I used [Horizontal Flipping](https://github.com/manavkataria/behavioral_cloning_car/blob/master/model.py#L141-L147) to Make this symmetric [Histogram: Symmetric But Unbalanced]. And lastly, [Histogram Equalization](https://github.com/manavkataria/behavioral_cloning_car/blob/master/model.py#L344-L350) for achieving balance in the training dataset [Histogram Equalization: Symmetric and Balanced].

**Raw Data Histogram: Asymmetric and Unbalanced**

![Histogram: Asymmetric and Unbalanced](https://cloud.githubusercontent.com/assets/2206789/22631651/24bdcd60-ebc6-11e6-98f6-46f0cc1e926e.png)

**Horizontally Flipped Data Histogram: Symmetric But Unbalanced**

![Histogram: Symmetric But Unbalanced](https://cloud.githubusercontent.com/assets/2206789/22631698/e0e85e38-ebc6-11e6-94cb-05f7739f188d.png)

**Fully Processed with Histogram Equalization: Symmetric and Balanced**

![Histogram Equalization: Symmetric and Balanced](https://cloud.githubusercontent.com/assets/2206789/22631705/f71f9f5e-ebc6-11e6-9315-c4fc5f3048ab.png)

**Loss Function 5 Epochs**

![Loss Function 5 Epochs](https://cloud.githubusercontent.com/assets/2206789/22679803/a52b2972-ecb9-11e6-83c1-c8b1eb066999.png)

Finally, the balanced dataset was [randomly shuffled](https://github.com/manavkataria/behavioral_cloning_car/blob/master/model.py#L366) before being fed into the model.

Once the dataset was balanced, the vehicle is able to drive autonomously around the track without leaving the road.

**Predictions for 120 Frames**

![Predictions for 120 Frames](https://cloud.githubusercontent.com/assets/2206789/22679856/e07909f4-ecb9-11e6-9840-46e91d3b68ca.png)

### Acknowledgements & References
* Sagar Bhokre for project skeleton and constant support
* Caleb Kirksey for his excellent company during the grind
* Mohan Karthik for an informative blogpost motivating dataset balancing
* Paul Hearty - for valuable [project tips](https://carnd-forums.udacity.com/questions/26214464/behavioral-cloning-cheatsheet) provided on Udacity forums that saved time. Especially the MVP: overfitting idea
* Udacity's [Project Rubric](https://review.udacity.com/#!/rubrics/432/view) coz its good have them listed here.

<!-- #### Common Questions Addressed via Slack:
TODO -->
