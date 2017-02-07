# Behavioral Cloning
Using Keras to train a deep neural network for predicting steering angles based on camera input. Trained on a Unity3D simulator.

### TLDR; Gone in 60 Seconds!
[![Youtube Video](https://cloud.githubusercontent.com/assets/2206789/22684201/316bb47c-ecd0-11e6-92c7-66eb5790d286.jpg)](https://goo.gl/zhD2jV)

---

** Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Sleazy Corner"
[image4]: ./examples/placeholder_small.png "Whiskey Lake"

## Rubric Points
### Here I describe the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually  

---
### Files
#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* `model.py` - the script to create and train the model
* `drive.py` - for driving the car in autonomous mode
* `model.h5` - a trained convolution neural network
* `utils.py` - shared utils across module
* `settings.py` - settings shared across module
* `plotter.py` - to plot histograms, predictions, loss function, etc.
* `run.sh` - cleanup, train, validate, plot/visualize
* `install.sh` - install dependencies
* `writeup_report.md` - summarizing the results

### Code Quality
#### 2. Functional Code
Using the Udacity provided simulator and my `drive.py` file, the car can be driven autonomously around the track by executing
```
$ python drive.py model.h5
```

#### 3. Submission code is usable and readable

The `model.py` file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model. It contains **detailed comments** to explain how the code works.

### Model Architecture and Training Strategy

#### 1. Architecture: nVidia End-to-End Deep Learning Network

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 [`model.py` lines 83-105](https://github.com/manavkataria/behavioral_cloning_car/blob/master/model.py#L83-L105)

The model includes `ReLU` layers to introduce nonlinearity.
![model](https://cloud.githubusercontent.com/assets/2206789/22705301/dcde791c-ed1f-11e6-9354-3fd2d59aeb80.jpg)

#### 2. Attempts to reduce overfitting in the model

The model contains [`Dropout`(https://github.com/manavkataria/behavioral_cloning_car/blob/master/model.py#L97) layers] in order to reduce overfitting. The Dropout was [set to 10%](https://github.com/manavkataria/behavioral_cloning_car/blob/master/model.py#L59)

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

##### Objective, Loss function and Hyper-Parameter tuning

I used `Mean Squared Error` as a loss metric. It seems reasonably appropriate to train the model to follow the training steering angles to some close enough extent. It was important to not let the loss function go very close to zero.

The optimization space was non-linear. Hence there were instances where the model training would not converge.
The model used an [`Adam`](https://github.com/manavkataria/behavioral_cloning_car/blob/master/model.py#L152) optimizer, so the learning rate was not tuned manually.

#### 4. Preprocessing

The Image data is [preprocessed](https://github.com/manavkataria/behavioral_cloning_car/blob/master/utils.py#L49-L56) in the model using the following techniques:
* RGB to Grayscale
* Crop to ROI Quadilateral
* Resize to a smaller `WIDTH = 200`, `HEIGHT = 66`
* Dynamic Range Adjustment

The steering angles were scaled up to [`STEERING_MULTIPLIER = 100`](https://github.com/manavkataria/behavioral_cloning_car/blob/master/settings.py#L10)


#### 3. Training Data: Dataset Balancing

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ...

For details about how I created the training data, see the next section.

### Training Strategy

#### 1. MVP Solution: Lean Design Approach

##### Build an Overfitted Model with Minimal Data
The overall strategy for deriving a model architecture was to overfit the model on three critical images. This saves time and validates the approach. I used each of the three images twice (3x2 = 6) each for training and validation (50% validation split). Thus making a total of 12 images in the dataset.

** Recovery: Extreme Left of Lane **
![center_2017_01_16_18_49_00_738](https://cloud.githubusercontent.com/assets/2206789/22707096/80e205fa-ed26-11e6-8f2d-114353c31d54.jpg)

** Drive Straight: Center of Lane ** ![center_2017_01_16_18_49_02_100](https://cloud.githubusercontent.com/assets/2206789/22707103/885d4f4c-ed26-11e6-8525-5d1a78465959.jpg)

** Recovery: Extreme Right of Lane ** ![center_2017_01_16_18_49_04_959](https://cloud.githubusercontent.com/assets/2206789/22707162/b5e5e5e6-ed26-11e6-829c-f119ac7ceafd.jpg)

** I ran this for 30 epochs to achieve satisfactory loss convergence. **
![Loss Function 30 Epochs](https://cloud.githubusercontent.com/assets/2206789/22679857/e079a576-ecb9-11e6-9e0a-f601946d25aa.png)

** The predictions came out close but not extremely overfitted, which is ideal! **
![Predictions for 12 Frames](https://cloud.githubusercontent.com/assets/2206789/22707512/da290d60-ed27-11e6-8b18-81d697aa34c7.png)


#### 2. Full Dataset
The next step was to run the model on the entire training dataset. The provided Udacity dataset had 8k images. The label distribution was quite Asymmetric and Unbalanced TODO:REF. I used Horizontal Flipping to Make this symmetric TODO:REF code and image. And then Histogram Equalization for achieving balanance in the training dataset.

 ** Raw Data Histogram: Asymmetric and Unbalanced **
 ![Histogram: Asymmetric and Unbalanced](https://cloud.githubusercontent.com/assets/2206789/22631651/24bdcd60-ebc6-11e6-98f6-46f0cc1e926e.png)

 ** Horizontally Flipped Data Histogram: Symmetric But Unbalanced **
 ![Histogram: Symmetric But Unbalanced](https://cloud.githubusercontent.com/assets/2206789/22631698/e0e85e38-ebc6-11e6-94cb-05f7739f188d.png)

 ** Fully Processed with Histogram Equalization: Symmetric and Balanced **
 ![Histogram Equalization: Symmetric and Balanced](https://cloud.githubusercontent.com/assets/2206789/22631705/f71f9f5e-ebc6-11e6-9315-c4fc5f3048ab.png)

 ** Loss Function 5 Epochs **
 ![Loss Function 5 Epochs](https://cloud.githubusercontent.com/assets/2206789/22679803/a52b2972-ecb9-11e6-83c1-c8b1eb066999.png)

 ** Predictions for 120 Frames **
 ![Predictions for 120 Frames](https://cloud.githubusercontent.com/assets/2206789/22679856/e07909f4-ecb9-11e6-9840-46e91d3b68ca.png)

 #### Common Questions Addressed via Slack:
 TODO

#### Process


My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, I modified the model so that ...

Then I ...

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
