# Behaviour cloning project report

(Resulting video)[https://youtu.be/85wMVQZB0W4]


# How to train
Model can be trained by simply running `python model.py`
All neccesary code to create model, read data and train will be executed.
Training data has to be placed in `./data` folder.


# Data augmentation strategies

On a plain data set model was not performing well and rather often deviating car outside the track.

First strategy that was applied was a simple flip of the image for every image from original data set.
Even so it gave a good improvement was not satisfactory.


Adding Left and Right images basically increased the data set and allowed for better generalization of the model.
But still was not enough to do the full track run.
Basis of data collection was udacity data set.

Important factor here was a randomization of various augmentations. 
Instead of just taking a flipped version of the image it was a subject to random factor 
(if randomly generated number is below certain threshold - add image, otherwise not)


Probably the most important improvement came when input data set was made more homogenous by removing lots of inputs 
with steering angles around zero.
* [before removing](Screenshot 2017-07-16 19.20.55)
* [after removing and adding flip + side images](Screenshot 2017-07-22 20.11.10)


Data was randomly dropped based on the steering angle. 95% of inputs with angle around 0+-0.05 
and 85% with angles below +-0.2 were dropped.
This helped to generalize the model to handle sharp turns as the amount of data with sharp angles had 
bigger presence in the input data set.

Above mentioned augmentations brought input dataset to about 10 000 images per epoch

# Model

After rounds of experiments a model based on Nvidia research was selected.
It has a good balance between training time and capacity to train.


Architecture:
All inputs are cropped to remove useless noise and normalized to zero mean and -1,1 range
 

First convolutional layers have a (5,5) kernel with (2,2) strides. Meaning after each layer input is reduces
by 2 times. Each layer has 24, 36, 48 filters respectively and is followed by a convolutional layer.

Last 2 convolutional layers have a (3,3) kernel with (1,1) strides.

Top of the endwork ends with 5 fully connected layers. 
Each of those is followed by a dropout layer.


To prevent overfiting and better generalization:
 * dropout layers were added almost after every fully connected hidden layer
 * max pooling was used for convolutional layers. First max pool has a wider kernel 
 to potentially cover the larger input size and spot larger features. All other pooling layers are 2x2 kernels. 
 All pooling layers have (1,1) stride to keep input dimentions in place. Convolutions reduce it enough by itself.

Adam optimizer is used for training.

Practice showed that training for 15 epochs is usually enough to lower the loss to drive the car in a decent manner.

## Experiment log (started after migrated to Nvidia DL model)

#### Experiment x.001
Setup: no flip, central + side images (correction=0.25)
Result: model deviates to left side and crashes on the bridge

#### Experiment x.002
Setup: no flip, central + side images (correction=0.25)
Dropped 80% of images with angles below 0.05 in absolute
Model: nvidia based

Retraining for 2 epochs: 
Result: decent drive. Passed turn 1, near crash on bridge (might be due to removed wast amount of straight line images),
 complete miss on turn 2


#### Experiment x.003
x.002 changes: instead of dropping images with stearing angle near zero on initial read of the file it was moved to 
generator. Which means we will be dropping different images on every epoch. 
This is to avoid potentially loosing some valuable training images when straight line is different from other cases 
(like bridge)

Retraining for 2 epochs: 
Result: gone of track turn 1


#### Experiment x.004
x.002 changes: added flipped images to the training set. Flip image is added for every center image in the training set. 

Retraining for 2 epochs: 
Result: no positive impact. Crashed on bridge

#### Experiment x.005
x.004 changes: introduced dropout to model. In between convolutional layers and before classifier. 
Training data set - removed near zero values to allow model to train with more shar turns data.
![label distribution](https://www.dropbox.com/s/rhrj90eyzi4ez0o/Screenshot%202017-07-16%2021.47.19.png?dl=0)

Retraining for 30 epochs. Samples used 1258
Result: Does sharp turns when can go straight.
Validation error stuck at a certain level - seems like overfitting.
More training allowed to pass critical corners but car takes to may turns when it is not needed. But not consistent.

Try to fine-tune it with full data set

#### Experiment x.006
x.005 changes: flip enabled with 60% probability, for side images too
Added more dropouts to the model.

Retraining for 30 epochs. Samples used 1258
Result: Not satisfactory. Car did not pass the first turn.

#### Experiment x.007
x.006 changes: Retrain x.006 on full data set 

Retraining for 15 epochs. Samples used 8038.
Result: bad. Ran off into water

#### Experiment x.008
x.005 changes: Flip is now 50% with original image (either one is selected based on a random number) 
```
if abs(s_angle) < 0.05 and rnd.random() <= 0.65: continue
if abs(s_angle) < 0.2 and rnd.random() <= 0.85: continue
```
is used for dropping images in generator (in run time).

Retraining for 30 epochs. Samples used (random).
Start fresh.

Result: Ran into self-oscilating scenario

#### Experiment x.009
x.008 changes: Train with full data set

Retraining for 15 epochs. Full sample set
Result: completely failed

#### Experiment x.010
x.008 changes: Train with full data set
x.008 approach to filtering image data set. 
Flip by it self is only probabilistic (70%) but main image is always present.
Batch size increased

Retraining for 5 epochs. Full sample set
Result: Deviates to left and crosses the line at the start

#### Experiment x.011
x.008 changes: instead of just evening the data, we are removing near zero images but keeping their side images.
Which means 2 spikes at 0.25 angles. Let's see what will happen. Hypothesis: jiggling car.
Flip: False
[chart](https://www.dropbox.com/s/7zml1b8f5iolue1/Screenshot%202017-07-17%2012.59.15.png?dl=0)

Retraining for 5 epochs. Full sample set
Result:

#### Experiment x.012
x.011 changes: dropping more almost zero and less close to zero angles. Getting to more uniform data distribution. 
Flip: True
[chart](https://www.dropbox.com/s/iwt5eu9m785rsl6/Screenshot%202017-07-17%2013.21.21.png?dl=0)

Retraining for 5 epochs. Full sample set (approx 6370 images after augmentation)
Result: (skipped to x.013)


#### Experiment x.013
x.012 changes: maxpool on all convolutional levels 

Retraining for 5 epochs. Full sample set ()
Result: faster training (suspicious). Wiggly driving and broke on a bridge.

#### Experiment x.014
x.012 changes: adjusted filtering of images (found error in simulated calculations) 
[chart](https://www.dropbox.com/s/25eno7lmd2qoa7s/Screenshot%202017-07-18%2001.13.33.png?dl=0)

Retraining for 10 epochs. Full sample set ()
Result: not bad but got to shikana turn

#### Experiment x.015
x.014 changes: drop only angles below 0.05. Always flip main image. 0.7 flip for side images
[chart](https://www.dropbox.com/s/jcgpnzjuw82jv41/Screenshot%202017-07-19%2000.18.15.png?dl=0)

Retraining for 15 epochs. Full sample set (18172)
Result: very good - done several laps. There is some problems in tricky left turn without right line but overall very good.
Done part of the 2nd track - stuck in the very sharp turn. Still need more data for sharp turns.
Retrained model the same does not show consistent performance.

#### Experiment x.016
x.015 changes: Added 2 brightness augmented images for sharp turns to increase data set volume (for angles over 0.4)
Found mistake in brightness augmentation code. 

Retraining for 15 epochs. Full sample set
Result: not that good. constantly turns on straights and missed "after bridge" turn

#### Experiment x.017
x.015 changes: kept more center images (0.93 drop rate)

Retraining for 15 epochs. Full sample set
Result: does not turn in sharp corners at all and even turns oposite 

#### Experiment x.018
x.015 changes: replaced maxpool with dropout

Retraining for 15 epochs. Full sample set
Result: no big difference

#### Experiment x.019
x.015 changes: added greyscale image as addition to dataset 

Retraining for 15 epochs. Full sample set
Result: failed. Biased to right.

#### Experiment x.020
x.019 changes: removed more images at the center - added conditions to drop 85% of below 0.2 angels

Retraining for 12 epochs. Full sample set
Result: failed on the bridge but good behaviour overall

#### Experiment x.021
x.020 changes: removed some maxpool layers and dropout layers.

Retraining for 15 epochs. Full sample set
Result: strange after the bridge - went off the road. "need more minerals"
Seems like because we are dropping a lot of images with zero values model either do not generalize well 
(no signs of overfiting). Let's try additional epoch with full dataset.

#### Experiment x.022
x.021 changes: greyscale for flips too. Probability 50%. Added also brightness augmentation 

Retraining for 15 epochs. Full sample set
Result: went straight of the track

#### Experiment x.023
x.022 changes: fine-tune x.022 on full dataset without removing close to zero angles. 

Fine tuning for 3 epochs. 
Result: too biased to going straight - failed

#### Experiment x.024
x.022 changes: disabled brightness augmentation 

Retraining for 15 epochs. Full sample set
Result: failed. Some problem with Grey images

#### Experiment x.025
x.024 changes: more pooling and dropout in layers. No grey or brightness augmented
Added additional verification logic to have some basic picture on how model reacts on all length of the track. 
Feeding images of one circle and comparing steering angles to recorded for training.

Retraining for 15 epochs. Full sample set
Result: almost a smooth run. One hickup - model lacks the ability to make very sharp turn. 
This is seem to be due to the lack of traing data with sharp turns. 

#### Experiment x.026
x.025 changes: added more recovery data (car going from one side of the track to another and all approaches to 
side lines are removed from the set.)
Speed 20

Result: very good. A bit wiggly on Track 1 and performs well on track 2 except for super sharp turn
