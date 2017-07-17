# Behaviour cloning project report

# How to train
Model can be trained by simply running `python model.py`
All neccesary code to create model, read data and train will be executed.
Training data has to be placed in `./data` folder.


# Data augmentation strategies

On a plain data set model was not performing well and rather often deviating car outside the track.

First strategy that was applied was a simple flip of the image for every image from original data set.
Even so it gave a good improvement


# Code comments
For simplicity and speed it was decided not to use Generator to load data. 
Several Gb is not a problem but speed ups process


## Experiment log (started after migrated to Nvidia DL model)

## Experiment x.001
Setup: no flip, central + side images (correction=0.25)
Result: model deviates to left side and crashes on the bridge

## Experiment x.002
Setup: no flip, central + side images (correction=0.25)
Dropped 80% of images with angles below 0.05 in absolute
Model: nvidia based

Retraining for 2 epochs: 
Result: decent drive. Passed turn 1, near crash on bridge (might be due to removed wast amount of straight line images),
 complete miss on turn 2


## Experiment x.003
x.002 changes: instead of dropping images with stearing angle near zero on initial read of the file it was moved to 
generator. Which means we will be dropping different images on every epoch. 
This is to avoid potentially loosing some valuable training images when straight line is different from other cases 
(like bridge)

Retraining for 2 epochs: 
Result: gone of track turn 1


## Experiment x.004
x.002 changes: added flipped images to the training set. Flip image is added for every center image in the training set. 

Retraining for 2 epochs: 
Result: no positive impact. Crashed on bridge

## Experiment x.005
x.004 changes: introduced dropout to model. In between convolutional layers and before classifier. 
Training data set - removed near zero values to allow model to train with more shar turns data

Retraining for 5 epochs: 
Result: 

