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

Result: 


