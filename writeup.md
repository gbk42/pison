# Pison Interview Project: Unsupervised Classification of Wrist Gestures

# Data Characterization

## Nerve Channel Data

## Dynamics Data

## Class Characterization

## Movement Impact on Sensor Data

# Feature Extraction and Clustering
## Time scale of information
Human reaction times range anywhere from 150 - 350 milliseconds and a runner with a somewhat higher cadence gait may approach or surpass up to 200 steps per minute, which corresponds to roughly 300 milliseconds between strides.  Therefore, the time windows between new neurological and physiological events should roughly inhabit this quarter of a second interval at the most extreme.  This will likely be a parameter chosen wisely during feature selection.

## Clustering results


# Classification

## Algorithms

## Results

# Conclusion

### Movement impact on sensor information
As movements increase, dynamics sensors become less reliable and much more distorted, but patterns in high passed ADC data remain somewhat consistent.  For instance in running rep 3, there is a nice pulse train formed in the high passed data that roughly corresponds to the start of IMU pulses.



### Description of Movements

There are three distinct movement patterns in this data that can be characterized by gyroscope and accelerometer data and visualized by plotting the quaterions in euler format.  Note that gravitational acceleration shows up as a constant -9.8 m/s/s when the device is oriented flat on a table.

1. Rotation about the Z axis with slight rotations in Y - this would look like turning a doorknob about a quarter turn.  A quarter turn aligns well with the accelerometer readings of `ax` and `az` meeting at the peak of the rotation, meaning gravitational acceleration is perfectly split between the two at 45 degrees of rotation.  There is wavering in the Y direction, which may be an artifact of turning the wrist drawing the forearm away from the middle of the body on the right and towards the body on the left.

2. Rotation about X.  This would look someone moving their hand out to hit a key on a keyboard at waist height.  We again see two axes, Y and Z, of acceleration meet at the midpoint of the gesture, meaning a 45 degree angle was reached.  

3. Pure and rapid rotation about the Y axis with accleration in the X and Y axes that are in phase.  For lack of a better description, this looks like someone brushing eraser dust off of a sheet of paper.

### Features of Interest

Basic STats
Correlation
Spectral Entropy
Min/Max
Segmented Mean/Var