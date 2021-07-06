import numpy as np
import matplotlib.pyplot as plt
import cv2
import csv
import os

lines = []
images = []
steering = []

# Include data provided by Udacity for training
with open('/opt/carnd_p3/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(csvfile) # skip first line with headers
    for line in reader:
        lines.append(line)

# Importing data without generator
for line in lines:
    steering_center = float(line[3])

    # Create adjusted steering measurements for the side camera images
    correction = 0.4 # this is a parameter to tune
    steering_left = steering_center + correction
    steering_right = steering_center - correction

    # Add images from center, left, and right cameras to training dataset
    path = "/opt/carnd_p3/data/IMG/" # path to Udacity training IMG directory
    center_path = path + line[0].split('/')[-1]
    left_path = path + line[1].split('/')[-1]
    right_path = path + line[2].split('/')[-1]
    
    img_center = cv2.imread(center_path)
    rgb_center = cv2.cvtColor(img_center, cv2.COLOR_BGR2RGB)
    images.append(rgb_center)
    steering.append([steering_center])
    
    img_left = cv2.imread(left_path)
    rgb_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB)
    images.append(rgb_left)
    steering.append([steering_left])
    
    img_right = cv2.imread(right_path)
    rgb_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2RGB)
    images.append(rgb_right)
    steering.append([steering_right])
    
    # Flip images and steering angles if more data is needed
    
    # Flip center images and steering angles
    images.append(np.fliplr(img_center))
    steering.append([steering_center*-1.0])
    """
    # Flip left images and steering angles
    images.append(np.fliplr(img_left))
    steering.append([steering_left*-1.0])
    # Flip right images and steering angles
    images.append(np.fliplr(img_right))
    steering.append([steering_right*-1.0])
    """
# Training dataset
X_train = np.array(images)
print(X_train.shape)
y_train = np.array(steering)
print(y_train.shape)

# Shuffle training data
#from sklearn.utils import shuffle
#X_train, y_train = shuffle(X_train, y_train) # model.fit() is already shuffling data

# Build training model
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers import Convolution2D, MaxPooling2D, Cropping2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()

# Added dropout to NVIDIA model from lecture videos
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Conv2D(24,(5,5),strides=(2,2),activation="relu"))
model.add(Conv2D(36,(5,5),strides=(2,2),activation="relu"))
model.add(Conv2D(48,(5,5),strides=(2,2),activation="relu"))
model.add(Conv2D(64,(3,3), activation="relu"))
model.add(Conv2D(64,(3,3), activation="relu"))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.3))
model.add(Dense(50))
model.add(Dropout(0.3))
model.add(Dense(10))
model.add(Dropout(0.3))
model.add(Dense(1))
#model.add(Dropout(0.2))

model.compile(loss='mse', optimizer='adam') # try different learning rates
history_object = model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, epochs=15)

# Print the keys contained in the history object
print(history_object.history.keys())

model.save('model.h5')
model.summary()

# Plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig('loss.png')