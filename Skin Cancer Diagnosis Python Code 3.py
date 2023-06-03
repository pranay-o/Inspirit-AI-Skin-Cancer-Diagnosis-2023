#Run this to download data and prepare our environment!

print("Downloading data...")

!wget -q --show-progress 'https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%206%20-%2010%20(Projects)/Project%20-%20(Healthcare%20B)%20Skin%20Cancer%20Diagnosis/X.npy'
!wget -q --show-progress 'https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%206%20-%2010%20(Projects)/Project%20-%20(Healthcare%20B)%20Skin%20Cancer%20Diagnosis/X_g.npy'
!wget -q --show-progress 'https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%206%20-%2010%20(Projects)/Project%20-%20(Healthcare%20B)%20Skin%20Cancer%20Diagnosis/y.npy'


print("Importing stuff...")

import os
import random
import numpy as np
import pandas as pd
import cv2

import matplotlib.pyplot as plt
import seaborn as sns

import keras.api.keras as keras
!pip install git+https://github.com/rdk2132/scikeras # workaround for scikeras deprecation
import scikeras
from scikeras.wrappers import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.applications.mobilenet import MobileNet
from keras.metrics import AUC

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

print("Done")

#Let's load in our data from last time!
IMG_WIDTH = 100
IMG_HEIGHT = 75
X = np.load("X.npy")
X_g = np.load("X_g.npy")
y = np.load("y.npy")

#Now, let's re-train the transfer learning model we built in File 2. We'll examine its effectiveness for different skin tones.
#Run this to Perform Data Augmentation!
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)
X_g_train, X_g_test, y_train, y_test = train_test_split(X_g, y, test_size=0.4, random_state=101)

X_augmented = []
X_g_augmented = []

y_augmented = []

for i in range(len(X_train)):
  transform = random.randint(0, 1)
  if (transform == 0):
    # Flip the image across the y-axis
    X_augmented.append(cv2.flip(X_train[i], 1))
    X_g_augmented.append(cv2.flip(X_g_train[i], 1))
    y_augmented.append(y_train[i])
  else:
    # Zoom 33% into the image
    zoom = 0.33

    centerX, centerY = int(IMG_HEIGHT/2), int(IMG_WIDTH/2)
    radiusX, radiusY = int((1-zoom)*IMG_HEIGHT*2), int((1-zoom)*IMG_WIDTH*2)

    minX, maxX = centerX-radiusX, centerX+radiusX
    minY, maxY = centerY-radiusY, centerY+radiusY

    cropped = (X_train[i])[minX:maxX, minY:maxY]
    new_img = cv2.resize(cropped, (IMG_WIDTH, IMG_HEIGHT))
    X_augmented.append(new_img)

    cropped = (X_g_train[i])[minX:maxX, minY:maxY]
    new_img = cv2.resize(cropped, (IMG_WIDTH, IMG_HEIGHT))
    X_g_augmented.append(new_img)

    y_augmented.append(y_train[i])

X_augmented = np.array(X_augmented)
X_g_augmented = np.array(X_g_augmented)

y_augmented = np.array(y_augmented)

X_train = np.vstack((X_train, X_augmented))
X_g_train = np.vstack((X_g_train, X_g_augmented))

y_train = np.append(y_train, y_augmented)

def transfer_learning_model():
  mobilenet_model = MobileNet(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), include_top=False, pooling="max")

  transfer_model = Sequential()
  transfer_model.add(mobilenet_model)
  transfer_model.add(Dropout(0.1))
  transfer_model.add(BatchNormalization())
  transfer_model.add(Dense(256, activation="relu"))
  transfer_model.add(Dropout(0.1))
  transfer_model.add(BatchNormalization())
  transfer_model.add(Dense(7, activation="softmax"))

  return transfer_model

#Transform our labels into One Hot encodings and process our variables!
y_train_roc = np.zeros((y_train.size, y_train.max().astype(int)+1))
y_train_roc[np.arange(y_train.size), y_train.astype(int)] = 1

y_test_roc = np.zeros((y_test.size, y_test.max().astype(int)+1))
y_test_roc[np.arange(y_test.size), y_test.astype(int)] = 1

# typecast labels and input
X_train = X_train.astype(np.float32)
y_train_roc = y_train_roc.astype(np.float32)
y_test_roc = y_test_roc.astype(np.float32)

#Set up and train the transfer model.
opt = keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)
transfer_model = KerasClassifier(model=transfer_learning_model, optimizer=opt, loss='categorical_crossentropy', epochs=20, batch_size=10, validation_batch_size=10, validation_split=.2, metrics=[keras.metrics.AUC()])
#train the model 
transfer_model.fit(X_train, y_train_roc, validation_data=(X_test, y_test_roc))
#predict 

#Run this to redefine `plot_cm()`!
def plot_cm(name, y_test_class, y_pred_class):
  cm = confusion_matrix(y_test_class, y_pred_class)
  classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

  classes_present = []
  for i in range(len(classes)):
    if i in y_pred_class or i in y_test_class:
      classes_present.append(classes[i])

  df_cm = pd.DataFrame(cm, index = [i for i in classes_present], columns = [i for i in classes_present])
  df_cm = df_cm.round(5)

  plt.figure(figsize = (12, 8))
  sns.heatmap(df_cm, annot=True, fmt='g')
  plt.title(name + " Model Confusion Matrix")
  plt.xlabel("Predicted Label")
  plt.ylabel("True Label")
  plt.show()

#The cell below converts 7-dimensional one hot encoding back to 1-dimensional numerical encoding.
def one_hot_to_numerical(y_pred):
  y_pred_1d = [0] * len(y_pred)
  for i in range(len(y_pred)):
    y_pred_1d[i] = np.where(y_pred[i] == 1)[0][0]
  return y_pred_1d

#For simplicity, we'll use accuracy as a metric. Later on, we'll compare the accuracy of our model for each of our skin tone classes. For now, output the confusion matrix and the overall accuracy of our model for the entire test set:
#accuracy 

y_pred = transfer_model.predict(X_test)
print(accuracy_score(y_test, one_hot_to_numerical(y_pred)))

plot_cm("Overall", y_test, one_hot_to_numerical(y_pred))

#Classifying Skin Tones
#Now, we'll see how well our model performs for images of different skin tones.
#We will classify each image in our test set based on a skin tone palette, shown below. We will use the average RGB pixel values to determine which skin tone each photo is closest to.

#Run the following code to find the average red, green, and blue values of each image in the test set.
# images of size 75x100 with 3 output rgb channels 
avg_rgb = X_test.mean(axis=2).mean(axis=1)
print(avg_rgb.shape)

#Now we have found the average red, green, and blue values for each image. We will use a clustering approach to find which of the six skin tones each image vector is closest to. We will use Euclidean Distance as our metric of similarity.
#We define the function closest_node(node, nodes) which takes in the average RGB values of a photo, and finds the closest skin color out of the nodes.
nodes = [[197, 140, 133], [236, 188, 180], [209, 163, 164], [161, 102, 94], [80, 51, 53], [89, 47, 42]]

# finds the closest color from the nodes array by index. 
def closest_node(node, nodes):
    nodes = np.asarray(nodes)
    dist_2 = np.sum((nodes - node)**2, axis=1)
    return np.argmin(dist_2)

#You can try it out with a particular image:
img_num = 2 #Choose any image here
img = X_test[img_num]
plt.imshow(img) 
plt.show()
print ("Skin tone class:", closest_node(avg_rgb[img_num], nodes))

#Iterate through avg_rgb, use closest_node to find the skin class for each image, and append the results to a skin_classes list.
skin_classes = []
for img in avg_rgb:
  skin_classes.append(closest_node(img, nodes))
print(skin_classes)

#We will visualize the results of our findings using a bar chart.
#Run to plot skin class frequency in the test set!

frequency = []
for i in range(6): 
  frequency.append(skin_classes.count(i))

plt.bar([0, 1, 2, 3, 4, 5], frequency, label="skin class frequencies")

# The following commands add labels to our figure.
plt.xlabel('Skin Classes')
plt.ylabel('Frequency')
plt.title('Frequency of Skin Classes')

#Measuring Fairness by Skin Tone
#Now that we have the overall accuracy of our model, let's break it down further by skin tone classes within the test set. The code below will select the portions of y_test and y_pred for each class; complete it to show the accuracy and confusion matrix for each class.
for skin_class in range(6):
  print ("Skin Class: {}".format(skin_class))
  mask = (np.array(skin_classes) == skin_class)
  y_test_class = y_test[mask]
  y_pred_class = y_pred[mask]
  print ("Number of images: {}".format(len(y_test_class)))
  if len(y_test_class) > 0:
    plot_cm("Skin Class: {}".format(skin_class), y_test_class, \
            one_hot_to_numerical(y_pred_class))