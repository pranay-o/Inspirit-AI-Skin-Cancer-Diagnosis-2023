#Run this to download data and prepare our environment.
print("Installing packages...")
!pip -q install hypopt tensorflowjs > /dev/null
!pip -q install git+https://github.com/rdk2132/scikeras # workaround for scikeras deprecation
import os
import cv2
import random
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import keras.api.keras as keras
import scikeras
import tensorflowjs as tfjs

from tqdm.notebook import tqdm
from keras.layers import * # import all, including Dense, add, Flatten, etc.
from keras.models import Model, Sequential
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score
from keras.applications.mobilenet import MobileNet
from hypopt import GridSearch

print("Downloading files...")
!wget -q --show-progress 'https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%206%20-%2010%20(Projects)/Project%20-%20(Healthcare%20B)%20Skin%20Cancer%20Diagnosis/X.npy'
!wget -q --show-progress 'https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%206%20-%2010%20(Projects)/Project%20-%20(Healthcare%20B)%20Skin%20Cancer%20Diagnosis/X_g.npy'
!wget -q --show-progress 'https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%206%20-%2010%20(Projects)/Project%20-%20(Healthcare%20B)%20Skin%20Cancer%20Diagnosis/y.npy'

# Set up Web App
os.makedirs("static/js", exist_ok=True)
!wget -q --show-progress 'https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%206%20-%2010%20(Projects)/Project%20-%20(Healthcare%20B)%20Skin%20Cancer%20Diagnosis/skin_cancer_diagnosis_script.js' &> /dev/null
output = 'static/js/skin_cancer_diagnosis_script.js'

print("Done!")

#Let's load in our data from last time!
X = np.load("X.npy")
X_g = np.load("X_g.npy")
y = np.load("y.npy")

#Run this to Perform Data Augmentation
IMG_WIDTH = 100
IMG_HEIGHT = 75

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)
X_g_train, X_g_test, y_train, y_test = train_test_split(X_g, y, test_size=0.4, random_state=101)

X_augmented = []
X_g_augmented = []

y_augmented = []

for i in tqdm(range(len(X_train))):
  transform = random.randint(0, 1)
  if (transform == 0):
    # Flip the image across the y-axis
    X_augmented.append(cv2.flip(X_train[i], 1))
    X_g_augmented.append(cv2.flip(X_g_train[i], 1))
    y_augmented.append(y_train[i])
  else:
    zoom = 0.33 # Zoom 33% into the image

    centerX, centerY = int(IMG_HEIGHT/2), int(IMG_WIDTH/2)
    radiusX, radiusY = int((1-zoom)*IMG_HEIGHT*2), int((1-zoom)*IMG_WIDTH*2)

    minX, maxX = centerX-radiusX, centerX+radiusX
    minY, maxY = centerY-radiusY, centerY+radiusY

    cropped = (X_train[i])[minX:maxX,  minY:maxY]
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

#Creating Machine Learning Models
#We'll use the the library Keras to create the models for our skin lesion classification. Let's start off by creating our own CNN model 

def CNNClassifier(epochs=20, batch_size=10, layers=5, dropout=0.5, activation='relu'):
  def set_params():
    i = 1  
  def create_model():
    model = Sequential()
    
    for i in range(layers):
      model.add(Conv2D(64, (3, 3), padding='same'))
      model.add(Activation(activation))
    
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation(activation))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout / 2.0))

    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation(activation))
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation(activation))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout / 2.0))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation(activation))
    model.add(Dropout(dropout))
    model.add(Dense(7))
    model.add(Activation('softmax'))

    # initiate RMSprop optimizer
    # Let's train the model using RMSprop
    return model
  opt = keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)
  return KerasClassifier(model=create_model, optimizer=opt, loss='categorical_crossentropy', epochs=epochs, batch_size=batch_size, verbose=1, validation_batch_size=batch_size, validation_split=.4, metrics=['accuracy'])

# Run this to process our X variables and transform our y labels into one hot encoded labels for training.
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)

y_train_onehot = np.zeros((y_train.size, y_train.max().astype(int)+1))
y_train_onehot[np.arange(y_train.size), y_train.astype(int)] = 1
y_train_onehot = y_train_onehot.astype(np.float32)

y_test_onehot = np.zeros((y_test.size, y_test.max().astype(int)+1))
y_test_onehot[np.arange(y_test.size), y_test.astype(int)] = 1
y_test_onehot = y_test_onehot.astype(np.float32)

#Let's initialize and train our CNN.
cnn = CNNClassifier()

cnn.fit(X_train, y_train_onehot,
        validation_data=(X_test, y_test_onehot))

#Let's save and download our trained model, so that we can use it in a web app later on.
tfjs.converters.save_keras_model(cnn.model_, 'cnn_model')

#Let's evaluate our model's performance! Let's start by defining our model_stats() and plot_cm() functions.
def model_stats(name, y_test, y_pred, y_pred_proba):
  y_pred_1d = [0] * len(y_test)
  for i in range(len(y_test)):
    y_pred_1d[i] = np.where(y_pred[i] == 1)[0][0]

  cm = confusion_matrix(y_test, y_pred_1d)

  print(name)

  accuracy = accuracy_score(y_test, y_pred_1d)
  print ("The accuracy of the model is " + str(round(accuracy, 5)))

  y_test_onehot = np.zeros((y_test.size, y_test.max().astype(int)+1))
  y_test_onehot[np.arange(y_test.size), y_test.astype(int)] = 1

  roc_score = roc_auc_score(y_test_onehot, y_pred_proba)

  print ("The ROC AUC Score of the model is " + str(round(roc_score, 5)))
  
  return cm

y_pred = cnn.predict(X_test)
y_pred_proba = cnn.predict_proba(X_test)

cnn_cm = model_stats("CNN", y_test, y_pred, y_pred_proba)

#Let's also redefine the `plot_cm()` function from our first file. 
classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

df_cm = pd.DataFrame(cm, index = [i for i in classes], columns = [i for i in classes])
df_cm = df_cm.round(5)

plt.figure(figsize = (12, 8))
sns.heatmap(df_cm,  annot=True, fmt='g')
plt.title(name + " Model Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

#Let's use the function to plot a confusion matrix of our model!
plot_cm("CNN", cnn_cm)

#It looks like our custom CNN's performance is better than the KNN and Decision Tree models. More training epochs or a bigger dataset would probably help with the performance.

#Use cnn_cm as the variable name for your model's performance.
y_pred = cnn.predict(X_test)
y_pred_proba = cnn.predict_proba(X_test)
cnn_cm = model_stats("CNN", y_test, y_pred, y_pred_proba)

plot_cm("CNN", cnn_cm)

#Let's try using a grid search with our CNN
#In the variable param_grid we can specify which parameters in our CNN we want to modify.
param_grid = {
              'epochs' :              [10, 20, 30],
              'batch_size' :          [32, 64, 128],
              'layers' :              [1, 3, 5],
              'dropout' :             [0.2, 0.3, 0.5],
              'activation' :          ['relu', 'elu']
             }

#With this parameter grid we would be training 162 different models! This is because the total number of hyperparameter combinations is calculated as 3 * 3 * 3 * 3 * 2. For testing out our grid search, let's redefine our parameter grid to just have four possible combinations.
param_grid = {
              'epochs' :              [10, 20],
              'dropout' :             [0.2, 0.3],
             }

#Let's create a validation slice in our dataset.
X_test_small, X_val, y_test_small, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=101)

#Here we've created a class for our Grid Search Classifier that can be used by the hypopt library for generating various models with different hyperparameters.
#Run this to define our Grid Search CNN Class
class gridSearchCNN():
    
    keras_model = None
    model = Sequential()
    epochs=1
    batch_size=10
    layers=5
    dropout=0.5
    activation='relu'
    
    def __init__(self, **params):
      pass
  
    def fit(self, X, y, sample_weight = None):
        print("fitting")
        self.keras_model.fit(X, y)
        print("fitted")
        return self.keras_model
    def predict(self, X):
        return self.keras_model.predict(X)
    def predict_proba(self, X):
        return self.keras_model.predict_proba(X)
    def score(self, X, y, sample_weight = None):
        print("scoring")
        y_pred_proba = self.keras_model.predict_proba(X)
        roc_auc_score_val = roc_auc_score(y, y_pred_proba)
        print("scored")
        return roc_auc_score_val
                
    
    def createKerasCNN(self,):
      
      def create_model():
        self.model = Sequential() 
        
        for i in range(self.layers):
          self.model.add(Conv2D(64, (3, 3), padding='same'))
          self.model.add(Activation(self.activation))
        
        self.model.add(Conv2D(64, (3, 3)))
        self.model.add(Activation(self.activation))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(self.dropout / 2.0))

        self.model.add(Conv2D(128, (3, 3), padding='same'))
        self.model.add(Activation(self.activation))
        self.model.add(Conv2D(128, (3, 3)))
        self.model.add(Activation(self.activation))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(self.dropout / 2.0))

        self.model.add(Flatten())
        self.model.add(Dense(512))
        self.model.add(Activation(self.activation))
        self.model.add(Dropout(self.dropout))
        self.model.add(Dense(7))
        self.model.add(Activation('softmax'))

        return self.model

      opt = keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)
      return KerasClassifier(model=create_model, optimizer=opt, loss='categorical_crossentropy', epochs=self.epochs, batch_size=self.batch_size, validation_batch_size=self.batch_size, validation_split=.4, metrics=[keras.metrics.AUC()])

    def get_params(self, deep = True):
        return {
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'layers': self.layers,
            'dropout': self.dropout,
            'activation': self.activation
            }

    def set_params(self, **params):
      if 'epochs' in params.keys():
        self.epochs = params['epochs']
      if 'batch_size' in params.keys():
        self.batch_size = params['batch_size']
      if 'layers' in params.keys():
        self.layers = params['layers']
      if 'dropout' in params.keys():
        self.dropout = params['dropout']
      if 'activation' in params.keys():
        self.activation = params['activation']
      
      self.keras_model = self.createKerasCNN()
      return self
    
#Run this to process our variables and implement One-Hot Encoding on our y values.
# Processing X_train
X_train = X_train.astype(np.float32)

# Encoding y_train
y_train_onehot = np.zeros((y_train.size, y_train.max().astype(int)+1))
y_train_onehot[np.arange(y_train.size), y_train.astype(int)] = 1
y_train_onehot = y_train_onehot.astype(np.float32)

# Processing X_val
X_val = X_val.astype(np.float32)

# Encoding y_val
y_val_onehot = np.zeros((y_val.size, y_val.max().astype(int)+1))
y_val_onehot[np.arange(y_val.size), y_val.astype(int)] = 1
y_val_onehot = y_val_onehot.astype(np.float32)

#Now let's implement our grid search to identify our optimal model parameters. We want to fit our grid search model. 
gs = GridSearch(model=gridSearchCNN(), param_grid=param_grid, parallelize=False)
gs.fit(X_train, y_train_onehot, X_val, y_val_onehot, verbose=1)

#Let's evaluate our model with our testing dataset. Recall the variables from our dataset's validation slice. We're going to want to also initialize an appropriate y_pred variable.
y_pred = gs.predict(X_test_small)
y_pred_proba = gs.predict_proba(X_test_small)
gs_cm = model_stats("Grid Search CNN", y_test_small, y_pred, y_pred_proba)

#Let's also plot the confusion matrix.
plot_cm("Grid Search CNN", gs_cm)

#Transfer Learning
#Below, we'll use the MobileNet model as a basis for our model.
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

#We no longer need the validation dataset as we aren't tweaking any hyperparameters anymore, so we can use our original, bigger X_test dataset that includes the validation data.
#Let's confirm that our training and testing variables are the right shapes. Once again, we'll use the one-hot encoded variables for the output: y_test_roc and y_train_roc.
print(X_train.shape)
print(y_train_onehot.shape)

print(X_test.shape)
print(y_test_onehot.shape)

#Let's define our transfer_model and train it below!
opt = keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)
transfer_model = KerasClassifier(model=transfer_learning_model, optimizer=opt, loss='categorical_crossentropy', epochs=20, batch_size=10, validation_batch_size=10, validation_split=.2, metrics=[keras.metrics.AUC()])
transfer_model.fit(X_train, y_train_onehot,
        validation_data=(X_test, y_test_onehot))

#Let's also observe its performance:
y_pred = transfer_model.predict(X_test)
y_pred_proba = transfer_model.predict_proba(X_test)
transfer_cm = model_stats("Transfer CNN", y_test, y_pred, y_pred_proba)

#Let's take a look at the confusion matrix.
plot_cm("Transfer Learning CNN", transfer_cm)

#Now that we've created our model, let's save it to a file we can load up later.
tfjs.converters.save_keras_model(transfer_model.model_, 'transfer_model')

#Now, the model should be saved to your computer through your browser. Unfortunately, tensorflowjs doesn't support this version of MobileNet, so we'll have to use our first CNN model for the website deployment.
# Now, our next step is to package this model into a mobile application. Run the code cell below.
!zip -r ./cnn_model.zip ./cnn_model/
