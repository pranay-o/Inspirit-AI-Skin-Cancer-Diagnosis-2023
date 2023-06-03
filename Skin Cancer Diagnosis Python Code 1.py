#Run this to download data and prepare our environment! 
!pip install hypopt
!pip install -U opencv-contrib-python==4.1.2.30

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os
import random
from PIL import Image

from tqdm.notebook import tqdm
from google.colab.patches import cv2_imshow
from hypopt import GridSearch

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, roc_auc_score

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score

# Prepare data
images_1 = os.makedirs('images_1', exist_ok=True)
images_2= os.makedirs('images_2', exist_ok=True)
images_all= os.makedirs('images_all', exist_ok=True)

!wget -q --show-progress 'https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%206%20-%2010%20(Projects)/Project%20-%20(Healthcare%20B)%20Skin%20Cancer%20Diagnosis/metadata.csv'
!wget -q --show-progress 'https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%206%20-%2010%20(Projects)/Project%20-%20(Healthcare%20B)%20Skin%20Cancer%20Diagnosis/images_1.zip'
!wget -q --show-progress 'https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%206%20-%2010%20(Projects)/Project%20-%20(Healthcare%20B)%20Skin%20Cancer%20Diagnosis/images_2.zip'
!wget -q --show-progress 'https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%206%20-%2010%20(Projects)/Project%20-%20(Healthcare%20B)%20Skin%20Cancer%20Diagnosis/hmnist_8_8_RGB.csv'
!unzip -q -o images_1.zip -d images_1 
!unzip -q -o images_2.zip -d images_2 

import os.path
from os import path
from distutils.dir_util import copy_tree

copy_tree('images_1', 'images_all', verbose=1)
copy_tree('images_2', 'images_all', verbose=1)
print("Downloaded Data")

IMG_WIDTH = 100
IMG_HEIGHT = 75

#We'll start off by separating our dataset into the X and y variables. Remember, X represents our input data (images), and y represents our data's labels (skin lesion classification). Each image is scaled down to be 100 px by 75 px to reduce the memory footprint. We'll also create a variable X_gray, that is the grayscale equivalent of our X variable.
X = []
X_gray = []

y = []

#Run this to initialize our X, X_gray, and y variables
metadata = pd.read_csv("metadata.csv")
metadata['category'] = metadata['dx'].replace({'akiec': 0, 'bcc': 1, 'bkl': 2, 'df': 3, 'mel': 4, 'nv': 5, 'vasc': 6,})


for i in tqdm(range(len(metadata))):
  image_meta = metadata.iloc[i]
  path = os.path.join('images_all', image_meta['image_id'] + '.jpg')
  img = cv2.imread(path,cv2.IMREAD_COLOR)
  img = cv2.resize(img,(IMG_WIDTH,IMG_HEIGHT))
  
  img_g = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  X_gray.append(img_g)

  X.append(img)
  y.append(image_meta['category'])

X_gray = np.array(X_gray)
X = np.array(X)
y = np.array(y)

#Let's take a look at an example of what our data looks like!
cv2_imshow(X[0])

#Let's take a look at the shape of our updated X, X_gray, and y variables
print(X_gray.shape)
print(X.shape)
print(y.shape)
#It looks like we've got a total of 10,015 images in our dataset. Plotting a graph of the distribution of labels found in the dataset can help us determine if we need to balance the data.

#Run this to plot the distribution of our dataset 
objects = ('akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc')
y_pos = np.arange(len(objects))
occurances = []

for obj in objects:
  occurances.append(np.count_nonzero(obj == metadata['dx']))

print(occurances)

plt.bar(y_pos, occurances, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Samples')
plt.title('Distribution of Classes Within Dataset')

plt.show()
#This bar chart clearly informs us that our dataset is very unbalanced. There are far more nevi samples than there are samples of any other class.

#For the sake of reducing execution and training time, we'll be cutting down the size of our dataset. But if you would like to observe the full performance of our model and see the complete extent of our visualizations, you can comment out the following lines and re-run the notebook. However, note that some code blocks will take much longer to run.
#Decide between which of the following methods you would like to use to reduce your dataset size. Only run one of the two code blocks. The first option reduces the dataset size far more than the second option. Specify which option you would like to proceed with by setting the value for the variable option.
sample_cap = 142
option = 1

#Option 1: Run this to reduce dataset size. This method caps each class at sample_cap samples.
if (option == 1):
  objects = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
  class_totals = [0,0,0,0,0,0,0]
  iter_samples = [0,0,0,0,0,0,0]
  indices = []

  for i in range(len(X)):
    class_totals[y[i]] += 1

  print("Initial Class Samples")
  print(class_totals)

  for i in range(len(X)):
    if iter_samples[y[i]] != sample_cap:
      indices.append(i)
      iter_samples[y[i]] += 1

  X = X[indices]
  X_gray = X_gray[indices]

  y = y[indices]

  class_totals = [0,0,0,0,0,0,0]

  for i in range(len(X)):
    class_totals[y[i]] += 1

  print("Modified Class Samples")
  print(class_totals)
else:
  print("This option was not selected")

#Option 2: Run this to reduce dataset size. This method only reduces the number of nv samples to be the same amount as the number of samples found in the second most prevalent class.
if (option == 2):
  objects = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
  class_totals = [0,0,0,0,0,0,0]

  for i in range(len(X)):
    class_totals[y[i]] += 1

  print("Initial Class Samples")
  print(class_totals)

  largest_index = class_totals.index(max(class_totals))
  class_totals[largest_index] = 0

  second_largest_val = max(class_totals)

  indices = []
  iter = 0
  for i in range(len(X)):
    if y[i] == largest_index:
      if iter != second_largest_val:
        indices.append(i)
        iter += 1
      else:
        continue
    else:
      indices.append(i)

  X = X[indices]
  X_gray = X_gray[indices]

  y = y[indices]

  class_totals = [0,0,0,0,0,0,0]

  for i in range(len(X)):
    class_totals[y[i]] += 1

  print("Modified Class Samples")
  print(class_totals)
else:
  print("This option was not selected")
#By running the second code block above, our dataset is no longer imbalanced. This would mean that we could use accuracy as a metric for performance. 

#OpenCV Image Manipulation
#Consider professional images taken with proper medical equipment by a dermatologist. These images are more likely to be clearer and in focus, when compared with those taken by an amateur with a cell phone camera. However, as both types of images are likely to be sent to our ML model for classification, its important that we prepare our model for both situations.
#One method of increasing our dataset's size is called data augmentation. Through data augmentation, we take existing images from our dataset, and duplicate a version of that image with an image transformation applied to it. This process can be repeated multiple times, and the dataset size can increase ten-fold or greater. 

#Data Augmentation
#Although our dataset is very expansive with over 10,000 images, we can generate more samples so that our model is prepared to cope with a more varied dataset. Through data augmentation, we can perform random operations such as a flip, blur, or zoom on existing images, to create new image samples. It's important to note that these data augmentation procedures should only be applied to the training dataset.

#Let's first complete our test/train split for both our grayscale image data and our color image data.
X_gray_train, X_gray_test, y_train, y_test = train_test_split(X_gray, y, test_size=0.4, random_state=101)

#Let's also perform a test/train split for X and y: the color image data and the respective labels. We need to create X_train, X_test, y_train, y_test.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)

#We'll now iterate through all the images in the training slice of our dataset and create a duplicate with a random transformation, doubling our training dataset's size. In this code block, we'll randomly decide to flip the image across the y-axis or apply a 33% zoom.
X_augmented = []
X_gray_augmented = []

y_augmented = []

for i in tqdm(range(len(X_train))):
  transform = random.randint(0,1)
  if (transform == 0):
    # Flip the image across the y-axis
    X_augmented.append(cv2.flip(X_train[i],1))
    X_gray_augmented.append(cv2.flip(X_gray_train[i],1))
    y_augmented.append(y_train[i])
  else:
    # Zoom 33% into the image
    zoom = 0.33

    centerX,centerY=int(IMG_HEIGHT/2),int(IMG_WIDTH/2)
    radiusX,radiusY= int((1-zoom)*IMG_HEIGHT*2),int((1-zoom)*IMG_WIDTH*2)

    minX,maxX=centerX-radiusX,centerX+radiusX
    minY,maxY=centerY-radiusY,centerY+radiusY

    cropped = (X_train[i])[minX:maxX, minY:maxY]
    new_img = cv2.resize(cropped, (IMG_WIDTH, IMG_HEIGHT))
    X_augmented.append(new_img)

    cropped = (X_gray_train[i])[minX:maxX, minY:maxY]
    new_img = cv2.resize(cropped, (IMG_WIDTH, IMG_HEIGHT))
    X_gray_augmented.append(new_img)

    y_augmented.append(y_train[i])

X_augmented = np.array(X_augmented)
X_gray_augmented = np.array(X_gray_augmented)

y_augmented = np.array(y_augmented)

X_train = np.vstack((X_train,X_augmented))
X_gray_train = np.vstack((X_gray_train,X_gray_augmented))

y_train = np.append(y_train,y_augmented)

#Run this to Combine Augmented Data with Existing Samples
X_augmented = np.array(X_augmented)
X_gray_augmented = np.array(X_gray_augmented)

y_augmented = np.array(y_augmented)

X_train = np.vstack((X_train,X_augmented))
X_gray_train = np.vstack((X_gray_train,X_gray_augmented))

y_train = np.append(y_train,y_augmented)

#Let's view the shape of our training variables after data augmentation.
print(X_gray_train.shape)
print(X_train.shape)
print(y_train.shape)

#Try performing two additional image transformations with OpenCV for data augmentation!
X_augmented = []
X_gray_augmented = []

y_augmented = []

for i in tqdm(range(len(X_train))):
  transform = random.randint(0,1)
  if (transform == 0):

    # Resize the image by half on each dimension, and resize back to original
    # dimensions

    small_image = cv2.resize(X_train[i],(IMG_WIDTH//2,IMG_HEIGHT//2))
    normal_image = cv2.resize(small_image,(IMG_WIDTH,IMG_HEIGHT))

    small_grayscale_image = cv2.resize(X_gray_train[i],(IMG_WIDTH//2,IMG_HEIGHT//2))
    normal_grayscale_image = cv2.resize(small_grayscale_image,(IMG_WIDTH,IMG_HEIGHT))

    X_augmented.append(normal_image)
    X_gray_augmented.append(normal_grayscale_image)
    y_augmented.append(y_train[i])
  else:

    # Blur the image with a 4 x 4 kernel

    X_augmented.append(cv2.blur(X_train[i],(4,4)))
    X_gray_augmented.append(cv2.blur(X_gray_train[i],(4,4)))
    y_augmented.append(y_train[i])

#Run this to Combine Augmented Data with Existing Samples 
X_gray_augmented = np.array(X_gray_augmented)

y_augmented = np.array(y_augmented)

X_train = np.vstack((X_train,X_augmented))
X_gray_train = np.vstack((X_gray_train,X_gray_augmented))

y_train = np.append(y_train,y_augmented)

#Now that we've implemented data augmentation into our pipeline and artificially generated more samples for our dataset, lets test out various ML models.
#Let's start off by creating a K Nearest Neighbors model.
knn = KNeighborsClassifier(n_neighbors=5)

#Let's perform an operation known as image flattening with our grayscale image data. In this operation, we reshape our images to be a one dimensional array of length 7500 instead of a matrix of dimensions (100 x 75).
X_g_train_flat = X_gray_train.reshape(X_gray_train.shape[0],-1)
X_g_test_flat = X_gray_test.reshape(X_gray_test.shape[0],-1)
print (X_g_train_flat.shape)
print (X_g_test_flat.shape)

#Let's train our models on our flattened grayscale images!
knn.fit(X_g_train_flat, y_train)

#A common way to measure our model's performance uses the Receiver Operator Curve, which shows the relationship between our model's true positive and true negative rate. This metric is especially useful with our scenario, since - unlike accuracy - it doesn't depend on balanced classes in our dataset.
#We'll define a function called model_stats() that prints the models performance. Specifically, it will print the model's name, its accuracy, and its ROC AUC value.
def model_stats(name, y_test, y_pred, y_pred_proba):
  cm = confusion_matrix(y_test, y_pred)

  print(name)

  accuracy = accuracy_score(y_test,y_pred)
  print ("The accuracy of the model is " + str(round(accuracy,5)))

  roc_score = roc_auc_score(y_test, y_pred_proba, multi_class='ovo')

  print ("The ROC AUC Score of the model is " + str(round(roc_score,5)))
  
  return cm

#Let's run the function and observe the performance of our K Nearest Neighbors model. Remember that we have seven classes, so an accuracy that seems horribly low (like 50%) isn't as bad as it might appear!
y_pred = knn.predict(X_g_test_flat)
y_pred_proba = knn.predict_proba(X_g_test_flat)

knn_cm = model_stats("K Nearest Neighbors",y_test,y_pred,y_pred_proba)

#There seems to a big discrepancy between our accuracy and ROC AUC scores. Why is that? Let's take a look at some plots of the confusion matrices. Let's create a function called plot_cm(), that we will use to plot the confusion matrices.
def plot_cm(name, cm):
  classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

  df_cm = pd.DataFrame(cm, index = [i for i in classes], columns = [i for i in classes])
  df_cm = df_cm.round(5)

  plt.figure(figsize = (12,8))
  sns.heatmap(df_cm, annot=True, fmt='g')
  plt.title(name + " Model Confusion Matrix")
  plt.xlabel("Predicted Label")
  plt.ylabel("True Label")
  plt.show()

#Let's run our new function for KNN classifier.
plot_cm("K Nearest Neighbors",knn_cm)
#It seems that while many nevi images were accurately classified, many other images of other classes were incorrectly classified as nevi. Due to our dataset being very imbalanced, the accuracy is misleading, as it is sensitive to imbalanced data. In addition, an AUC ROC score close to 0.5 indicates that the model is not capable of discriminating between the classes very well at all.

#Let's try modifying our KNN model's architecture and hyperparameters to increase our model's performance. We can use a library called hypopt to automate this process through a grid search. We'll automatically try out many possible hyperparameters for our machine learning algorithm to see which give the best performance.
X_gray_test, X_gray_val, y_g_test, y_g_val = train_test_split(X_gray_test, y_test, test_size=0.5, random_state=101)

X_gray_test_flat = np.reshape(X_gray_test,(X_gray_test.shape[0],X_gray_test.shape[1]*X_gray_test.shape[2]))
X_gray_val_flat = np.reshape(X_gray_val,(X_gray_val.shape[0],X_gray_val.shape[1]*X_gray_val.shape[2]))
X_gray_test.shape

#In the variable param_grid we can specify which parameters in our KNN Classifier we want to modify.
param_grid = {
              'n_neighbors' :     [2, 3, 4, 5],
              'weights' :          ['uniform', 'distance'],
              'algorithm' :        ['ball_tree', 'kd_tree', 'brute']
             }

#Let's initialize and fit our grid search optimizer.
gs_knn = GridSearch(model=KNeighborsClassifier(),param_grid=param_grid)
gs_knn.fit(X_g_train_flat.astype(np.float32), y_train.astype(np.float32))
gs_knn_gray = GridSearch(model=KNeighborsClassifier(),param_grid=param_grid)
gs_knn_gray.fit(X_gray_val_flat.astype(np.float32), y_g_val.astype(np.float32))

#Now, the model will be trained with the best hyperparameters. 
y_pred = gs_knn.predict(X_gray_test_flat)
y_pred_proba = gs_knn.predict_proba(X_gray_test_flat)
gs_knn_cm = model_stats("Grid Search KNN",y_g_test,y_pred,y_pred_proba)

#Let's also plot the confusion matrix.
plot_cm("Grid Search KNN",gs_knn_cm)
#Seems like the grid search didn't improve the model's performance. It could be that this ML model is unable to handle the dimensionality of our dataset.
