#Run this to download data and prepare our environment!
from google.colab.output import eval_js
!pip install -U folium==0.2.1
!pip install -U packaging==20.9

import time
start_time = time.time()

import tensorflow as tf

!pip install tensorflowjs 
import tensorflowjs as tfjs

from google.colab import files

import requests, io, zipfile
import os

# Prepare data

images_1 = os.makedirs('images_1', exist_ok=True)
images_2= os.makedirs('images_2', exist_ok=True)
images_all= os.makedirs('images_all', exist_ok=True)

metadata_path = 'metadata.csv'
image_path_1 = 'images_1.zip'
image_path_2 = 'images_2.zip'
images_rgb_path = 'hmnist_8_8_RGB.csv'

!wget -O metadata.csv 'https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%206%20-%2010%20(Projects)/Project%20-%20(Healthcare%20B)%20Skin%20Cancer%20Diagnosis/metadata.csv'
!wget -O images_1.zip 'https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%206%20-%2010%20(Projects)/Project%20-%20(Healthcare%20B)%20Skin%20Cancer%20Diagnosis/images_1.zip'
!wget -O images_2.zip 'https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%206%20-%2010%20(Projects)/Project%20-%20(Healthcare%20B)%20Skin%20Cancer%20Diagnosis/images_2.zip'
!wget -O hmnist_8_8_RGB.csv 'https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%206%20-%2010%20(Projects)/Project%20-%20(Healthcare%20B)%20Skin%20Cancer%20Diagnosis/hmnist_8_8_RGB.csv'
!unzip -q -o images_1.zip -d images_1 
!unzip -q -o images_2.zip -d images_2 

!pip install patool
import patoolib

import os.path
from os import path

os.makedirs("static/js")
!wget -O static/js/skin_cancer_diagnosis_script.js 'https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%206%20-%2010%20(Projects)/Project%20-%20(Healthcare%20B)%20Skin%20Cancer%20Diagnosis/skin_cancer_diagnosis_script.js'
output = 'static/js/skin_cancer_diagnosis_script.js'

print("Downloaded Data")

#Now, we'll use JavaScript and HTML to construct a simple website that hosts your website, which you can access from your browser. The following code segment provides the public URL from which you can access your website.
print(eval_js("google.colab.kernel.proxyPort(8000)"))
#Whenever you want to host the website, run the code segment below and click on the URL above to check it out.
!python -m http.server 8000


%%writefile index.html
<!-- Demo: https://rganesh22.github.io/htmlmockups/tensorflowjs.html -->
<!-- HTML allows us to create UI elements we can interact with-->

<head>

    <!-- CSS helps us design the feel and style of the UI. In this example, we're using Google's Material Design-->

    <!-- Import Google Icon Font -->
    <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
    <!-- Import CSS for Materialize -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/materialize/1.0.0/css/materialize.min.css">

     <!--Optimize Viewport for Mobile Devices-->
     <meta name="viewport" content="width=device-width, initial-scale=1.0"/>

</head>
<body>

    <!-- Title Bar -->
    <nav>
        <div class="nav-wrapper">
          <a href="#" class="brand-logo">Skin Cancer Diagnosis</a>
        </div>
      </nav>

    <!-- Contain for all other HTML Elements -->
    <div style="padding:5%;">

        <!-- Loading Bar -->
        <h4 id="loadingmodel">Loading ML Model</h4>
        <div id="progressbar" class="progress">
            <div class="indeterminate"></div>
        </div>

        <!-- Image File Input -->
        <div class="file-field input-field">
            <div class="btn">
              <span>Select Image</span>
              <input type="file" accept="image/*" onchange="onFileSelected(event)">
            </div>
            <div class="file-path-wrapper">
              <input class="file-path validate" type="text">
            </div>
          </div>
        
        <!-- Image to be Classified -->
        <img id="image" width="100" height="75"></img>

        <!-- Add New Lines -->

        <br/>
        <br/>
        
        <!-- Button to Perform Classification -->
        <a onclick="predict()" class="waves-effect waves-light btn">Classify Image</a>
        
        <!-- Text Fields for the Prediction and the Probability -->
        <h3>Prediction</h3>

        <b><p id="prediction"></p></b>
        <p id="probability"></p>

    </div>
        
    <!-- Import JS Libraries -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/materialize/1.0.0/js/materialize.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@2.0.0/dist/tf.min.js"></script>
    <script src="static/js/skin_cancer_diagnosis_script.js"></script>

    <!-- Javascript allows us to apply logic to our UI elements and programmatically control the website -->
    <!-- We'll be using Tensorflow JS to perform our model inference -->

    <script>
        
        // Initialize our HTML elements as JS objects

        var imgtag = document.getElementById("image")
        var prediction_text = document.getElementById("prediction")
        var probability_text = document.getElementById("probability")

        var progressbar = document.getElementById("progressbar")
        var loadingmodel = document.getElementById("loadingmodel")

        
    </script>
</body>
