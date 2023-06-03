# Inspirit-AI-Skin-Cancer-Diagnosis-2023

In this project, we will be be diagnosing skin lesion images for signs of skin cancer. To perform this task, we'll be working with an array of machine learning methods and models. We'll also be developing a web app to deploy our machine learning models! From there, we'll employ some unsupervised ML tecnhiques for data visualizations and perform skin cancer image segmentation in addition to just classification!

The general outline for this project is as follows:

File 1: Exploring Skin Cancer data and developing basic ML models with Computer Vision

File 2: Developing more advanced ML models and deploying ML to a web app

File 3: Checking for bias in ML models performing skin cancer diagnosis

File 4: Exploring more advanced ML methods for skin cancer diagosis and lesion segmentation

# File 1 Overview

Understanding our dataset

Performing data preprocessing

Learning how to manipulate images with OpenCV

Artificially increasing our dataset's size

Creating basic ML models with our dataset

Understanding our Dataset

Our dataset contains over 10,000 skin lesion images that fall into one of seven classes. These classes are melanocytic nevus, melanoma, benign keratosis, basal cell carcionoma, actinic keratosis, dermatofibroma, and vascular lesions.

Our images are sourced from the HAM10000 dataset which is publically available. Each image image contains RGB data and is of the pixel dimensions 800 x 600. The images in the dataset are collected from a dermoscope, a tool that is used by dermatologists to image skin lesions. A dermoscope enhances images by providing maginification and adequate lighting.
