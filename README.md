# CIFAR-10-Classification-with-Supervised-and-Unsupervised-Learning

1. Task Given

The task of this project is to classify an image into one of ten classes using CIFAR- 10
dataset for this classification task. Two approaches implemented are:

I. Supervised Learning Approach (SLA):Build a Neural Network Classifier (NN)
with one hidden layer to be trained and tested on CIFAR-10 dataset.

II. Unsupervised Learning Approach (USLA): USLA is a two-step learning approach
for image classification.
(a) STEP 1: Extract image features using a Convolutional AutoEncoder (ConvAE) forCIFAR- 10 dataset using an open-source neural-network library,
Keras.
(b) STEP 2: Classify Auto-Encoded image features using K-Means clustering
algorithm using sklearns.cluster.KMeans (off-the-shelf clustering libraries)


2. CIFAR-10 Dataset

For training and testing of the classifiers, we will use the CIFAR-10 dataset. The
CIFAR-10 dataset consists of 60000 32x32 color images in 10 classes, with 6000
images per class. There are 50000 training images and 10000 test images. The classes
are completely mutually exclusive.


3. Import Libraries and Load Data (Step 1 & Step 2)

Step 1: Import Libraries
1) NumPy - NumPy is used to work with arrays
2) sklearn.preprocessing.MinMaxScaler - Transform features by scaling each feature to a
given range
3) matplotlib.pyplot - Provides a MATLAB-like plotting framework to plot accuracy and
loss function
4) tensorflow.keras.layers - Layers are the basic building blocks of neural networks in
Keras
5) sklearn.metrics.confusion_matrix - Compute confusion matrix to evaluate the accuracy
of a classification
6) sklearn.metrics.accuracy_score - Accuracy classification score
7) sklearn.cluster.KMeans - K-Means clustering to cluster the input
8) scipy.optimize.linear_sum_assignment – To solve the confusion matrix and calculate
accuracy

Step 2: Data Loading

We use tensorflow.keras.datasets.cifar10.load_data() to load the CIFAR-10 dataset into
Training and Testing data.


Implement SLA and USLA. Both implementation are explained in detail in the report.pdf in the repo with analysis and comparison of model paramters and results of both approaches
