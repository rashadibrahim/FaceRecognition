# FaceRecognition
Face Recognition with Principal Component Analysis (PCA) &amp; K-Nearest Neighbors (KNN) using AT&amp;T Database of Faces

## Dataset:
### The AT&T Database of Faces
The AT&T Database of Faces is a collection of face images captured between April 1992 and April 1994 at the lab. 
This dataset was utilized in a collaborative face recognition project involving the Speech, Vision, and Robotics Group of the Cambridge University Engineering Department. It comprises ten different images for each of the 40 distinct subjects, taken at various times and under different conditions such as varying lighting, facial expressions, and details like the presence or absence of glasses.

Each face image is in Portable Grey Map (PGM) format, viewable on UNIX systems using the 'xv' program, with a size of 92x112 pixels and 256 grey levels per pixel. The dataset is organized into 40 directories, one for each subject, with ten images per subject. The subjects are labeled with names of the form sX, where X indicates the subject number.

Dataset Link: https://www.kaggle.com/datasets/kasikrit/att-database-of-faces

The Folder Names Have Been Cahnged To Actual Names Just For Representation purposes.
So We Can Assume That The Folder Name Is The Person’s Second Name.

![Figure_2](https://github.com/user-attachments/assets/c6d50ff0-68dc-4a8f-a403-04eec404499e)


## Algorithms:
### 1. Principal Component Analysis (PCA) Algorithm
#### Objective:
PCA aims to transform high-dimensional data into a lower-dimensional space while preserving the maximum amount of variance. It achieves this by identifying the principal components, which are linear combinations of the original features.
#### Steps:
1- Covariance Matrix:
PCA starts by computing the covariance matrix of the input data. The covariance matrix represents the relationships between different features.

2- Eigenvalue Decomposition:
The next step involves decomposing the covariance matrix into its eigenvectors and eigenvalues.
Eigenvectors represent the directions of maximum variance, and eigenvalues represent the magnitude of variance along these directions.

3- Selection of Principal Components:
PCA ranks the eigenvectors based on their corresponding eigenvalues. The eigenvectors with higher eigenvalues are selected as principal components.
The number of principal components chosen (controlled by n_components) determines the dimensionality of the reduced data.


We start By Calculating the Covariance Matrix Which quantifies the degree to which two variables change together. In other words, it describes how much one variable tends to increase or decrease when the other variable increases or decreases.
And from it we get the Eigenvectors and Eigenvalues which tells us in which direction our data spread and by how much.
After that we pick the desired number of Eigenvectors that have the biggest Eigenvalues.
Which we use to make a Transformation Matrix That will be used to project the data to lower dimensionality.

![pca](https://github.com/user-attachments/assets/423bf66b-c48b-4645-a5c8-9f41190f45d2)

### 2. K-Nearest Neighbors (KNN) Algorithm
#### Objective:
KNN is a non-parametric, instance-based learning algorithm used for classification and regression tasks. It classifies a new data point based on the majority class of its k-nearest neighbors.
#### Steps:
1- Distance Calculation:
For a given data point, KNN calculates the distances to all other data points in the training set using a distance metric (e.g., Euclidean distance).

2- Neighbor Selection:
The algorithm identifies the k-nearest neighbors based on the smallest distances.

3- Majority Voting:
For classification tasks, the class label of the new data point is determined by majority voting among its k-nearest neighbors.

## Adding User Images:
The AddUserImages.py Code Simply Allows Users To Add Their Images To The Dataset Folder.

So The User Will Enter His Name In the Command Prompt As It Will Be Used As The Folder Name That Will Contain His Images, Which In Turn Will Be Used As A Label For His Images And Then His Camera Will Open And Take 10 Images, Convert Them To Gray Scale And Then Store The Images In The User’s Folder.

Now, If The Model is Trained Again, It Will Include The User Images In the Training Data. 
