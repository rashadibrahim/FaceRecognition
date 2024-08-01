import os
import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Function to load images and labels from a directory
def load_images(directory):
    images = []
    labels = []
    names = []
    label = 0

    for filename in os.listdir(directory):
        personFoler = os.path.join(directory, filename)
        for file in os.listdir(personFoler):
            if file.endswith(".pgm"):
                img_path = os.path.join(personFoler, file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (100, 100))  # Resize for consistency
                images.append(img.flatten())  # Flatten the image 
                labels.append(label)
        names.append(filename)
        label += 1
        
    return np.array(images), np.array(labels), names

# Load images and labels
data_directory = "./AT&T Database of Faces"
images, labels, persons = load_images(data_directory)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Apply PCA for dimensionality reduction
n_components = 50  # You can adjust this based on your dataset and computational resources
pca = PCA(n_components=n_components)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Train a k-Nearest Neighbors (KNN) classifier
knn_classifier = KNeighborsClassifier(n_neighbors=1)
knn_classifier.fit(X_train_pca, y_train)

# Make predictions on the test set
y_pred = knn_classifier.predict(X_test_pca)

# Evaluate the accuracy
accuracy = accuracy_score(y_test, y_pred)
#imgIndex = 1

print("Accuracy: {:.2f}%".format(accuracy * 100))

for imgIndex in range(6):
    print(f"Predicted Label: {persons[knn_classifier.predict([X_test_pca[imgIndex]])[0]]}")
    print(f"Actual Label: {persons[y_test[imgIndex]]}")

    original_image = X_test[imgIndex].reshape(100, 100)
    zoomed_image = cv2.resize(original_image, None, fx=4, fy=4, interpolation=cv2.INTER_LINEAR)
    cv2.imshow('Zoomed Image', zoomed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
# Wait for a key press and then close the window
