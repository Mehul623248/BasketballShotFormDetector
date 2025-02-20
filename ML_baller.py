# Conducted a research project looked into tracking correct basketball shooting form.
# We collected different shooting forms from different areas on a court and ran ML models to gauge whether the shooting form was good or not.
# Preprocessing of basketball shooting images using pose detection tools (MMPose)
# Extracted features from images using HOG, LBP, and Gabor
# We utilized Python and Python ML libraries (scikit learn) for the ML models.

import os
import cv2
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score, KFold, cross_val_predict
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from skimage.feature import hog
from skimage.feature import local_binary_pattern
from skimage.feature import graycomatrix
from skimage.feature import graycoprops
from skimage.filters import gabor
from skimage.color import rgb2gray
from sklearn.preprocessing import StandardScaler

# Function to load and preprocess images
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            # Resize the image to a fixed size
            img = cv2.resize(img, (100, 100))
            images.append(img)
    return images

# Load and preprocess data
data = []
labels = []
participants = []

for participant_folder in os.listdir('mmpose_images'):
    participant_path = 'mmpose_images'
    gesture_path = participant_path
    images = load_images_from_folder(gesture_path)
    data.extend(images)
    # print(gesture_folder.split("_")[1])

    participants.extend([participant_folder] * len(images))

for participant in participants:
    if("good" in participant):
        labels.append(1)
    else:
        labels.append(0)


data = np.array(data)
labels = np.array(labels)
participants = np.array(participants)


# Feature extraction
hog_features = []
lbp_features = []
gabor_features = []

for img in data:
    # Histogram of Oriented Gradients (HOG) feature extraction
    hog_feat = hog(img, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=False)
    hog_features.append(hog_feat)

    # Local Binary Patterns (LBP) feature extraction
    lbp_feat = local_binary_pattern(img, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp_feat.ravel(), bins=np.arange(0, 10), range=(0, 10))
    lbp_features.append(lbp_hist)


    # Gabor feature extraction
    gabor_feats, _ = gabor(img, frequency=0.6)  # Extract only real parts, discard imaginary parts
    gabor_energy = np.mean(gabor_feats ** 2, axis=(0, 1)).ravel()  # Calculate energy of the Gabor response
    gabor_features.append(gabor_energy)

# Combine features
hog_features = np.array(hog_features)
lbp_features = np.array(lbp_features)

gabor_features = np.array(gabor_features)

# Normalize feature vectors for other methods
scaler_lbp = StandardScaler()
lbp_features_normalized = scaler_lbp.fit_transform(lbp_features)


scaler_gabor = StandardScaler()
gabor_features_normalized = scaler_gabor.fit_transform(gabor_features)

# Ensure lengths match after feature extraction
assert len(hog_features) == len(lbp_features)  == len(gabor_features) == len(labels) == len(participants)

# Stack normalized feature arrays horizontally
features = np.hstack((hog_features, lbp_features_normalized,gabor_features_normalized))

print(len(participants))
print(len(features))
print(len(labels))

from sklearn.metrics import precision_score, recall_score

# Define classifiers
svm_classifier = SVC(kernel='linear')

# Leave-One-Participant-Out cross-validation
logo = LeaveOneGroupOut()
y_true = labels
y_pred = cross_val_predict(svm_classifier, features, labels, groups=participants, cv=logo)
logo_scores = cross_val_score(svm_classifier, features, labels, groups=participants, cv=logo)

print("Leave-One-Participant-Out Cross-Validation Mean Accuracy:", logo_scores.mean())
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
print(f"SVM LOPO Precision:", precision)
print(f"SVM LOPO Recall:", recall)

from sklearn.ensemble import RandomForestClassifier

# Define the Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Leave-One-Participant-Out cross-validation
logo = LeaveOneGroupOut()
y_true = labels
y_pred = cross_val_predict(rf_classifier, features, labels, groups=participants, cv=logo)
logo_scores = cross_val_score(rf_classifier, features, labels, groups=participants, cv=logo)

print("Leave-One-Participant-Out Cross-Validation Mean Accuracy (Random Forest):", logo_scores.mean())
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
print(f"RF LOPO Precision:", precision)
print(f"RF LOPO Recall:", recall)

# 10-fold cross-validation across all participants
kf = KFold(n_splits=10, shuffle=True)
y_true = labels
y_pred = cross_val_predict(svm_classifier, features, labels, groups=participants, cv=logo)
kf_scores = cross_val_score(svm_classifier, features, labels, cv=kf)


print("10-Fold Cross-Validation Across All Participants Mean Accuracy:", kf_scores.mean())
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
print(f"Across LOPO Precision:", precision)
print(f"Across LOPO Recall:", recall)
