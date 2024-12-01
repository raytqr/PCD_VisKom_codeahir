import streamlit as st
import cv2
import numpy as np
import os
import glob
import mahotas as mt
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt

# Function to convert RGB to XYZ
def rgb_to_xyz(rgb):
    rgb = np.array(rgb) / 255.0
    rgb = rgb * 100
    xyz_matrix = np.array([[0.412452, 0.357580, 0.180423],
                           [0.212671, 0.715160, 0.072169],
                           [0.019334, 0.119193, 0.950227]])

    xyz = np.dot(xyz_matrix, rgb)
    return xyz

# Function to convert XYZ to Lab
def xyz_to_lab(xyz):
    white_ref = np.array([95.047, 100.000, 108.883])

    xyz = xyz / white_ref

    mask = xyz > 0.008856
    xyz[mask] = np.cbrt(xyz[mask])
    xyz[~mask] = (7.787 * xyz[~mask]) + (16 / 116)

    L = (116 * xyz[1]) - 16
    a = 500 * (xyz[0] - xyz[1])
    b = 200 * (xyz[1] - xyz[2])

    return np.array([L, a, b])

# Function to convert RGB to Lab
def rgb_to_lab(rgb):
    xyz = rgb_to_xyz(rgb)
    lab = xyz_to_lab(xyz)
    return lab

# Function to convert an image from RGB to Lab
def image_rgb_to_lab(image):
    h, w, _ = image.shape
    lab_image = np.zeros((h, w, 3), dtype=np.float32)
    for i in range(h):
        for j in range(w):
            rgb_pixel = image[i, j]
            lab_pixel = rgb_to_lab(rgb_pixel)
            lab_image[i, j] = lab_pixel

    return lab_image

# Function to extract haralick textures from an image
def extract_haralick_features(image):
    textures = mt.features.haralick(image)
    ht_mean = textures.mean(axis=0)
    return ht_mean

# Function to extract color histogram features (for Lab channels)
def extract_color_histogram(image, bins=256):
    lab_image = image_rgb_to_lab(image)
    hist_features = []
    for channel in range(3):  # L, a, and b channels
        hist = cv2.calcHist([lab_image], [channel], None, [bins], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()  # Normalize and flatten the histogram
        hist_features.extend(hist)  # Combine histograms from L, a, b
    return hist_features

# Function to combine Haralick features and color histogram features, prioritizing color features
def extract_combined_features(image, texture_weight=0.2, color_weight=0.8):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    haralick_features = extract_haralick_features(gray)
    color_features = extract_color_histogram(image)
    
    haralick_features = np.array(haralick_features)
    color_features = np.array(color_features)
    
    combined_features = np.hstack([haralick_features * texture_weight, color_features * color_weight])
    return combined_features

# Streamlit UI setup
st.title("Image Classification with Haralick and Color Histograms")
st.write("This application classifies images using a combination of Haralick texture features and color histograms (Lab color space).")

# Load the training dataset (Ensure the path to your dataset is correct)
train_path = "dataset/train"
train_names = os.listdir(train_path)

# Empty list to hold feature vectors and train labels
train_features = []
train_labels = []

# Initialize classifier
clf_svm = None

# Extract features for training (can be moved to a separate function)
if len(train_features) == 0:  # Ensure it's trained once
    st.write("[STATUS] Extracting training features...")
    for train_name in train_names:
        cur_path = os.path.join(train_path, train_name)
        cur_label = train_name
        for file in glob.glob(os.path.join(cur_path, "*.jpg")):
            image = cv2.imread(file)
            features = extract_combined_features(image)
            train_features.append(features)
            train_labels.append(cur_label)

    st.write(f"Training features shape: {np.array(train_features).shape}")
    st.write(f"Training labels shape: {np.array(train_labels).shape}")

    # Train the classifier once
    clf_svm = LinearSVC(random_state=9, max_iter=10000)
    clf_svm.fit(train_features, train_labels)

# Streamlit file uploader (for testing individual images)
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])

if uploaded_file:
    # Process uploaded image
    image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
    st.image(image, channels="BGR", caption="Uploaded Image")

    # Extract combined features from the uploaded image
    features = extract_combined_features(image)

    # Make prediction using the trained classifier
    if clf_svm:
        prediction = clf_svm.predict(features.reshape(1, -1))[0]
        st.write(f"Prediction: {prediction}")
    else:
        st.write("Classifier is not trained.")

# Test the model with the dataset
st.subheader("Test Images from Dataset")
col1, col2, col3, col4 = st.columns(4)  # Create 4 columns

# Loop over each image and place it in the respective column
for idx, file in enumerate(glob.glob(os.path.join("dataset/test", "*.jpg"))):
    image = cv2.imread(file)
    features = extract_combined_features(image)
    if clf_svm:
        prediction = clf_svm.predict(features.reshape(1, -1))[0]
    else:
        prediction = "No classifier available"
    
    # Display the result
    file_name = os.path.basename(file)
    
    # Resize image to make sure it fits neatly into the columns
    resized_image = cv2.resize(image, (300, 300))  # Adjust size as needed (e.g., 300x300)

    # Show images in columns with consistent size
    with col1 if idx % 4 == 0 else col2 if idx % 4 == 1 else col3 if idx % 4 == 2 else col4:
        st.image(resized_image, channels="BGR", caption=f"{file_name}\nPrediction: {prediction}", use_container_width=True)
