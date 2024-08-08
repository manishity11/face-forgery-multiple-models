import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess_input
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess_input

# Load the trained models
mobilenet_model_path = 'best_model_net.keras'
densenet_model_path = 'best_model_densenet121.keras'

# Custom objects if needed (not likely necessary for MobileNetV2 or DenseNet121)
mobilenet_model = tf.keras.models.load_model(mobilenet_model_path)
densenet_model = tf.keras.models.load_model(densenet_model_path)

# Function to preprocess and predict using a specified model
def preprocess_and_predict(image_path, model, preprocess_input, target_size):
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    predictions = model.predict(img_array)
    return predictions

# Streamlit app
st.title("Deepfake Detection with MobileNetV2 and DenseNet121")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Save the uploaded file temporarily
    with open(os.path.join("temp", uploaded_file.name), "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    
    # Predict using MobileNetV2
    st.subheader("MobileNetV2 Prediction")
    mobilenet_predictions = preprocess_and_predict(os.path.join("temp", uploaded_file.name), mobilenet_model, mobilenet_preprocess_input, (224, 224))
    st.write(f"Predictions: {mobilenet_predictions}")

    # Predict using DenseNet121
    st.subheader("DenseNet121 Prediction")
    densenet_predictions = preprocess_and_predict(os.path.join("temp", uploaded_file.name), densenet_model, densenet_preprocess_input, (224, 224))
    st.write(f"Predictions: {densenet_predictions}")
