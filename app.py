import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input as mobilenet_preprocess_input
from tensorflow.keras.applications.convnext import ConvNeXtBase, preprocess_input as convnext_preprocess_input
from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input as densenet_preprocess_input

# Load the trained models
mobilenet_model_path = 'path/to/your/trained_mobilenetv2_model.h5'
convnext_model_path = 'path/to/your/trained_convnext_model.h5'
densenet_model_path = 'path/to/your/trained_densenet121_model.h5'

mobilenet_model = tf.keras.models.load_model(mobilenet_model_path)
convnext_model = tf.keras.models.load_model(convnext_model_path)
densenet_model = tf.keras.models.load_model(densenet_model_path)

st.title("Deepfake Detection with MobileNetV2, ConvNeXt, and DenseNet121")
st.write("Upload an image to predict if it's real or fake using MobileNetV2, ConvNeXt, and DenseNet121 models.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

def preprocess_and_predict(model, img_path, preprocess_input_func):
    target_size = (224, 224)
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input_func(img_array)
    prediction = model.predict(img_array)
    return prediction

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess and predict using MobileNetV2
    mobilenet_prediction = preprocess_and_predict(mobilenet_model, uploaded_file, mobilenet_preprocess_input)
    mobilenet_label = "Fake" if mobilenet_prediction[0][0] > 0.5 else "Real"
    mobilenet_confidence = mobilenet_prediction[0][0]

    # Preprocess and predict using ConvNeXt
    convnext_prediction = preprocess_and_predict(convnext_model, uploaded_file, convnext_preprocess_input)
    convnext_label = "Fake" if convnext_prediction[0][0] > 0.5 else "Real"
    convnext_confidence = convnext_prediction[0][0]

    # Preprocess and predict using DenseNet121
    densenet_prediction = preprocess_and_predict(densenet_model, uploaded_file, densenet_preprocess_input)
    densenet_label = "Fake" if densenet_prediction[0][0] > 0.5 else "Real"
    densenet_confidence = densenet_prediction[0][0]

    # Display the results
    st.write(f"**MobileNetV2 Prediction:** {mobilenet_label} (Confidence: {mobilenet_confidence:.4f})")
    st.write(f"**ConvNeXt Prediction:** {convnext_label} (Confidence: {convnext_confidence:.4f})")
    st.write(f"**DenseNet121 Prediction:** {densenet_label} (Confidence: {densenet_confidence:.4f})")

    # Count the number of Real and Fake predictions
    class_counts = {"Real": 0, "Fake": 0}
    class_counts[mobilenet_label] += 1
    class_counts[convnext_label] += 1
    class_counts[densenet_label] += 1

    st.write(f"**Class Counts:** {class_counts}")

# Function to display examples
def display_examples():
    examples_path = 'path/to/examples/'  # Update with your examples path
    example_images = [os.path.join(examples_path, fname) for fname in os.listdir(examples_path) if fname.endswith(('jpg', 'jpeg', 'png'))]
    
    st.write("Example Images:")
    for example_img_path in example_images:
        example_img = image.load_img(example_img_path, target_size=(224, 224))
        st.image(example_img, caption=os.path.basename(example_img_path), use_column_width=True)

# Button to show examples
if st.button("Show Examples"):
    display_examples()
