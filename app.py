import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np
from PIL import Image

# --- Page configuration ---
st.set_page_config(page_title="Image Classifier", page_icon="🥦", layout="centered")

# --- Page header and subtext ---
st.title("🥗 Image Classification App")
st.markdown("Upload an image of a fruit or vegetable, and the model will predict what it is.")

# --- Load trained Keras model ---
model = load_model('/Users/salmanmaarouf/Documents/Projects/image_classification/Image_classify.keras')

# --- Class labels for prediction ---
data_cat = [
    'apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum',
    'carrot', 'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant',
    'garlic', 'ginger', 'grapes', 'jalepeno', 'kiwi', 'lemon', 'lettuce',
    'mango', 'onion', 'orange', 'paprika', 'pear', 'peas', 'pineapple',
    'pomegranate', 'potato', 'raddish', 'soy beans', 'spinach', 'sweetcorn',
    'sweetpotato', 'tomato', 'turnip', 'watermelon'
]

# --- Image size for model input ---
img_height = 180
img_width = 180

# --- Image uploader widget ---
uploaded_file = st.file_uploader("Upload an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # --- Display the uploaded image ---
    st.image(uploaded_file, caption="Uploaded Image", width=250)
    
    # --- Preprocess the image ---
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((img_height, img_width))
    img_arr = tf.keras.utils.img_to_array(image)
    img_bat = tf.expand_dims(img_arr, 0)  # Add batch dimension

    # --- Predict and process result ---
    predictions = model.predict(img_bat)
    score = tf.nn.softmax(predictions[0])
    predicted_label = data_cat[np.argmax(score)]
    confidence = np.max(score) * 100

    # --- Display results ---
    st.markdown(f"### Prediction: **{predicted_label.capitalize()}**")
    st.markdown(f"**Confidence:** {confidence:.2f}%")

# --- Footer ---
st.markdown("---")
st.caption("Built with TensorFlow and Streamlit")
