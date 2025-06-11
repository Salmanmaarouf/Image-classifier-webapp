# 🥗 Fruit & Vegetable Image Classifier

This is a web-based image classification app built with **Streamlit** and **TensorFlow**. It allows users to upload an image of a fruit or vegetable, and the model will predict what it is, along with a confidence score.

## 🚀 Live Demo

👉 [Click here to try the app](https://fruitandvegimageclassification.streamlit.app/)

## 🧠 Model

The model was trained on a dataset of 36 fruit and vegetable categories using TensorFlow/Keras. It was saved in `.keras` format and is loaded into the app for prediction.

## 📁 Project Structure

image-classifier-webapp/
├── app.py # Streamlit frontend
├── Image_classify.keras # Trained model file
├── requirements.txt # Python dependencies
├── runtime.txt # Python version for Streamlit Cloud
└── README.md # This file


## 🧪 How It Works

1. User uploads an image (JPG, PNG, or JPEG)
2. Image is resized to 180x180 and preprocessed
3. Model predicts class probabilities
4. Top prediction is shown along with confidence score

## 🖼 Supported Classes

- Apple, Banana, Beetroot, Bell Pepper, Cabbage, Carrot, Cauliflower, Chilli Pepper, Corn, Cucumber, Eggplant, Garlic, Ginger, Grapes, Jalepeno, Kiwi, Lemon, Lettuce, Mango, Onion, Orange, Paprika, Pear, Peas, Pineapple, Pomegranate, Potato, Raddish, Soy Beans, Spinach, Sweetcorn, Sweetpotato, Tomato, Turnip, Watermelon, etc.

## 🧰 Tech Stack

- Python
- TensorFlow / Keras
- Streamlit
- Pillow (for image processing)
- NumPy

## 🧠 Model Training

The image classification model was trained using **TensorFlow's Keras API** on a labeled dataset of fruits and vegetables. The training pipeline involved:

- 📂 **Dataset**: Custom folder-structured image dataset with 36 class folders (e.g. `train/apple`, `train/tomato`, etc.)
- 🧹 **Preprocessing**:
  - Resized all images to `180x180`
  - Normalized pixel values to `[0, 1]`
  - Augmented training data (e.g., flip, rotation)
- 🧠 **Model Architecture**:
  - Sequential CNN with:
    - Conv2D, MaxPooling2D, Dropout layers
    - Dense layers with ReLU and softmax output
  - Categorical cross-entropy loss
  - Adam optimizer
- 🏋️ **Training**:
  - Epochs: 10–20 (adjusted based on overfitting)
  - Achieved high training accuracy (>95%)
- 💾 **Saved Model**:
  - Saved using `model.save('Image_classify.keras')` in the `.keras` format
  - Compatible with TensorFlow 2.x+

## ⚙️ Installation (Local)

```bash
git clone https://github.com/Salmanmaarouf/image-classifier-webapp.git
cd image-classifier-webapp
pip install -r requirements.txt
streamlit run app.py
