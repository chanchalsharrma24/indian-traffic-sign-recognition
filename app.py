"""Traffic Sign Classifier: upload a traffic sign image and get the predicted sign.

Run the notebook first so that models/traffic_sign_cnn.keras and models/class_names.json exist,
then start the app with:  streamlit run app.py
"""

import json

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

IMG_SIZE = (64, 64)  # must match the size used for training


@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("models/traffic_sign_cnn.keras")
    with open("models/class_names.json") as f:
        class_names = {int(k): v for k, v in json.load(f).items()}
    return model, class_names


def preprocess(image):
    """Same steps as in training: RGB, resize to 64x64, scale pixels to 0-1."""
    image = image.convert("RGB").resize(IMG_SIZE)
    array = np.asarray(image, dtype="float32") / 255.0
    return np.expand_dims(array, axis=0)


st.set_page_config(page_title="Traffic Sign Classifier", page_icon="🚦")
st.title("Traffic Sign Classifier")
st.write("Upload an image of an Indian traffic sign to predict its class.")

model, class_names = load_model()
uploaded = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    image = Image.open(uploaded)
    st.image(image, caption="Uploaded image", width=200)

    if st.button("Predict"):
        probabilities = model.predict(preprocess(image), verbose=0)[0]
        top3 = probabilities.argsort()[::-1][:3]

        st.success(f"Predicted traffic sign: **{class_names[top3[0]]}**")
        st.write(f"Confidence: {probabilities[top3[0]]:.2%}")

        st.caption("Other likely signs")
        for i in top3[1:]:
            st.write(f"- {class_names[i]} ({probabilities[i]:.2%})")
