import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]
MODEL_PATH  = "modelsaved.keras"

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

st.title("🥔 Potato Disease Classifier")

uploaded = st.file_uploader("Upload a potato leaf image", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, use_container_width=True)

    model = load_model()

    img = np.array(image.resize((256, 256))) / 255.0
    img = np.expand_dims(img, axis=0)

    predictions = model.predict(img)
    label       = CLASS_NAMES[np.argmax(predictions[0])]
    confidence  = round(float(np.max(predictions[0])) * 100, 2)

    st.subheader(f"Result: {label}")
    st.write(f"Confidence: {confidence}%")
    