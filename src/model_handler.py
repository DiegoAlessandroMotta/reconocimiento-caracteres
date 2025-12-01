import streamlit as st
import numpy as np
import random
import tensorflow as tf
from pathlib import Path
from PIL import Image

@st.cache_resource
def load_trained_model(path=None):
    try:
        if not Path(path).exists():
            raise FileNotFoundError(f"El archivo del modelo no existe: {path}")

        model = tf.keras.models.load_model(str(path))
        return model
    except FileNotFoundError:
        st.error(f"Error: El modelo no se encontró en {path}")
        st.stop()
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        st.stop()

def predict_digit(model, image: Image.Image, certainty_threshold: float = 70.0) -> tuple[int | None, float, Image.Image | None]:
    from image_processor import preprocess_image, preprocess_image_hog
    
    if image is None:
        return None, 0.0, None

    if model is None:
        processed_array, processed_image = preprocess_image(image)
        predicted_digit = random.randint(0, 9)
        certainty = random.random() * 100
        if certainty < certainty_threshold:
            return None, certainty, processed_image
        return predicted_digit, certainty, processed_image

    input_shape = None
    try:
        input_shape = model.input_shape
    except Exception:
        try:
            input_shape = model.layers[0].input_shape
        except Exception:
            input_shape = None

    if input_shape and len(input_shape) == 2:
        processed_array, processed_image = preprocess_image_hog(image)
        processed_array = np.expand_dims(processed_array, axis=0)
    else:
        processed_array, processed_image = preprocess_image(image)
        processed_array = np.expand_dims(processed_array, axis=(0, -1))
    
    predictions = model.predict(processed_array, verbose=0)
    predicted_digit = np.argmax(predictions[0])
    certainty = np.max(predictions[0]) * 100

    if certainty < certainty_threshold:
        return None, certainty, processed_image

    return predicted_digit, certainty, processed_image
