import streamlit as st
import numpy as np
import tensorflow as tf
from pathlib import Path
from PIL import Image

@st.cache_resource
def load_trained_model(path=None):
    if path is None:
        script_dir = Path(__file__).parent.absolute()
        project_root = script_dir.parent
        path = project_root / "model" / "reconocimiento-caracteres.model.keras"
    
    try:
        model = tf.keras.models.load_model(str(path))
        return model
    except FileNotFoundError:
        st.error(f"Error: El modelo no se encontró en {path}")
        st.stop()
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        st.stop()

def predict_digit(model, image: Image.Image, certainty_threshold: float = 70.0) -> tuple[int | None, float]:
    from image_processor import preprocess_image
    
    if image is None:
        return None, 0.0

    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image, verbose=0)
    predicted_digit = np.argmax(predictions[0])
    certainty = np.max(predictions[0]) * 100

    # Si la certeza es menor al threshold, no clasificar
    if certainty < certainty_threshold:
        return None, certainty

    return predicted_digit, certainty
