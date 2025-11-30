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


def predict_digit(model, image: Image.Image) -> tuple[int, float]:
    from image_processor import preprocess_image
    
    if image is None:
        return None, None

    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image, verbose=0)
    predicted_digit = np.argmax(predictions[0])
    certainty = np.max(predictions[0]) * 100

    return predicted_digit, certainty
