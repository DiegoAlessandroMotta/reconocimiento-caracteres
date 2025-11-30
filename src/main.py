import streamlit as st
from model_handler import load_trained_model, predict_digit
from session_state import initialize_session_state, reset_predictions, get_input_hash
from ui_components import (
    render_mode_selector,
    render_upload_mode,
    render_draw_mode,
    render_input_mode,
    render_predict_button,
    render_results
)

st.set_page_config(
    page_title="Clasificador de Dígitos",
    page_icon="🔢",
    layout="centered"
)

st.title("Clasificador de Dígitos Manuscritos")

cnn_model = load_trained_model()

initialize_session_state()

render_mode_selector()

current_input_hash = get_input_hash()
if st.session_state.last_input_hash is not None and current_input_hash != st.session_state.last_input_hash:
    st.session_state.predictions = []

st.session_state.last_input_hash = current_input_hash

if st.session_state.current_mode == 'upload':
    render_upload_mode()
elif st.session_state.current_mode == 'draw':
    render_draw_mode()
else:
    render_input_mode()

if render_predict_button():
    new_prediction, new_certainty = predict_digit(cnn_model, st.session_state.current_image)
    st.session_state.predictions.append((new_prediction, new_certainty))

if st.session_state.predictions:
    render_results(st.session_state.predictions)
