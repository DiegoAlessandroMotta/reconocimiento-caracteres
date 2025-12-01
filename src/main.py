import streamlit as st
from pathlib import Path
from model_handler import load_trained_model, predict_digit
from session_state import initialize_session_state, reset_predictions, get_input_hash
from ui_components import (
    render_mode_selector,
    render_upload_mode,
    render_draw_mode,
    render_input_mode,
    render_predict_buttons,
    render_results
)

st.set_page_config(
    page_title="Clasificador de Dígitos",
    page_icon="🔢",
    layout="centered"
)

st.title("Clasificador de Dígitos Manuscritos")

initialize_session_state()

cnn_model = st.session_state.cnn_model
hog_model = st.session_state.hog_model

render_mode_selector()

if st.session_state.current_mode == 'upload':
    render_upload_mode()
elif st.session_state.current_mode == 'draw':
    render_draw_mode()
else:
    render_input_mode()

current_input_hash = get_input_hash()
if current_input_hash != st.session_state.last_input_hash:
    st.session_state.predictions = []

st.session_state.last_input_hash = current_input_hash

which_button = render_predict_buttons()
if which_button:
    script_dir = Path(__file__).parent.absolute()
    project_root = script_dir.parent
    if which_button == 'cnn':
        if st.session_state.cnn_model is None:
            st.session_state.cnn_model = load_trained_model(path=project_root / 'model' / 'reconocimiento-caracteres.model.keras')
        cnn_model = st.session_state.cnn_model
        new_prediction, new_certainty, processed_image = predict_digit(cnn_model, st.session_state.current_image)
        model_tag = 'CNN'
    elif which_button == 'hog':
        if st.session_state.hog_model is None:
            st.session_state.hog_model = load_trained_model(path=project_root / 'model' / 'hog-descriptor.model.h5')
        hog_model = st.session_state.hog_model
        new_prediction, new_certainty, processed_image = predict_digit(hog_model, st.session_state.current_image)
        model_tag = 'HOG'

    st.session_state.predictions.append((new_prediction, new_certainty, model_tag))
    st.session_state.processed_image = processed_image

if st.session_state.predictions:
    render_results(st.session_state.predictions)
