import streamlit as st

def initialize_session_state():
    if 'current_mode' not in st.session_state:
        st.session_state.current_mode = None
    if 'current_image' not in st.session_state:
        st.session_state.current_image = None
    if 'predictions' not in st.session_state:
        st.session_state.predictions = []
    if 'last_input_hash' not in st.session_state:
        st.session_state.last_input_hash = None

def reset_predictions():
    st.session_state.current_image = None
    st.session_state.predictions = []
    st.session_state.last_input_hash = None

def reset_mode_state():
    reset_predictions()
    st.session_state.current_mode = None

def get_current_prediction():
    return st.session_state.predictions[-1] if st.session_state.predictions else (None, 0.0)

def get_input_hash():
    if st.session_state.current_image is not None:
        try:
            return hash(st.session_state.current_image.tobytes())
        except (AttributeError, TypeError):
            return None
    return None
