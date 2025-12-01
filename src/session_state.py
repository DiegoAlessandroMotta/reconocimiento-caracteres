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
    if 'processed_image' not in st.session_state:
        st.session_state.processed_image = None
    if 'cnn_model' not in st.session_state:
        st.session_state.cnn_model = None
    if 'hog_model' not in st.session_state:
        st.session_state.hog_model = None

def reset_predictions():
    st.session_state.current_image = None
    st.session_state.predictions = []
    st.session_state.last_input_hash = None
    st.session_state.processed_image = None

def reset_mode_state():
    reset_predictions()
    st.session_state.current_mode = None

def get_current_prediction():
    if not st.session_state.predictions:
        return (None, 0.0, None)
    last = st.session_state.predictions[-1]
    if len(last) == 3:
        return last
    elif len(last) == 2:
        pred, certainty = last
        return (pred, certainty, None)
    else:
        try:
            pred = last[0]
            certainty = float(last[1])
            return (pred, certainty, None)
        except Exception:
            return (None, 0.0, None)

def get_input_hash():
    if st.session_state.current_image is not None:
        try:
            return hash(st.session_state.current_image.tobytes())
        except (AttributeError, TypeError):
            return None
    return None
