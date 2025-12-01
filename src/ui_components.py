import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from session_state import reset_predictions, get_input_hash

def render_mode_selector():
    st.subheader("Selecciona un método de entrada:")
    col1_mode, col2_mode = st.columns(2)

    with col1_mode:
        if st.button("Subir Imagen", key="select_upload_mode", width="stretch"):
            st.session_state.current_mode = 'upload'
            reset_predictions()

    with col2_mode:
        if st.button("Dibujar Dígito", key="select_draw_mode", width="stretch"):
            st.session_state.current_mode = 'draw'
            reset_predictions()


def render_upload_mode():
    st.write("Selecciona una imagen de un dígito para clasificarla")

    uploaded_file = st.file_uploader(
        "Haz click o arrastra para cargar una imagen",
        type=["jpg", "jpeg", "png", "gif", "bmp", "webp"],
        key="file_uploader_upload_mode"
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.session_state.current_image = image
        col_left, col_center, col_right = st.columns([1, 2, 1])
        with col_center:
            st.image(image=image, width=400)
    else:
        reset_predictions()
        st.info("No hay imagen cargada.")

def render_draw_mode():
    st.write("Dibuja un dígito para que el modelo lo clasifique.")

    col_left, col_center, col_right = st.columns([1, 2, 1])
    with col_center:
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0.0)",
            stroke_width=25,
            stroke_color="#000000",
            background_color="#FFFFFF",
            update_streamlit=True,
            height=300,
            width=400,
            drawing_mode="freedraw",
            key="canvas_draw_mode",
        )

    if canvas_result.json_data is not None and len(canvas_result.json_data["objects"]) > 0:
        drawn_image = Image.fromarray(canvas_result.image_data).convert("L")
        st.session_state.current_image = drawn_image
    else:
        reset_predictions()

def render_input_mode():
    st.info("Selecciona si quieres subir una imagen o dibujar un dígito para clasificarla.")

def render_predict_buttons():
    if st.session_state.current_image is None:
        return None

    col1, col2 = st.columns(2)
    with col1:
        cnn_clicked = st.button("Predecir (CNN)", key="predict_cnn_button")
    with col2:
        hog_clicked = st.button("Predecir (HOG)", key="predict_hog_button")

    if cnn_clicked:
        return 'cnn'
    if hog_clicked:
        return 'hog'
    return None

def render_results(predictions):
    st.subheader("Resultados de la Predicción:")

    col_left, col_center, col_right = st.columns([1,2,1])

    with col_center:
        last_pred = predictions[-1]
        if len(last_pred) == 3:
            pred, certainty, model_tag = last_pred
        else:
            pred, certainty = last_pred
            model_tag = None

        if pred is None:
            st.markdown(f"**No se pudo identificar el dígito** (certeza: {certainty:.2f}%)")
            st.info("La imagen no parece contener un dígito claro y el modelo no está seguro de la clasificación.")
        else:
            if model_tag:
                st.markdown(f"**Predicción: {pred} con una certeza del {certainty:.2f}%** [{model_tag}]")
            else:
                st.markdown(f"**Predicción: {pred} con una certeza del {certainty:.2f}%**")

        if st.session_state.processed_image is not None:
            st.image(st.session_state.processed_image, width=128, caption="Imagen procesada (28×28)")

    for idx, entry in enumerate(reversed(predictions)):
        if idx > 0:
            if idx == 1:
                st.markdown("---")
                st.markdown("**Predicciones previas:**")
            if len(entry) == 3:
                pred, certainty, model_tag = entry
            else:
                pred, certainty = entry
                model_tag = None

            if pred is None:
                if model_tag:
                    st.markdown(f"No clasificado ({certainty:.2f}% de certeza) [{model_tag}]")
                else:
                    st.markdown(f"No clasificado ({certainty:.2f}% de certeza)")
            else:
                if model_tag:
                    st.markdown(f"{pred} con {certainty:.2f}% de certeza [{model_tag}]")
                else:
                    st.markdown(f"{pred} con {certainty:.2f}% de certeza")

    if predictions and predictions[-1][0] is not None:
        st.balloons()
