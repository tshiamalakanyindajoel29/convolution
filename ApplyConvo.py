import streamlit as st
from convolution.convolution import apply_convolution
import tempfile
import os

st.set_page_config(page_title="Convolution App", layout="centered")
st.title("Application de Convolution avec TensorFlow")

uploaded_file = st.file_uploader("Choisissez une image à convoluer", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
        temp.write(uploaded_file.read())
        temp_path = temp.name

    if st.button("Appliquer la convolution"):
        st.info("Exécution du script convolution.py ...")
        try:
            apply_convolution(temp_path)
            st.success("Convolution terminée.")
        except Exception as e:
            st.error(f"Erreur : {e}")

st.markdown("---")
st.caption("Téléversez une image puis appliquez la convolution.")
