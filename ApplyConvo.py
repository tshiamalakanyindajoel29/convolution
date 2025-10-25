import streamlit as st
from PIL import Image
import numpy as np
st.set_page_config(page_title="Convolution Application", layout="centered")

st.title("Convolution Application")

try:
    from model import apply_convolution1
    module_loaded = True
except ImportError:
    st.error("Error: The 'convolution.py' module was not found or the 'apply_convolution()' function is not defined. Ensure the file exists and contains the function.")
    module_loaded = False

uploaded_file = st.file_uploader("Choose an image to convolve", type=["jpg", "jpeg", "png"])

if st.button("Apply convolution to the image"):
    if not module_loaded:
        st.error("Cannot execute convolution: module not loaded.")
    elif uploaded_file is None:
        st.error("Please upload an image first.")
    else:
        st.info("Executing convolution...")
        try:
            image = Image.open(uploaded_file)
            image_array = np.array(image)  
            result_image = apply_convolution(image_array)
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Image")
                st.image(image, use_column_width=True)
            with col2:
                st.subheader("Image After Convolution")
                st.image(result_image, use_column_width=True)
            
            st.success("Convolution completed!")
        except Exception as e:
            st.error(f"An error occurred during convolution: {e}")

st.markdown("---")
st.caption("Ensure that `convolution.py` contains the `apply_convolution(image_array)` function that takes a NumPy array as input and returns the processed image.")
