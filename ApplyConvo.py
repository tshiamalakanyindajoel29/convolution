import streamlit as st==3.13
import sys
sys.path.append(convolution)
from convolution import apply_convolution 

st.set_page_config(page_title="Convolution App", layout="centered")
st.title("Application de Convolution avec TensorFlow")

if st.button("Appliquer la convolution sur l'image"):
    st.info("Exécution du script convolution.py ...")
    try:
        apply_convolution()
        st.success("Convolution terminée. Résultat affiché dans une fenêtre externe.")
    except Exception as e:
        st.error(f"Une erreur est survenue : {e}")

st.markdown("---")
st.caption("Assurez-vous que le fichier convolution.py contient bien la fonction apply_convolution() et que le chemin de l’image est correct.")
