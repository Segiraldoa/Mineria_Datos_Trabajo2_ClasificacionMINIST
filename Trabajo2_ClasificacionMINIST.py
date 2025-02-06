import streamlit as st
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import gzip
import pickle
import sklearn

def preprocesse_image(image):
    image = image.convert("L")  # Convertir a escala de grises
    image = image.resize((28, 28))  # Redimensionar a 28x28
    image_array = img_to_array(image) / 255.0  # Normalizar
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def load_model():
    filename = "model_trained_classifier.pkl.gz"
    with gzip.open(filename, "rb") as f:
        model = pickle.load(f)
    return model

def main():
    # Configuración de página y estilo
    st.set_page_config(page_title="Clasificación MNIST", page_icon="🔢", layout="centered")
    
    # Estilos CSS personalizados
    st.markdown(
        """
        <style>
        .main {
            background-color: #f4f4f4;
        }
        .stTitle {
            font-size: 36px;
            text-align: center;
            font-weight: bold;
            color: #ffffff;
        }
        .stMarkdown {
            font-size: 18px;
            text-align: center;
            color: #ffffff;
        }
        .stButton button {
            background-color: #008CBA;
            color: white;
            padding: 10px 20px;
            font-size: 16px;
            border-radius: 5px;
        }
        .stButton button:hover {
            background-color: #005f7f;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    
    # Título y descripción
    st.markdown("<h1 class='stTitle'>🔢 Clasificación de Dígitos MNIST</h1>", unsafe_allow_html=True)
    st.markdown("<p class='stMarkdown'>Este modelo escala los datos con StandarScaler y utiliza el método Kernel Ridge Regression con hiperparámetros alpha:0.1 y kernel:rbf para clasificar imágenes de la base de datos MNIST.</p>", unsafe_allow_html=True)
    
    # Subir imagen
    uploaded_file = st.file_uploader("📂 Cargar una imagen", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Imagen Original", use_column_width=True)
        
        # Procesamiento de imagen
        preprocessed_image = preprocesse_image(image)
        with col2:
            st.image(preprocessed_image, caption="Imagen Preprocesada", use_column_width=True)
        
        # Botón para clasificar
        if st.button("🔍 Clasificar Imagen"):
            model = load_model()
            prediction = model.predict(preprocessed_image.reshape(1, -1))
            st.success(f"✅ La imagen fue clasificada como: **{prediction[0]}**")
            
if __name__ == '__main__':
    main()
