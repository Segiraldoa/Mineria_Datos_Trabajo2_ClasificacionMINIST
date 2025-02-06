import streamlit as st
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import gzip
import pickle
import sklearn

def preprocesse_image(image):
  image=image.convert("L")
  image=image.resize((28,28))
  image_array=img_to_array(image)/255.0
  image_array=np.expand_dims(image_array, axis=0)
  return image_array

def load_model():
  filename= "model_trained_classifier.pkl.gz"
  with gzip.open(filename,"rb")as f:
    model=pickle.load(f)
  return model

def main():
  st.title("Clasificación de la base de datos mnist")
  st.markdown("El modelo final fue escalado con StandarScaler y el metodo de regresión fue Kernel Ridge con los hiperparametros alpha:0.1, kernel:rbf")
  st.markdown("Suba la imagen para clasificar")

  uploaded_file = st.file_uploader("Selecciona una imagen",type=["jpg","png","jpeg"])

  if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image,caption="Imagen subida")
    preprocessed_image=preprocesse_image(image)
    st.image(preprocessed_image,caption="Imagen preprocesada")

    if st.button("Clasificar imagen"):
      model=load_model()
      st.markdown("Imagen clasificada")
      prediction=model.predict(preprocessed_image.reshape(1,-1))
      st.markdown(f"La imagen fue clasificada como: {prediction}")


  

if __name__=='__main__':
  main()

