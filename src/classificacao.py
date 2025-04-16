# classificacao.py
"""
Este script implementa a classificação das imagens em quatro classes: COVID-19, Normal, Pneumonia Viral e Pneumonia Bacteriana.
Utiliza um modelo previamente treinado para fazer a inferência nas imagens de entrada.

## Diretórios de entrada:
- Imagens localizadas em `dataset/{class_name}/images/`
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input

# Caminho para o modelo treinado
model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "unet", "models", "lungUnet.h5")

# Carregar o modelo treinado
model = load_model(model_path)  # Caminho ajustado conforme a estrutura

# Função para carregar e pré-processar a imagem
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))  # Ajuste conforme necessário
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

# Função para realizar a previsão da classe
def predict_class(img_path):
    img = preprocess_image(img_path)
    prediction = model.predict(img)
    return prediction

# Caminho das imagens
image_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset", "COVID-19", "images")  # Exemplo para COVID-19, altere conforme necessário

# Listando as imagens para classificação
for img_name in os.listdir(image_dir):
    img_path = os.path.join(image_dir, img_name)
    pred = predict_class(img_path)
    print(f"Imagem {img_name} - Previsão: {pred}")

