import numpy as np
import cv2
import tensorflow as tf
import os
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Caminhos para os modelos
base_dir = os.path.dirname(os.path.dirname(__file__))
unet_model_path = os.path.join(base_dir, "unet", "models", "lungUnet.h5")
classifier_model_path = os.path.join(base_dir, "unet", "models", "keras_model.h5")

# Carregar modelos
unet_model = load_model(unet_model_path, compile=False)
classifier_model = load_model(classifier_model_path, compile=False)

# Classes do modelo de classificação
CLASS_NAMES = ['COVID-19', 'Normal', 'Pneumonia']

# Função para processar imagem
def preprocess_image(image_path, img_size=(256, 256)):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, img_size)
    image = image / 255.0  # Normalização
    return image

# Função para segmentar os pulmões
def segment_lungs(image):
    img_input = np.expand_dims(image, axis=(0, -1))  # Adiciona batch e canal
    mask = unet_model.predict(img_input)[0, :, :, 0]  # Predição
    mask = (mask > 0.5).astype(np.uint8)  # Binariza
    return mask

# Função para classificar a imagem
def classify_lung(image):
    img_input = np.expand_dims(np.stack([image] * 3, axis=-1), axis=0)  # Converte para RGB
    prediction = classifier_model.predict(img_input)
    class_idx = np.argmax(prediction)
    class_name = CLASS_NAMES[class_idx]
    return class_name

# Função para marcar a área da doença
def draw_annotation(original_img, mask, class_name):
    img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(img, class_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    return img

# Função principal
def process_xray(image_path):
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    processed_image = preprocess_image(image_path)
    lung_mask = segment_lungs(processed_image)
    class_name = classify_lung(processed_image)
    result_image = draw_annotation(original_image, lung_mask, class_name)
    
    plt.figure(figsize=(6,6))
    plt.imshow(result_image, cmap='gray')
    plt.axis('off')
    plt.show()
    
# Testar com uma imagem
# Substitua 'sample_xray.jpg' pelo caminho da sua imagem
test_image_path = os.path.join(base_dir, "test_images", "img.png")

# Verificar se o arquivo existe
if os.path.exists(test_image_path):
    process_xray(test_image_path)
else:
    print(f"Arquivo de teste não encontrado: {test_image_path}")
    print("Por favor, adicione uma imagem de teste ou ajuste o caminho.")

