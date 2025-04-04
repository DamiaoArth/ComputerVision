import os
import numpy as np
import tensorflow as tf
import cv2

# Carregar o modelo salvo
best_model_file = "maskmodel/bombardilo/lungUnet.h5"
model = tf.keras.models.load_model(best_model_file)

# Definir dimensões da imagem
Width, Height = 256, 256

# Diretórios de entrada e saída
dataset_dirs = ["dataset/COVID-19", "dataset/Normal", "dataset/Pneumonia"]
mask_output_dirs = ["dataset/mask/COVID-19", "dataset/mask/Normal", "dataset/mask/Pneumonia"]

# Criar diretórios de saída, se não existirem
for output_dir in mask_output_dirs:
    os.makedirs(output_dir, exist_ok=True)

# Processar cada diretório do dataset
for input_dir, output_dir in zip(dataset_dirs, mask_output_dirs):
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Caminhos das imagens
            img_path = os.path.join(input_dir, filename)
            mask_path = os.path.join(output_dir, filename)

            # Carregar e processar imagem
            img = cv2.imread(img_path)
            img_resized = cv2.resize(img, (Width, Height))
            img_normalized = img_resized / 255.0
            imgForModel = np.expand_dims(img_normalized, axis=0)

            # Prever máscara
            prediction = model.predict(imgForModel)
            resultMask = prediction[0]
            
            # Converter para binário
            resultMask[resultMask <= 0.5] = 0
            resultMask[resultMask > 0.5] = 255
            
            # Redimensionar para o tamanho original
            mask_resized = cv2.resize(resultMask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA)
            
            # Salvar a máscara
            cv2.imwrite(mask_path, mask_resized)
            print(f"Máscara salva em: {mask_path}")

print("Processamento concluído!")

