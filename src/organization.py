# organization.py
"""
Este script organiza os dados para treinamento e validação, movendo as imagens e máscaras para diretórios específicos.
"""

import os
import shutil

# Diretório base do projeto
base_dir = os.path.dirname(os.path.dirname(__file__))

# Caminho de origem (onde as imagens e máscaras originais estão)
src_images_dir = os.path.join(base_dir, "dataset", "Pneumonia", "images")  # Exemplo para Pneumonia, altere conforme necessário
src_masks_dir = os.path.join(base_dir, "dataset", "Pneumonia", "masks")    # Exemplo para Pneumonia, altere conforme necessário

# Caminho de destino (diretório de imagens e máscaras organizadas)
train_dir = os.path.join(base_dir, "yolo", "dataset_yolo", "images", "train")
val_dir = os.path.join(base_dir, "yolo", "dataset_yolo", "images", "val")
train_masks_dir = os.path.join(base_dir, "yolo", "dataset_yolo", "labels", "train")
val_masks_dir = os.path.join(base_dir, "yolo", "dataset_yolo", "labels", "val")

# Função para mover arquivos para os diretórios de treino/validação
def move_files(src_dir, dest_dir):
    for file in os.listdir(src_dir):
        if file.endswith('.jpg'):  # Ajuste conforme o formato das suas imagens
            src_file = os.path.join(src_dir, file)
            dest_file = os.path.join(dest_dir, file)
            shutil.move(src_file, dest_file)

# Organizar imagens de treino e validação
move_files(src_images_dir, train_dir)
move_files(src_masks_dir, train_masks_dir)

