# Arquivo de utilidades para processamento de imagens e manipulação de dados

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from torchvision import transforms

# Funções de pré-processamento de imagens
def load_image(image_path):
    """Carrega uma imagem a partir do caminho especificado"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Imagem não encontrada: {image_path}")
    
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Não foi possível carregar a imagem: {image_path}")
    
    # Converter de BGR para RGB (padrão OpenCV para padrão matplotlib/PIL)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def resize_image(image, target_size=(224, 224)):
    """Redimensiona uma imagem para o tamanho alvo"""
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

def normalize_image(image):
    """Normaliza os valores dos pixels para o intervalo [0, 1]"""
    return image.astype(np.float32) / 255.0

def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """Aplica CLAHE (Contrast Limited Adaptive Histogram Equalization) para melhorar o contraste"""
    if len(image.shape) == 3 and image.shape[2] == 3:  # Imagem colorida
        img_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(img_lab)
        
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        l_clahe = clahe.apply(l)
        
        img_clahe = cv2.merge((l_clahe, a, b))
        return cv2.cvtColor(img_clahe, cv2.COLOR_LAB2RGB)
    else:  # Imagem em escala de cinza
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(image)

def apply_gaussian_blur(image, kernel_size=(5, 5)):
    """Aplica desfoque gaussiano para reduzir ruído"""
    return cv2.GaussianBlur(image, kernel_size, 0)

# Funções de visualização
def plot_image(image, title="Imagem", cmap=None):
    """Plota uma imagem com matplotlib"""
    plt.figure(figsize=(10, 8))
    plt.imshow(image, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def plot_multiple_images(images, titles=None, cmap=None, rows=1):
    """Plota múltiplas imagens em uma grade"""
    n = len(images)
    cols = int(np.ceil(n / rows))
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    
    for i, ax in enumerate(axes):
        if i < n:
            ax.imshow(images[i], cmap=cmap)
            if titles is not None and i < len(titles):
                ax.set_title(titles[i])
            ax.axis('off')
        else:
            fig.delaxes(ax)
    
    plt.tight_layout()
    plt.show()

# Funções de transformação para PyTorch
def get_transform(train=True):
    """Retorna transformações para treinamento ou validação"""
    if train:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

# Funções de manipulação de dados
def create_data_splits(data_dir, val_split=0.15, test_split=0.15):
    """Cria divisões de dados para treinamento, validação e teste"""
    classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    train_data = []
    val_data = []
    test_data = []
    
    for cls in classes:
        class_dir = os.path.join(data_dir, cls)
        files = [os.path.join(class_dir, f) for f in os.listdir(class_dir) 
                if os.path.isfile(os.path.join(class_dir, f))]
        
        np.random.shuffle(files)
        
        n_files = len(files)
        n_val = int(n_files * val_split)
        n_test = int(n_files * test_split)
        n_train = n_files - n_val - n_test
        
        train_data.extend([(f, cls) for f in files[:n_train]])
        val_data.extend([(f, cls) for f in files[n_train:n_train+n_val]])
        test_data.extend([(f, cls) for f in files[n_train+n_val:]])
    
    return train_data, val_data, test_data