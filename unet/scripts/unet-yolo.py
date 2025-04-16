import os
import cv2
import numpy as np

# Diretórios
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
mask_dirs = [
    os.path.join(base_dir, "dataset", "mask", "COVID-19"),
    os.path.join(base_dir, "dataset", "mask", "Normal"),
    os.path.join(base_dir, "dataset", "mask", "Pneumonia")
]
output_dirs = [
    os.path.join(base_dir, "yolo", "yolo_labels", "COVID-19"),
    os.path.join(base_dir, "yolo", "yolo_labels", "Normal"),
    os.path.join(base_dir, "yolo", "yolo_labels", "Pneumonia")
]

# Criar diretórios de saída, se não existirem
for output_dir in output_dirs:
    os.makedirs(output_dir, exist_ok=True)

# Função para converter máscara em bounding box no formato YOLO
def mask_to_yolo_bbox(mask_path, img_shape):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    h_img, w_img = img_shape
    bboxes = []
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        x_center = (x + w / 2) / w_img
        y_center = (y + h / 2) / h_img
        w_norm = w / w_img
        h_norm = h / h_img
        bboxes.append(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
    
    return bboxes

# Processar cada máscara e salvar como rótulo YOLO
for mask_dir, output_dir in zip(mask_dirs, output_dirs):
    for filename in os.listdir(mask_dir):
        mask_path = os.path.join(mask_dir, filename)
        img_shape = cv2.imread(mask_path).shape[:2]
        yolo_bboxes = mask_to_yolo_bbox(mask_path, img_shape)
        
        if yolo_bboxes:
            label_path = os.path.join(output_dir, filename.replace(".png", ".txt").replace(".jpg", ".txt"))
            with open(label_path, "w") as f:
                f.write("\n".join(yolo_bboxes))
            print(f"Rótulo salvo: {label_path}")

print("Conversão concluída!")
