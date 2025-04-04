# 📂 YOLO Labels por Classe

Este diretório contém os arquivos de rótulos (labels) no formato YOLO para as imagens segmentadas do dataset de imagens médicas. Os rótulos são organizados por classe de doença, facilitando a utilização com modelos YOLO.

## 📁 Estrutura
yolo_labels/ 
├── COVID-19/ # Rótulos para imagens com COVID-19 
├── Normal/ # Rótulos para imagens normais 
├── Pneumonia/ # Rótulos para imagens com Pneumonia 
└── README.md

## 🔍 Descrição
- **COVID-19/**: Contém arquivos de rótulo para imagens com diagnóstico de COVID-19.
- **Normal/**: Contém arquivos de rótulo para imagens normais (sem diagnóstico de doença).
- **Pneumonia/**: Contém arquivos de rótulo para imagens com diagnóstico de Pneumonia.

Cada arquivo de rótulo segue o formato YOLO, onde:
- A primeira coluna é o ID da classe (0 para COVID-19, 1 para Normal, 2 para Pneumonia).
- As 4 colunas seguintes representam as coordenadas normalizadas do centro do bounding box, seguidas pela largura e altura normalizadas.

## 📌 Notas
- Esses arquivos de rótulo podem ser usados diretamente para treinar modelos YOLOv8.
- Para cada imagem no dataset, existe um arquivo de rótulo correspondente que contém as informações sobre as anomalias (se houver).

## 📖 Referências
- Formato YOLO de rótulos

