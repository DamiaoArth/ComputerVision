# 📂 U-Net

Este diretório contém todos os dados, modelos e scripts relacionados à segmentação pulmonar utilizando a U-Net.

## 📁 Estrutura
```
unet/
├── data/
│   ├── ClinicalReadings/  # Leituras clínicas associadas às imagens
│   ├── CXR_png/           # Imagens de raio-X em formato PNG
│   ├── ManualMask/        # Máscaras manuais para treinamento
│   ├── train_imgs.npy     # Imagens pré-processadas para treinamento
│   ├── train_masks.npy    # Máscaras pré-processadas para treinamento
│   ├── valid_imgs.npy     # Imagens pré-processadas para validação
│   └── valid_masks.npy    # Máscaras pré-processadas para validação
├── models/
│   ├── keras_model.h5     # Modelo treinado usando Keras
│   └── lungUnet.h5        # Modelo U-Net para segmentação pulmonar
├── scripts/
│   ├── stp1unet.py        # Pré-processamento das imagens
│   ├── stp2unet.py        # Treinamento do modelo U-Net
│   ├── stp3unet.py        # Avaliação do modelo
│   ├── stp4unet.py        # Geração de máscaras preditas
│   ├── stp5unet.py        # Conversão de máscaras para YOLO
│   └── unet-yolo.py       # Integração com YOLO
├── __pycache__/           # Arquivos cacheados do Python
└── README.md
```

## 🔍 Descrição
- **data/**: Diretório contendo imagens e máscaras organizadas para treinamento e validação.
- **models/**: Modelos treinados e prontos para inferência.
- **scripts/**: Scripts para treinamento, inferência e integração com YOLO.

## 📌 Notas
- Os arquivos `.npy` são gerados durante o pré-processamento e facilitam o treinamento do modelo.
- Os scripts foram organizados para realizar todo o pipeline de segmentação e preparação dos dados para uso com YOLO.

## 📖 Referências
- Arquitetura U-Net: [Paper Original](https://arxiv.org/abs/1505.04597)
- Keras Documentation: [Keras Docs](https://keras.io/)


