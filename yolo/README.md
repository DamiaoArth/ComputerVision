# 📂 YOLO

Este diretório contém todos os dados e arquivos necessários para o treinamento, teste e inferência utilizando o modelo YOLOv8.

## 📁 Estrutura
```
yolo/
├── dataset_yolo/
│   ├── images/           # Imagens formatadas para treinamento YOLO
│   ├── labels/           # Labels no formato YOLO (.txt)
│   └── data.yaml         # Arquivo de configuração YOLOv8
├── yolo_labels/          # Labels separadas por classe
│   ├── COVID-19/
│   ├── Normal/
│   ├── Pneumonia/
│   └── README.md
├── maskmodel/            # Dados processados para treinamento com YOLO
│   ├── data/             
│   └── README.md
├── runs/                 # Resultados de treinamento e inferência
│   ├── detect/
│   ├── train/
│   └── weights/
└── README.md
```

## 🔍 Descrição
- **dataset_yolo/**: Imagens e labels organizadas para treinamento e teste do YOLOv8.
- **yolo_labels/**: Diretórios contendo arquivos `.txt` com labels para cada classe específica.
- **maskmodel/**: Dados processados pela U-Net prontos para uso com YOLO.
- **runs/**: Diretório onde os pesos treinados e resultados de detecção são salvos.

## 📌 Notas
- O arquivo `data.yaml` deve ser configurado com os caminhos corretos para `images/` e `labels/`, além de especificar as classes que o modelo será treinado para detectar.
- Os labels `.txt` devem seguir o formato YOLOv8 (classe, x_center, y_center, width, height).

## 📖 Referências
- Documentação oficial do YOLOv8: [YOLOv8 Docs](https://github.com/ultralytics/ultralytics)


