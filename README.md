# Computer Vision Project: Pulmonary Disease Detection

## 📌 Projeto

Este projeto visa desenvolver um pipeline completo de visão computacional para a análise de imagens de raio-X de pulmões com o objetivo de detectar doenças pulmonares, como COVID-19 e Pneumonia, além de diferenciar entre condições normais e patológicas. O projeto utiliza técnicas como Segmentação de Imagens (U-Net) e Detecção de Objetos (YOLOv8).

---

## 📂 Estrutura do Projeto

```
computerVision/
├── dataset/                # Imagens e máscaras organizadas por classes
├── yolo/                   # Dados e modelos YOLOv8
├── unet/                   # Dados, modelos e scripts da U-Net
├── src/                    # Scripts principais do projeto
├── requirements.txt        # Dependências do projeto
├── README.md               # Documentação principal
├── LICENSE                 # Licença do projeto
└── .gitignore              # Arquivos ignorados pelo Git
```

### Diretórios principais:

- `dataset/`: Contém imagens e máscaras organizadas em classes (`COVID-19`, `Normal`, `Pneumonia`).
- `yolo/`: Contém os arquivos preparados para treinamento YOLOv8 e scripts de inferência.
- `unet/`: Inclui os dados processados, modelos treinados e scripts para segmentação usando U-Net.
- `src/`: Scripts principais do projeto que orquestram todo o pipeline.

---

## 🚀 Funcionalidades

- **Segmentação Pulmonar:** Utiliza U-Net para gerar máscaras de áreas de interesse dos pulmões.
- **Detecção de Anomalias:** Uso do YOLOv8 para detecção de anomalias nas imagens segmentadas.
- **Classificação de Imagens:** Diferenciação entre imagens normais e patológicas.

---

## 🔧 Instalação

1. Clone o repositório:
```bash
    git clone https://github.com/seu-usuario/computerVision.git
```

2. Navegue até o diretório do projeto:
```bash
    cd computerVision
```

3. Instale as dependências:
```bash
    pip install -r requirements.txt
```

---

## 📁 Estrutura Detalhada

Cada diretório possui seu próprio arquivo `README.md` explicando o propósito, estrutura e uso dos arquivos presentes.

- [`dataset/`](./dataset/README.md)
- [`yolo/`](./yolo/README.md)
- [`unet/`](./unet/README.md)
- [`src/`](./src/README.md)

---

## 📜 Licença

Este projeto está licenciado sob a Licença MIT. Veja o arquivo [`LICENSE`](./LICENSE) para mais detalhes.

---

## 🤝 Contribuição

Contribuições são bem-vindas! Para contribuir, siga as seguintes etapas:

1. Faça um Fork do projeto.
2. Crie um branch para sua feature (`git checkout -b feature/AmazingFeature`).
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`).
4. Faça o push para o branch (`git push origin feature/AmazingFeature`).
5. Abra um Pull Request.

Sinta-se à vontade para abrir Issues para sugestões ou reportar bugs.

---

## 📧 Contato

Para dúvidas ou sugestões, entre em contato em contatoarthurdamiao@gmail.com.


