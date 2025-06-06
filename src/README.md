# 📂 Src

Este diretório contém os scripts principais que implementam o pipeline completo do projeto, incluindo pré-processamento, treinamento, inferência e avaliação.

## 📁 Estrutura
```
src/
├── classificacao.py          # Script principal para classificação das imagens
├── organization.py           # Organização e estruturação dos dados
├── test.py                   # Testes e validações dos modelos
├── utils.py                  # Funções utilitárias e auxiliares
├── verificacao_modelo.py     # Validação de desempenho e ajustes
└── README.md
```

## 🔍 Descrição
- **classificacao.py**: Implementa a classificação das imagens em `COVID-19`, `Normal`, `Pneumonia Viral` e `Pneumonia Bacteriana`.
- **organization.py**: Estrutura os dados para treinamento e inferência, organizando-os em diretórios adequados.
- **test.py**: Executa testes para avaliar a performance dos modelos treinados.
- **utils.py**: Contém funções auxiliares para pré-processamento de imagens e manipulação de dados.
- **verificacao_modelo.py**: Realiza verificações de qualidade e ajustes finos nos modelos.

## 📌 Notas
- Os scripts foram organizados para cobrir todo o pipeline de pré-processamento, treinamento, inferência e avaliação dos modelos.
- Os resultados dos testes são salvos automaticamente para análise posterior.

## 📖 Referências
- Python 3.10+
- Bibliotecas utilizadas especificadas em `requirements.txt`


