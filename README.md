# Projeto de Classificação de Penguins

Este repositório contém todos os arquivos utilizados no trabalho de classificação de espécies de pinguins utilizando modelos de aprendizado de máquina.

## Arquivos incluídos

### 1. Dados
- `./data/penguins.csv` — arquivo original
- `./data/penguins_preprocessed.csv` — arquivo pré-processado

### 2. Scripts
- `./scripts/preprocess.py` — script responsável por limpar, tratar valores faltantes e aplicar one-hot encoding.
- `./scripts/train_models.py` — script que treina Logistic Regression, Random Forest e SVM, além de salvar modelos e scaler.
- `./scripts/train_models.py` — script para simulação.

### 3. Modelos
- `./models/log_reg.pkl`
- `./models/random_forest.pkl`
- `./models/svm_rbf.pkl`
- `./models/scaler.pkl`

### 4. Logs
- `./logs/training_logs.txt` — registro do processo de treinamento.

## Como executar

### Pré-processamento
```
python preprocess.py
```

### Treinar modelos
```
python train_models.py
```
### Testar modelos
```
python test_models.py
```
Os modelos serão salvos automaticamente na pasta do projeto.

## Requisitos
- Python 3.8+
- pandas
- scikit-learn
- joblib
- pickle
