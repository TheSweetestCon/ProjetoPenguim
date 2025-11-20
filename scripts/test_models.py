import pandas as pd
import joblib
import pickle
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

#Caminhos do projeto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

NUM_COLS = ['bill_length_mm','bill_depth_mm','flipper_length_mm','body_mass_g']

#Carregar modelos
def load_models():
    models = {
        "Logistic Regression": pickle.load(open(os.path.join(MODELS_DIR, "log_reg.pkl"), "rb")),
        "Random Forest": pickle.load(open(os.path.join(MODELS_DIR, "random_forest.pkl"), "rb")),
        "SVM (RBF)": pickle.load(open(os.path.join(MODELS_DIR, "svm_rbf.pkl"), "rb"))
    }
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    return models, scaler

#Teste com dataset completo
def test_with_dataset():
    print("\n=== Testando modelos com o dataset completo ===")
    df = pd.read_csv(os.path.join(DATA_DIR, "penguins_preprocessed.csv"))

    X = df.drop(columns=["species"])
    y = df["species"]

    models, scaler = load_models()

    X_scaled = X.copy()
    X_scaled[NUM_COLS] = scaler.transform(X[NUM_COLS])

    for name, model in models.items():
        pred = model.predict(X_scaled)
        print(f"\nModelo: {name}")
        print("Acurácia:", accuracy_score(y, pred))
        print("Precisão:", precision_score(y, pred, average='macro'))
        print("Recall:", recall_score(y, pred, average='macro'))
        print("F1-score:", f1_score(y, pred, average='macro'))

#Input manual
def get_manual_inputs():
    print("\nInforme os atributos para prever a espécie:")
    return {
        "bill_length_mm": float(input("bill_length_mm: ")),
        "bill_depth_mm": float(input("bill_depth_mm: ")),
        "flipper_length_mm": float(input("flipper_length_mm: ")),
        "body_mass_g": float(input("body_mass_g: ")),
        "island_Dream": int(input("island_Dream (0 ou 1): ")),
        "island_Torgersen": int(input("island_Torgersen (0 ou 1): ")),
        "sex_male": int(input("sex_male (0 ou 1): ")),
        "year": 2008
    }

#Sample padrão
def get_sample():
    print("\nUsando o exemplo padrão...")
    return {
        "bill_length_mm": 43.2,
        "bill_depth_mm": 18.1,
        "flipper_length_mm": 197,
        "body_mass_g": 3500,
        "island_Dream": 0,
        "island_Torgersen": 1,
        "sex_male": 1,
        "year": 2008
    }

#Teste com o exemplo
def test_single_example(sample):
    df_template = pd.read_csv(os.path.join(DATA_DIR, "penguins_preprocessed.csv"))
    expected_columns = df_template.drop(columns=["species"]).columns.tolist()

    row = {col: sample.get(col, 0) for col in expected_columns}
    sample_df = pd.DataFrame([row])

    models, scaler = load_models()
    sample_df[NUM_COLS] = scaler.transform(sample_df[NUM_COLS])

    for name, model in models.items():
        pred = model.predict(sample_df)
        print(f"\nModelo: {name} → Espécie prevista: {pred[0]}")

#Menu
if __name__ == "__main__":

    while True:
        print("\n=== MENU DE TESTES ===")
        print("1 - Testar com dataset completo")
        print("2 - Testar com parâmetros manuais")
        print("3 - Testar com sample padrão")
        print("4 - Sair")

        option = input("\nEscolha uma opção: ")

        if option == "1":
            test_with_dataset()

        elif option == "2":
            sample = get_manual_inputs()
            test_single_example(sample)

        elif option == "3":
            sample = get_sample()
            test_single_example(sample)

        elif option == "4":
            print("\nSaindo...")
            break

        else:
            print("\nOpção inválida, tente novamente!")