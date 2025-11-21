import pandas as pd
import pickle
import joblib
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

#Ler o csv
df = pd.read_csv('./data/penguins_preprocessed.csv')

X = df.drop(columns=['species'])
y = df['species']

num_cols = ['bill_length_mm','bill_depth_mm','flipper_length_mm','body_mass_g']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

joblib.dump(scaler, './models/scaler.pkl')

models = {
    'log_reg': LogisticRegression(max_iter=500),
    'random_forest': RandomForestClassifier(n_estimators=300),
    'svm_rbf': SVC(kernel='rbf', C=10, gamma='auto', class_weight='balanced')
}

logs = []
times = {}

#Realiza o training
for name, model in models.items():
    start = time.time()
    model.fit(X_train, y_train)
    end = time.time()
    
    elapsed = end - start
    times[name] = elapsed

    pickle.dump(model, open(f'./models/{name}.pkl', 'wb'))
    logs.append(f'Modelo {name} treinado com sucesso em {elapsed:.4f} segundos.')

#Log
with open('./logs/training_logs.txt', 'w') as f:
    f.write("\n".join(logs))