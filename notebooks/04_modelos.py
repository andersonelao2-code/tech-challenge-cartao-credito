"""Etapas 14-17: treino dos 4 modelos (Regressao Logistica, Arvore, Random Forest, XGBoost)."""
import time
import pickle
from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

pasta_resultados = Path(__file__).resolve().parent.parent / "resultados"

X_treino = pd.read_csv(pasta_resultados / "X_treino.csv")
X_teste = pd.read_csv(pasta_resultados / "X_teste.csv")
y_treino = pd.read_csv(pasta_resultados / "y_treino.csv")["target"]
y_teste = pd.read_csv(pasta_resultados / "y_teste.csv")["target"]

numericas = X_treino.select_dtypes(include="number").columns
categoricas = X_treino.select_dtypes(exclude="number").columns

log = [f"Colunas numericas ({len(numericas)}): {numericas.tolist()}",
       f"Colunas categoricas ({len(categoricas)}): {categoricas.tolist()}"]

pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("esc", StandardScaler())]), numericas),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categoricas),
])

modelos = {
    "logistica": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    "arvore": DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=42),
    "floresta": RandomForestClassifier(n_estimators=300, max_depth=10, class_weight="balanced", random_state=42, n_jobs=-1),
    "xgboost": XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8,
                              colsample_bytree=0.8, eval_metric="logloss", random_state=42,
                              scale_pos_weight=(y_treino == 0).sum() / (y_treino == 1).sum()),
}

pipelines = {}
probabilidades = {}
for nome, modelo in modelos.items():
    inicio = time.time()
    pipe = Pipeline([("pre", pre), ("modelo", modelo)])
    pipe.fit(X_treino, y_treino)
    duracao = time.time() - inicio
    prob = pipe.predict_proba(X_teste)[:, 1]
    pipelines[nome] = pipe
    probabilidades[nome] = prob
    log.append(f"\n{nome}: treinado em {duracao:.1f}s | "
               f"len(prob)={len(prob)} | prob min={prob.min():.4f} max={prob.max():.4f} media={prob.mean():.4f}")

# pickle usado só localmente, entre scripts deste mesmo projeto (nunca lido de fonte externa)
with open(pasta_resultados / "pipelines.pkl", "wb") as f:
    pickle.dump(pipelines, f)
pd.DataFrame(probabilidades).to_csv(pasta_resultados / "probabilidades_teste.csv", index=False)

for texto in log:
    print(texto)

with open(pasta_resultados / "04_modelos.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(log))
