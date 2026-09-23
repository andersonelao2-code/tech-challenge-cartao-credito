"""Etapas 18-24: avaliacao comparativa, matriz de confusao, precision/recall/F1/ROC-AUC, feature importance."""
import pickle
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay,
)
from sklearn.inspection import permutation_importance
from sklearn.dummy import DummyClassifier

pasta_resultados = Path(__file__).resolve().parent.parent / "resultados"

X_teste = pd.read_csv(pasta_resultados / "X_teste.csv")
X_treino = pd.read_csv(pasta_resultados / "X_treino.csv")
y_teste = pd.read_csv(pasta_resultados / "y_teste.csv")["target"]
y_treino = pd.read_csv(pasta_resultados / "y_treino.csv")["target"]
probs = pd.read_csv(pasta_resultados / "probabilidades_teste.csv")
with open(pasta_resultados / "pipelines.pkl", "rb") as f:
    pipelines = pickle.load(f)

log = []

# baseline
baseline = DummyClassifier(strategy="most_frequent").fit(X_treino, y_treino)
pred_baseline = baseline.predict(X_teste)

linhas_tabela = []
THRESHOLD = 0.5
for nome in ["logistica", "arvore", "floresta", "xgboost"]:
    prob = probs[nome].values
    pred = (prob >= THRESHOLD).astype(int)
    linhas_tabela.append({
        "modelo": nome,
        "threshold": THRESHOLD,
        "accuracy": accuracy_score(y_teste, pred),
        "balanced_accuracy": balanced_accuracy_score(y_teste, pred),
        "precision": precision_score(y_teste, pred, zero_division=0),
        "recall": recall_score(y_teste, pred, zero_division=0),
        "f1": f1_score(y_teste, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_teste, prob),
    })
linhas_tabela.append({
    "modelo": "baseline_dummy",
    "threshold": None,
    "accuracy": accuracy_score(y_teste, pred_baseline),
    "balanced_accuracy": balanced_accuracy_score(y_teste, pred_baseline),
    "precision": precision_score(y_teste, pred_baseline, zero_division=0),
    "recall": recall_score(y_teste, pred_baseline, zero_division=0),
    "f1": f1_score(y_teste, pred_baseline, zero_division=0),
    "roc_auc": 0.5,
})
tabela = pd.DataFrame(linhas_tabela).round(4)
log.append("=== ETAPA 18: TABELA COMPARATIVA (threshold=0.5) ===")
log.append(tabela.to_string(index=False))
tabela.to_csv(pasta_resultados / "tabela_metricas.csv", index=False)

melhor_modelo = tabela[tabela["modelo"] != "baseline_dummy"].sort_values("roc_auc", ascending=False).iloc[0]["modelo"]
log.append(f"\nModelo com maior ROC-AUC: {melhor_modelo}")

# ---- Etapa 19: matriz de confusao (de todos, salvando PNG) ----
log.append("\n=== ETAPA 19: MATRIZ DE CONFUSAO ===")
fig, eixos = plt.subplots(1, 4, figsize=(20, 5))
for eixo, nome in zip(eixos, ["logistica", "arvore", "floresta", "xgboost"]):
    pred = (probs[nome].values >= THRESHOLD).astype(int)
    matriz = confusion_matrix(y_teste, pred)
    ConfusionMatrixDisplay(matriz, display_labels=["bom (0)", "risco (1)"]).plot(ax=eixo, cmap="Blues", colorbar=False)
    eixo.set_title(nome)
    log.append(f"{nome}: VN={matriz[0,0]} FP={matriz[0,1]} FN={matriz[1,0]} VP={matriz[1,1]} "
               f"soma={matriz.sum()} (deve bater com len(y_teste)={len(y_teste)})")
plt.tight_layout()
plt.savefig(pasta_resultados / "matrizes_confusao.png", dpi=120)
plt.close()

# ---- Etapa 23: curvas ROC ----
fig, eixo = plt.subplots(figsize=(7, 6))
for nome in ["logistica", "arvore", "floresta", "xgboost"]:
    RocCurveDisplay.from_predictions(y_teste, probs[nome].values, name=nome, ax=eixo)
eixo.plot([0, 1], [0, 1], linestyle="--", color="gray", label="aleatorio (AUC=0.5)")
eixo.set_title("Curvas ROC - comparacao dos 4 modelos")
eixo.legend()
plt.tight_layout()
plt.savefig(pasta_resultados / "curvas_roc.png", dpi=120)
plt.close()
log.append("\nGraficos salvos: matrizes_confusao.png, curvas_roc.png")

# ---- Etapa 24: feature importance (permutation, no melhor modelo) ----
log.append(f"\n=== ETAPA 24: FEATURE IMPORTANCE (permutation, modelo={melhor_modelo}) ===")
resultado_perm = permutation_importance(
    pipelines[melhor_modelo], X_teste, y_teste, scoring="roc_auc", n_repeats=10, random_state=42, n_jobs=-1
)
tabela_importancia = pd.DataFrame({
    "importancia_media": resultado_perm.importances_mean,
    "desvio_padrao": resultado_perm.importances_std,
}, index=X_teste.columns).sort_values("importancia_media", ascending=False)
top10 = tabela_importancia.head(10)
log.append(top10.to_string())

fig, eixo = plt.subplots(figsize=(8, 5))
top10["importancia_media"].sort_values().plot.barh(ax=eixo, xerr=top10["desvio_padrao"].reindex(top10["importancia_media"].sort_values().index))
eixo.set_title(f"Top 10 features por permutation importance ({melhor_modelo})")
eixo.set_xlabel("queda media de ROC-AUC ao embaralhar a coluna")
plt.tight_layout()
plt.savefig(pasta_resultados / "feature_importance.png", dpi=120)
plt.close()

with open(pasta_resultados / "melhor_modelo.txt", "w") as f:
    f.write(melhor_modelo)

for texto in log:
    print(texto)

with open(pasta_resultados / "05_avaliacao.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(log))
