"""Etapa 25: SHAP no melhor modelo (XGBoost)."""
import pickle
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

pasta_resultados = Path(__file__).resolve().parent.parent / "resultados"

X_treino = pd.read_csv(pasta_resultados / "X_treino.csv")
X_teste = pd.read_csv(pasta_resultados / "X_teste.csv")
# pickle usado só localmente, entre scripts deste mesmo projeto (nunca lido de fonte externa)
with open(pasta_resultados / "pipelines.pkl", "rb") as f:
    pipelines = pickle.load(f)
with open(pasta_resultados / "melhor_modelo.txt") as f:
    melhor_modelo = f.read().strip()

pipe = pipelines[melhor_modelo]
pre = pipe.named_steps["pre"]
modelo = pipe.named_steps["modelo"]

amostra = X_teste.head(200)
amostra_transformada = pre.transform(amostra)
nomes_colunas = pre.get_feature_names_out()

explicador = shap.TreeExplainer(modelo)
valores_shap = explicador.shap_values(amostra_transformada)

plt.figure()
shap.summary_plot(valores_shap, amostra_transformada, feature_names=nomes_colunas, plot_type="bar", show=False, max_display=15)
plt.tight_layout()
plt.savefig(pasta_resultados / "shap_importancia_global.png", dpi=120)
plt.close()

plt.figure()
shap.summary_plot(valores_shap, amostra_transformada, feature_names=nomes_colunas, show=False, max_display=15)
plt.tight_layout()
plt.savefig(pasta_resultados / "shap_summary.png", dpi=120)
plt.close()

import numpy as np
importancia_shap = pd.Series(np.abs(valores_shap).mean(axis=0), index=nomes_colunas).sort_values(ascending=False)

log = [f"Modelo explicado: {melhor_modelo}", f"Amostra usada: {len(amostra)} clientes do teste",
       "\nTop 15 features por |SHAP| medio (importancia global):",
       importancia_shap.head(15).to_string()]

for texto in log:
    print(texto)

with open(pasta_resultados / "06_shap.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(log))
