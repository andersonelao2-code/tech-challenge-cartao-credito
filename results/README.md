# results/

| Pasta | Conteúdo | Versionado? |
|---|---|---|
| `figures/` | gráficos exportados pelos notebooks (`.png`) | sim |
| `metrics/` | tabelas comparativas (`.csv`) | sim |
| `models/` | modelos treinados (`modelos.joblib`) | **não** (ver `.gitignore`) |

Os modelos podem ser regenerados rodando o notebook 03. As figuras e as métricas ficam versionadas
para que o avaliador veja os resultados sem executar nada.

## Métricas

| Arquivo | Origem | Conteúdo |
|---|---|---|
| `validacao_cruzada.csv` | notebook 03 | média e desvio de AUC-ROC, average precision, recall e F1 em 5 folds |
| `metricas_teste.csv` | notebook 04 | acurácia, acurácia balanceada, precisão, recall, F1 e AUC-ROC no teste, incluindo o baseline |
| `captura_por_fatia.csv` | notebook 04 | fração dos maus pagadores encontrada ao revisar os 5%, 10%, 20% e 30% mais arriscados |
| `cenario_so_cadastro.csv` | notebook 04 | XGBoost com e sem as variáveis de histórico de crédito |
| `permutation_importance.csv` | notebook 04 | queda do AUC-ROC ao embaralhar cada variável (XGBoost) |
| `shap_importancia.csv` | notebook 04 | média do valor absoluto de SHAP por variável (XGBoost) |

## Figuras

| Arquivo | Origem |
|---|---|
| `01_distribuicoes_numericas.png`, `01_distribuicoes_categoricas.png` | EDA: distribuições |
| `01_correlacoes.png` | EDA: correlação de Spearman |
| `01_outliers.png` | EDA: boxplots |
| `01_balanceamento.png`, `01_taxa_mau_pagador_por_categoria.png` | EDA: classes e taxa por perfil |
| `03_validacao_cruzada.png` | comparação dos modelos na validação cruzada |
| `04_matrizes_confusao.png`, `04_curvas_roc_pr.png` | avaliação no teste |
| `04_permutation_importance.png`, `04_shap_summary.png` | importância das variáveis |
