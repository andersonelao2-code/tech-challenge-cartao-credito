# notebooks/

Ordem numerada obrigatória. Cada notebook roda de cima para baixo em ambiente limpo
(`Kernel → Restart & Run All`) sem erro, e as saídas (tabelas e gráficos) ficam salvas para que o
avaliador veja os resultados sem executar nada.

| Arquivo | Escopo |
|---|---|
| `01_eda.ipynb` | dicionário de variáveis, distribuições, correlações, outliers, balanceamento, taxa de mau pagador por perfil |
| `02_preprocessamento.ipynb` | nulos, duplicatas, definição e justificativa do alvo, checagem de vazamento, features, seleção de atributos, normalização |
| `03_modelagem.ipynb` | split estratificado, 4 modelos (Regressão Logística, Árvore, Random Forest, XGBoost), validação cruzada, treino final |
| `04_avaliacao.ipynb` | métricas no teste, matriz de confusão, curvas ROC e precisão × recall, priorização, cenário sem histórico, permutation importance, SHAP, implicações e limitações |

## Regras seguidas

- Células numeradas em ordem crescente (`[1]`, `[2]`, `[3]`…), resultado de uma execução limpa.
- Todo gráfico tem um parágrafo de interpretação em markdown logo abaixo.
- A primeira célula de cada notebook define `RANDOM_STATE = 42` e importa os caminhos de
  `src/config.py`.
- Código reutilizado entre notebooks fica em `src/`.
- Os notebooks 03 e 04 dependem da saída do 02 (`data/processed/base_modelagem.csv`), e o 04
  depende dos modelos salvos pelo 03 (`results/models/modelos.joblib`).
