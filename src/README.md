# src/

Funções reutilizadas por mais de um notebook.

| Arquivo | Responsabilidade |
|---|---|
| `config.py` | caminhos, semente, nome do alvo, códigos de status de atraso, código sentinela de `DAYS_EMPLOYED` |
| `data.py` | carregar os CSVs brutos e salvar/carregar a base tratada |
| `preprocessing.py` | nulos, duplicatas, construção do alvo, features de histórico e cadastrais, checagem de vazamento, split |
| `models.py` | pré-processador (imputação, escala, one-hot) e os 4 modelos em `Pipeline` |
| `evaluation.py` | métricas de classificação, tabela comparativa, captura de maus pagadores por fatia do ranking |

Nos notebooks: `from src.config import RANDOM_STATE`
