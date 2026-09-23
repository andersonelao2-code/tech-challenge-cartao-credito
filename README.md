# Predição de Aprovação de Cartão de Crédito — Tech Challenge Fase 2 (POSTECH/FIAP)

Modelo de classificação supervisionada para identificar clientes com maior evidência histórica
de risco de crédito, a partir de dados cadastrais e de histórico de pagamento.

## Pergunta de negócio

Com os dados disponíveis no momento da análise, como identificar clientes com maior evidência
histórica de risco? O dataset não traz uma coluna pronta de "aprovado/reprovado" — o target foi
construído a partir do histórico de status de pagamento (ver seção Target).

## Dados

- `dados/application_record.csv` — atributos cadastrais (438.557 linhas, 18 colunas).
- `dados/credit_record.csv` — histórico mensal de status de crédito (1.048.575 linhas, 3 colunas,
  45.985 clientes únicos).
- Fonte: fornecida pelo desafio (link no enunciado).

## Estrutura

```
dados/          application_record.csv, credit_record.csv (não versionados — ver .gitignore)
notebooks/      scripts .py por etapa (01 a 07) + montar_notebook.py + predicao_cartao_credito.ipynb
resultados/     saídas reais de cada etapa (.txt, .csv, .png, tabela_metricas.csv)
```

## Como rodar

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap nbformat nbclient
python notebooks/01_leitura_entendimento.py
python notebooks/02_qualidade_target_merge.py
python notebooks/03_features_split.py
python notebooks/04_modelos.py
python notebooks/05_avaliacao.py
python notebooks/06_shap.py
python notebooks/07_interpretacao_negocio.py
# ou, pra gerar/re-executar o notebook único:
python notebooks/montar_notebook.py
```

## Target

Regra: `risco = 1` se o cliente teve pelo menos um mês com `STATUS` em `{2,3,4,5}` (atraso ≥ 30
dias) em todo o histórico observado; `risco = 0` caso contrário. É uma hipótese acadêmica
documentada, não uma política real de negócio.

Classes extremamente desbalanceadas: **98,31% bons pagadores / 1,69% com evidência de risco**
(616 de 36.457 clientes, após o merge).

## Vazamento de dados encontrado e corrigido

A primeira versão da feature `eventos_risco` (contagem de meses em atraso) tinha correlação
**1.0** com o target — porque foi calculada com a mesma regra que define o target. Isso gerou
acurácia/ROC-AUC = 1.0 artificiais nos 4 modelos, um sinal claro de bug, não de sucesso. A
feature (e sua derivada `proporcao_meses_risco`) foi removida antes do treino final. Ver
`resultados/03_features_split.txt` para a checagem e o log completo.

## Modelos e métricas (teste, 7.292 clientes, threshold 0.5)

| modelo | accuracy | balanced_accuracy | precision | recall | f1 | roc_auc |
|---|---|---|---|---|---|---|
| logística | 0,7652 | 0,7088 | 0,0457 | 0,6504 | 0,0855 | 0,7922 |
| árvore | 0,7770 | 0,6788 | 0,0432 | 0,5772 | 0,0803 | 0,7393 |
| floresta | 0,8829 | 0,6807 | 0,0685 | 0,4715 | 0,1196 | 0,7893 |
| **xgboost** | 0,8382 | 0,7020 | 0,0577 | 0,5610 | 0,1047 | **0,8096** |
| baseline (dummy) | 0,9831 | 0,5000 | 0,0000 | 0,0000 | 0,0000 | 0,5000 |

Melhor modelo por ROC-AUC: **XGBoost**. Tabela completa em `resultados/tabela_metricas.csv`.

## Variáveis mais relevantes (permutation importance + SHAP, concordam entre si)

1. `meses_observados` — tempo de histórico do cliente
2. `meses_quitado` — meses com status "quitado" (C)
3. `meses_sem_emprestimo` — meses com status "sem empréstimo" (X)
4. `DAYS_BIRTH` — idade
5. `AMT_INCOME_TOTAL` — renda

**Limitação importante:** `meses_observados` ser a variável mais importante é esperado pela
própria definição do target (mais tempo observado = mais chances de ter tido um mês ruim) — não
é vazamento (não usa dado futuro), mas indica que parte do que o modelo aprende é "tempo de
exposição", não só "perfil de risco intrínseco". Ver `resultados/07_interpretacao_negocio.txt`
para a discussão completa, incluindo a ressalva de fairness sobre `CODE_GENDER`.

## Limitações e uso responsável

O modelo apoia **priorização de análise**, não deve ser usado para aprovação/rejeição automática
de crédito. Precision baixa (4-7%) no threshold padrão significa muitos falsos positivos; recall
de até ~65% significa que ainda escapa cerca de 1 em cada 3 casos de risco real. Nenhum número
neste projeto foi inventado — todos vêm da execução real da pipeline sobre o dataset fornecido.

## Apresentação e vídeo

Pendente — roteiro de slides e vídeo de até 5 minutos a produzir separadamente (ver documento de
apresentação completo entregue junto com este repositório).
