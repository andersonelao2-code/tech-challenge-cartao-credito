# Tech Challenge — Fase 2 | POSTECH Data Analytics

**Predição de bons e maus pagadores em pedidos de cartão de crédito com aprendizado de máquina.**

---

## 1. Identificação

| Campo | Valor |
|---|---|
| Turma | 12DTAT |
| Grupo | Grupo 12 |
| Data de entrega | 08/10/2026 |

### Integrantes

| Nome completo | RM | E-mail |
|---|---|---|
| Anderson Leão da Silva | RM377734 | andersonleao@bb.com.br |
| Jessika Midory Fukuyama | RM377809 | je.fukuyama@gmail.com |
| Leticia Nery Barbosa Dias | RM377749 | leticianerybd@gmail.com |
| Pricilla Teixeira da Silva | RM377773 | pricillateixeira@bb.com.br |

---

## 2. Links da entrega

Estes três links são **obrigatórios** e devem ser idênticos aos do PDF de submissão.

| Item | Link |
|---|---|
| Repositório | <https://github.com/andersonelao2-code/tech-challenge-cartao-credito> |
| Vídeo executivo (≤ 5 min) | <https://drive.google.com/file/d/1_43ickCRbePLVy5IpjC_-KAQMdxM7fkm/view> |
| Apresentação | <https://github.com/andersonelao2-code/tech-challenge-cartao-credito/blob/main/docs/apresentacao_executiva.pdf> |

---

## 3. O problema

Conceder crédito é a atividade central de uma instituição financeira e também a sua maior fonte de
risco: cada cartão aprovado para um cliente que não paga vira prejuízo. Analisar manualmente todos
os pedidos é caro e lento, e regras fixas ("renda acima de X") deixam passar padrões que só
aparecem na combinação de várias características.

Este projeto usa **classificação supervisionada** para estimar, a partir dos dados cadastrais e do
histórico de crédito, a probabilidade de um cliente ser mau pagador. O objetivo é **ordenar os
pedidos por risco** e concentrar a análise humana onde ela faz mais diferença.

### Variável alvo

A base não traz uma coluna "aprovado/reprovado". O alvo **`mau_pagador`** foi construído a partir
do histórico mensal de pagamento (`credit_record.csv`):

> `mau_pagador = 1` quando o cliente teve **pelo menos um mês com atraso de 60 dias ou mais**
> (`STATUS` 2, 3, 4 ou 5) no histórico observado; `mau_pagador = 0` caso contrário.

**Por que 60 dias**, olhando o pior status já registrado por cliente:

- **Atraso de 1 a 29 dias** é o pior status de **75,4%** dos clientes: é o comportamento comum, não
  indica risco.
- **Corte em 30 dias** marcaria **11,6%** dos clientes, incluindo muitos que atrasaram uma única
  fatura.
- **Corte em 60 dias** marca **1,45%** dos clientes, que deixaram de pagar pelo menos **duas
  faturas seguidas**, um sinal de dificuldade financeira real.

Detalhes em `notebooks/02_preprocessamento.ipynb`, seção 3.

**Distribuição das classes na base de modelagem:** 35.841 bons pagadores (98,31%) e 616 maus
pagadores (1,69%). O desbalanceamento é severo e orienta a escolha de métricas e de modelos.

### Dataset

| Campo | Valor |
|---|---|
| Fonte | [link do Tech Challenge (Google Drive)](https://drive.google.com/file/d/1z4yEyiCE_CGCWbvAAZQZSz-5-E5T5eYd/view?usp=sharing), mesma base pública do Kaggle "Credit Card Approval Prediction" |
| Linhas × colunas | `application_record.csv`: 438.557 × 18 · `credit_record.csv`: 1.048.575 × 3 · base de modelagem: 36.457 × 23 (36.457 contas de 9.728 pessoas) |
| Período / versão | histórico de até 61 meses por cliente (`MONTHS_BALANCE` de −60 a 0); versão distribuída no Tech Challenge Fase 2 |
| Licença de uso | uso acadêmico, conforme disponibilizado pela POSTECH/FIAP |

Descrição das variáveis:

| Variável | Tipo | Descrição |
|---|---|---|
| `ID` | inteiro | identificador do cliente (liga as duas tabelas) |
| `CODE_GENDER` | categórica | gênero |
| `FLAG_OWN_CAR` / `FLAG_OWN_REALTY` | categórica | possui carro / imóvel |
| `CNT_CHILDREN` | inteiro | número de filhos |
| `AMT_INCOME_TOTAL` | contínua | renda anual |
| `NAME_INCOME_TYPE` | categórica | tipo de renda (assalariado, empresário, aposentado, servidor, estudante) |
| `NAME_EDUCATION_TYPE` | categórica | escolaridade |
| `NAME_FAMILY_STATUS` | categórica | estado civil |
| `NAME_HOUSING_TYPE` | categórica | tipo de moradia |
| `DAYS_BIRTH` | inteiro | idade em dias (negativo) → convertida em `idade_anos` |
| `DAYS_EMPLOYED` | inteiro | tempo de emprego em dias (negativo; `365243` = sem vínculo) → `anos_empregado` + `sem_vinculo_emprego` |
| `FLAG_MOBIL` | binária | possui celular (constante, removida) |
| `FLAG_WORK_PHONE` / `FLAG_PHONE` / `FLAG_EMAIL` | binária | possui telefone comercial / fixo / e-mail |
| `OCCUPATION_TYPE` | categórica | ocupação (30,6% nulos → categoria `Nao_informado`) |
| `CNT_FAM_MEMBERS` | contínua | tamanho da família |
| `MONTHS_BALANCE` | inteiro | mês de referência do histórico (0 = atual, −1 = anterior…) |
| `STATUS` | categórica | `0` = 1 a 29 dias de atraso · `1` = 30 a 59 · `2` = 60 a 89 · `3` = 90 a 119 · `4` = 120 a 149 · `5` = mais de 150 dias ou prejuízo · `C` = quitado · `X` = sem empréstimo |
| `meses_observados` | inteiro | *derivada:* meses de histórico do cliente |
| `meses_quitado` | inteiro | *derivada:* meses com status `C` |
| `meses_sem_emprestimo` | inteiro | *derivada:* meses com status `X` |
| `grupo_cliente` | inteiro | *derivada:* identifica a pessoa (IDs com cadastro 100% idêntico); usada só para separar treino e teste, nunca como variável do modelo |
| `mau_pagador` | binária | **alvo** (ver acima) |

---

## 4. Como reproduzir

```bash
git clone https://github.com/andersonelao2-code/tech-challenge-cartao-credito.git
cd tech-challenge-cartao-credito

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

Baixe o dataset e coloque `application_record.csv` e `credit_record.csv` em `data/raw/`. Os dados
**não** são versionados; veja `data/README.md`.

Depois execute os notebooks nesta ordem, no Jupyter, no VS Code ou pela linha de comando:

| # | Notebook | O que faz |
|---|---|---|
| 1 | `notebooks/01_eda.ipynb` | Análise exploratória: distribuições, correlações, outliers, balanceamento |
| 2 | `notebooks/02_preprocessamento.ipynb` | Nulos, duplicatas, pessoas repetidas, definição do alvo, vazamento, features |
| 3 | `notebooks/03_modelagem.ipynb` | Split e validação cruzada por pessoa, efeito das pessoas repetidas, treino de 4 modelos |
| 4 | `notebooks/04_avaliacao.ipynb` | Métricas no teste com intervalo de confiança, priorização, cenário só cadastro, importância de variáveis, conclusões |

Execução de todos pela linha de comando, sem abrir o Jupyter:

```bash
for n in 01_eda 02_preprocessamento 03_modelagem 04_avaliacao; do
  python -c "import nbformat as f, nbclient as c; nb = f.read('notebooks/$n.ipynb', 4); c.NotebookClient(nb, resources={'metadata': {'path': 'notebooks'}}).execute(); f.write(nb, 'notebooks/$n.ipynb')"
done
```

**Semente fixa:** `RANDOM_STATE = 42`, declarada na primeira célula de cada notebook e em
`src/config.py`. Rodar os notebooks na ordem acima, a partir de um ambiente limpo, reproduz
exatamente os números da seção 5.

---

## 5. Resultados

**Treino e teste separados por pessoa.** A base registra a mesma pessoa com vários IDs (uma conta
por cartão): os 36.457 IDs pertencem a só 9.728 pessoas. Num split aleatório, 89% do teste teria a
mesma pessoa no treino, e o modelo seria avaliado em quem ele já viu. Por isso o split e a
validação cruzada usam `StratifiedGroupKFold` por `grupo_cliente`: todos os IDs de uma pessoa ficam
do mesmo lado (`notebooks/02_preprocessamento.ipynb`, seção 7).

Conjunto de teste: 7.295 clientes (20%, estratificado, nenhuma pessoa em comum com o treino), com
124 maus pagadores. Corte de decisão de 0,5.

| Modelo | Acurácia | Acurácia balanceada | Precisão | Recall | F1 | AUC-ROC |
|---|---|---|---|---|---|---|
| **XGBoost** | 0,8355 | 0,6905 | 0,0554 | 0,5403 | 0,1004 | **0,7965** |
| Regressão Logística | 0,7740 | **0,7146** | 0,0480 | 0,6532 | 0,0895 | 0,7880 |
| Random Forest | 0,9069 | 0,6238 | **0,0644** | 0,3306 | **0,1078** | 0,7497 |
| Árvore de Decisão | 0,6458 | 0,6772 | 0,0334 | **0,7097** | 0,0638 | 0,7448 |
| Baseline (sempre "bom") | 0,9830 | 0,5000 | 0,0000 | 0,0000 | 0,0000 | 0,5000 |

Intervalo de 95% do AUC-ROC no teste (bootstrap): XGBoost de 0,757 a 0,831; Regressão Logística de
0,744 a 0,831.

Validação cruzada (5 folds por pessoa, só no treino), AUC-ROC médio ± desvio, e quanto ele subiria
se a divisão ignorasse as pessoas repetidas:

| Modelo | AUC-ROC (por pessoa) | AUC-ROC (por linha) | Otimismo |
|---|---|---|---|
| XGBoost | 0,796 ± 0,028 | 0,824 | +0,027 |
| Regressão Logística | 0,785 ± 0,028 | 0,797 | +0,012 |
| Random Forest | 0,748 ± 0,023 | 0,809 | +0,060 |
| Árvore de Decisão | 0,737 ± 0,016 | 0,739 | +0,003 |

A validação cruzada e o teste ficam praticamente iguais (0,796 contra 0,797), sem sinal de
overfitting. O Random Forest é o mais inflado pela divisão por linha, porque consegue "decorar" a
combinação exata de idade, renda e tempo de emprego de cada pessoa.

**Modelo escolhido:** **XGBoost**, por ter o maior AUC-ROC na validação cruzada, confirmado no
teste. Revisando os **20% de pedidos mais arriscados, ele encontra 63,7% dos maus pagadores**,
contra 20% de uma escolha aleatória. A **Regressão Logística fica tecnicamente empatada** (os
intervalos se sobrepõem) e é a alternativa natural quando for preciso explicar cada decisão.

**Métricas priorizadas:** **AUC-ROC** como principal e **recall** como secundária.

- **Acurácia não serve:** com 98,3% de bons pagadores, o baseline que responde sempre "bom" tem
  98,3% de acurácia e não encontra ninguém.
- **AUC-ROC** mede se o modelo coloca os maus pagadores no topo do ranking, independentemente do
  corte escolhido. É exatamente o uso proposto (priorizar a análise).
- **Recall** vem logo depois porque, em crédito, deixar passar um mau pagador (falso negativo,
  perda do valor emprestado) costuma custar mais que mandar um bom pagador para análise (falso
  positivo).

Todas as métricas, tabelas e gráficos ficam em `results/metrics/` e `results/figures/`.

---

## 6. Principais conclusões

1. **O modelo encontra os maus pagadores muito melhor que o acaso, mas deve priorizar a análise,
   não decidir sozinho.** Revisar os 10% de pedidos mais arriscados já encontra 40% dos maus
   pagadores (4 vezes o acaso), e os 20% mais arriscados encontram cerca de 6 em cada 10. Por outro
   lado, a precisão é de cerca de 6%: recusar automaticamente todo cliente marcado como risco
   negaria crédito a cerca de 17 bons pagadores para cada mau pagador evitado.
2. **O histórico com o banco é o que mais prevê o risco.** As três variáveis mais importantes, com
   concordância entre permutation importance e SHAP, vêm do histórico:
   - **tempo de histórico:** mais meses aumentam o risco;
   - **meses com saldo quitado:** reduzem o risco;
   - **meses sem crédito em uso:** reduzem o risco.

   Entre os dados cadastrais, tempo de emprego e renda maiores indicam risco menor, mas com peso
   pequeno.
3. **O modelo serve para quem já é cliente; o cadastro sozinho quase não prevê o risco.** Treinado
   só com dados cadastrais, o AUC-ROC do XGBoost cai de 0,797 para 0,592 (os outros modelos ficam
   entre 0,55 e 0,59, perto do sorteio), e a captura nos 20% mais arriscados cai de 64% para 37%.
   Para solicitantes novos, a decisão precisa de birôs de crédito externos.
4. **Nenhuma característica isolada define um mau pagador.** Todas as correlações individuais com
   o alvo ficam abaixo de 0,03 (exceto o tempo de histórico, com 0,10). O ganho vem da combinação de
   variáveis, o que justifica um modelo em vez de regras simples.
5. **A qualidade dos dados exigiu decisões explícitas:**
   - o código `365243` em `DAYS_EMPLOYED` identifica aposentados sem vínculo (17% da base);
   - a ocupação é nula em 30,6% dos casos;
   - há 47 IDs duplicados com dados divergentes;
   - a mesma pessoa aparece com vários IDs (36.457 contas de 9.728 pessoas), o que exigiu separar
     treino e teste por pessoa;
   - uma feature com vazamento (correlação 1,0 com o alvo) foi identificada e removida.

### Limitações e próximos passos

- **Tempo de exposição:** a variável mais importante (`meses_observados`) mede em parte há quanto
  tempo o cliente é observado. Mais tempo significa mais chances de registrar um atraso. Não é
  vazamento, mas indica que o modelo mistura "cliente antigo" com "cliente arriscado". Próximo
  passo: exigir um mínimo de meses de histórico ou normalizar pelo tempo de observação.
- **Janela temporal:** features de histórico e alvo vêm do mesmo período. Em produção, as features
  deveriam ser calculadas antes de uma data de corte e o alvo medido depois dela.
- **Identificação de pessoas:** o `grupo_cliente` junta IDs com cadastro 100% idêntico. Se a mesma
  pessoa atualizou algum dado entre uma conta e outra, ela conta como duas pessoas.
- **Amostra pequena:** 124 maus pagadores no teste. Diferenças de 0,01 no AUC-ROC entre modelos não
  são significativas (ver os intervalos de confiança no notebook 04).
- **Equidade:** `CODE_GENDER` influencia a previsão. Antes de qualquer uso real, é necessária uma
  auditoria de equidade (erros por gênero) e a avaliação de remover a variável.
- **Corte de decisão e custos:** o corte de 0,5 é didático. Com os custos reais de um calote e de
  um cliente perdido, dá para escolher o corte que minimiza o prejuízo. Também cabem ajuste de
  hiperparâmetros e calibração das probabilidades.
- **Definição do alvo:** o corte de 60 dias é uma hipótese de modelagem. Testar 30 dias, ou uma
  janela fixa de desempenho, e comparar a estabilidade do modelo.

---

## 7. Estrutura do repositório

```
.
├── data/          dados brutos (raw) e tratados (processed) — não versionados
├── notebooks/     análise em ordem numerada (01 a 04), com saídas salvas
├── src/           funções reutilizadas pelos notebooks (config, dados, features, modelos, métricas)
├── results/       figuras e métricas versionadas; modelos regeneráveis (não versionados)
├── docs/          apresentação executiva
└── submissao/     geração do PDF de submissão com os três links
```

Detalhes e convenções em [`ESTRUTURA.md`](ESTRUTURA.md).
Antes de enviar, percorra o [`CHECKLIST.md`](CHECKLIST.md).

---

## 8. Tecnologias

Python 3.12 · pandas 3.0 · NumPy 2.5 · scikit-learn 1.9 · XGBoost 3.4 · SHAP 0.52 · Matplotlib 3.11 ·
seaborn 0.13 · joblib · Jupyter (ipykernel / nbclient). Versões exatas em `requirements.txt`.
