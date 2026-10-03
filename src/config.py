"""Configuração central do projeto: caminhos, semente e constantes.

Importe daqui em todos os notebooks para que os resultados sejam reproduzíveis.
"""

from pathlib import Path

# --- Semente -----------------------------------------------------------------
# Use em TODO ponto que envolva aleatoriedade: train_test_split, modelos, CV.
RANDOM_STATE = 42

# --- Caminhos ----------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
MODELS = RESULTS / "models"
METRICS = RESULTS / "metrics"

# --- Dataset -----------------------------------------------------------------
APPLICATION_FILE = DATA_RAW / "application_record.csv"   # cadastro dos solicitantes
CREDIT_FILE = DATA_RAW / "credit_record.csv"             # histórico mensal de crédito
PROCESSED_FILE = "base_modelagem"                         # saída do notebook 02
TARGET = "mau_pagador"                                    # 1 = mau pagador, 0 = bom pagador
GROUP = "grupo_cliente"                                   # mesma pessoa = mesmo perfil cadastral (vários IDs)

# Códigos de STATUS do credit_record que indicam atraso de 60 dias ou mais
# (legenda oficial: 0 = 1-29 dias, 1 = 30-59, 2 = 60-89, 3 = 90-119, 4 = 120-149,
# 5 = mais de 150 dias ou prejuízo, C = quitado no mês, X = sem empréstimo no mês).
STATUS_ATRASO_60_MAIS = {"2", "3", "4", "5"}

# Valor sentinela de DAYS_EMPLOYED usado na base para quem não tem vínculo
# empregatício (todos os casos são NAME_INCOME_TYPE == "Pensioner").
DIAS_EMPREGO_SENTINELA = 365243

# --- Split -------------------------------------------------------------------
TEST_SIZE = 0.2
CV_FOLDS = 5
