"""Limpeza, definição do alvo e feature engineering."""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from src.config import (
    DIAS_EMPREGO_SENTINELA,
    GROUP,
    RANDOM_STATE,
    STATUS_ATRASO_60_MAIS,
    TARGET,
    TEST_SIZE,
)


def check_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Resumo de nulos por coluna, em contagem e percentual."""
    total = df.isna().sum()
    pct = (total / len(df) * 100).round(2)
    return (
        pd.DataFrame({"nulos": total, "pct": pct})
        .query("nulos > 0")
        .sort_values("nulos", ascending=False)
    )


def remove_duplicate_ids(application: pd.DataFrame) -> pd.DataFrame:
    """Mantém uma linha por ID.

    A base tem IDs repetidos com dados cadastrais diferentes entre as cópias —
    não há como saber qual é a correta, então fica a primeira ocorrência.
    """
    return application.drop_duplicates(subset="ID", keep="first").reset_index(drop=True)


def add_client_group(application: pd.DataFrame) -> pd.DataFrame:
    """Marca com o mesmo grupo_cliente os IDs que têm o cadastro inteiro idêntico.

    A base registra a mesma pessoa com vários IDs (um por conta/cartão): todas as
    colunas cadastrais coincidem, inclusive a data de nascimento em dias. Se essas
    cópias caírem uma no treino e outra no teste, o modelo é avaliado em quem ele
    já viu. O grupo serve para manter cada pessoa inteira de um lado só do split.
    Precisa ser calculado antes de build_features, que arredonda idade e emprego.
    """
    df = application.copy()
    colunas = [c for c in df.columns if c != "ID"]
    df[GROUP] = df.groupby(colunas, dropna=False, sort=False).ngroup()
    return df


def build_target(credit: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por cliente com o alvo binário.

    mau_pagador = 1 se o cliente teve pelo menos um mês com atraso de 60 dias
    ou mais (STATUS 2, 3, 4 ou 5) no histórico observado; 0 caso contrário.
    """
    atraso = credit["STATUS"].isin(STATUS_ATRASO_60_MAIS)
    alvo = atraso.groupby(credit["ID"]).any().astype(int).rename(TARGET)
    return alvo.reset_index()


def build_history_features(credit: pd.DataFrame) -> pd.DataFrame:
    """Features de histórico por cliente que NÃO usam os status de atraso.

    Contagens de meses em atraso ficam de fora de propósito: são a própria regra
    do alvo e vazariam a resposta para o modelo (ver check_leakage).
    """
    status = credit["STATUS"]
    agrupado = credit.assign(
        quitado=status.eq("C"),
        sem_emprestimo=status.eq("X"),
    ).groupby("ID")
    return agrupado.agg(
        meses_observados=("MONTHS_BALANCE", "nunique"),
        meses_quitado=("quitado", "sum"),
        meses_sem_emprestimo=("sem_emprestimo", "sum"),
    ).reset_index()


def check_leakage(credit: pd.DataFrame, target: pd.DataFrame) -> float:
    """Correlação entre 'teve algum mês em atraso' e o alvo.

    Mostra por que meses_em_atraso não pode ser feature: a correlação é 1.0.
    """
    em_atraso = (
        credit["STATUS"].isin(STATUS_ATRASO_60_MAIS)
        .groupby(credit["ID"]).sum()
        .rename("meses_em_atraso")
        .reset_index()
    )
    juntos = target.merge(em_atraso, on="ID")
    return juntos["meses_em_atraso"].gt(0).astype(int).corr(juntos[TARGET])


def build_features(application: pd.DataFrame) -> pd.DataFrame:
    """Features cadastrais derivadas, em unidades de negócio.

    - idade_anos e anos_empregado: DAYS_* estão em dias negativos contados a
      partir da data da solicitação; em anos ficam legíveis para o negócio.
    - sem_vinculo_emprego: DAYS_EMPLOYED = 365243 é um código da base para quem
      não trabalha (aposentados). Vira uma flag, e anos_empregado fica nulo nesses
      casos (imputado pela mediana dentro do Pipeline, só com dados de treino).
    - OCCUPATION_TYPE nulo vira a categoria "Nao_informado": a ausência é
      informativa (concentrada em aposentados), então não é imputada pela moda.
    - FLAG_MOBIL sai: tem um único valor em toda a base, não discrimina nada.
    """
    df = application.copy()
    sentinela = df["DAYS_EMPLOYED"].eq(DIAS_EMPREGO_SENTINELA)

    df["idade_anos"] = (-df["DAYS_BIRTH"] / 365.25).round(1)
    df["sem_vinculo_emprego"] = sentinela.astype(int)
    df["anos_empregado"] = np.where(sentinela, np.nan, (-df["DAYS_EMPLOYED"] / 365.25).round(1))
    df["OCCUPATION_TYPE"] = df["OCCUPATION_TYPE"].fillna("Nao_informado")

    return df.drop(columns=["DAYS_BIRTH", "DAYS_EMPLOYED", "FLAG_MOBIL"])


def split(df: pd.DataFrame):
    """Split treino/teste por pessoa, estratificado, idêntico em todos os notebooks.

    Usa a primeira dobra de um StratifiedGroupKFold com 1/TEST_SIZE dobras (20% de
    teste): todos os IDs da mesma pessoa (grupo_cliente) ficam do mesmo lado e a
    proporção de maus pagadores é preservada.
    """
    X = df.drop(columns=[TARGET, "ID", GROUP])
    y = df[TARGET]
    cv = StratifiedGroupKFold(n_splits=round(1 / TEST_SIZE), shuffle=True,
                              random_state=RANDOM_STATE)
    treino, teste = next(cv.split(X, y, groups=df[GROUP]))
    return X.iloc[treino], X.iloc[teste], y.iloc[treino], y.iloc[teste]


def split_aleatorio(df: pd.DataFrame):
    """Split 80/20 por linha, ignorando as pessoas repetidas.

    Só para comparação: mostra quanto a avaliação fica otimista quando a mesma
    pessoa pode cair no treino e no teste.
    """
    X = df.drop(columns=[TARGET, "ID", GROUP])
    y = df[TARGET]
    return train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
