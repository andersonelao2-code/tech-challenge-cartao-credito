"""Carregamento e persistência de dados."""

import pandas as pd

from src.config import APPLICATION_FILE, CREDIT_FILE, DATA_PROCESSED


def _exigir(caminho) -> None:
    if not caminho.exists():
        raise FileNotFoundError(
            f"{caminho} não encontrado. Veja data/README.md para baixar o dataset."
        )


def load_application() -> pd.DataFrame:
    """Cadastro dos solicitantes (data/raw/application_record.csv), sem transformação."""
    _exigir(APPLICATION_FILE)
    return pd.read_csv(APPLICATION_FILE)


def load_credit() -> pd.DataFrame:
    """Histórico mensal de crédito (data/raw/credit_record.csv), sem transformação.

    STATUS é lido como texto porque mistura dígitos (0-5) com letras (C, X).
    """
    _exigir(CREDIT_FILE)
    return pd.read_csv(CREDIT_FILE, dtype={"STATUS": str})


def save_processed(df: pd.DataFrame, name: str) -> None:
    """Grava um dataset tratado em data/processed/."""
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PROCESSED / f"{name}.csv", index=False)


def load_processed(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_PROCESSED / f"{name}.csv")
