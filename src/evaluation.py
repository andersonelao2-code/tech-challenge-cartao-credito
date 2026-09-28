"""Métricas e comparação entre modelos."""

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import METRICS


def score(y_true, y_pred, y_proba=None) -> dict[str, float]:
    """Conjunto de métricas para classificação binária.

    Acurácia sozinha engana em base desbalanceada — por isso precisão, recall,
    F1, acurácia balanceada e AUC vêm junto.
    """
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_proba is not None:
        out["roc_auc"] = roc_auc_score(y_true, y_proba)
    return out


def comparison_table(results: dict[str, dict], name: str = "comparacao_modelos",
                     sort_by: str = "roc_auc") -> pd.DataFrame:
    """Monta a tabela comparativa e salva em results/metrics/."""
    df = pd.DataFrame(results).T.round(4).sort_values(sort_by, ascending=False)
    df.index.name = "modelo"
    METRICS.mkdir(parents=True, exist_ok=True)
    df.to_csv(METRICS / f"{name}.csv")
    return df


def recall_at_top(y_true, y_proba, fraction: float) -> float:
    """Fração dos maus pagadores capturados ao revisar os `fraction` mais arriscados."""
    ordem = pd.Series(y_proba).rank(ascending=False, method="first")
    corte = ordem <= len(ordem) * fraction
    y = pd.Series(y_true).reset_index(drop=True)
    return y[corte.values].sum() / y.sum()
