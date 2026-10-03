"""Métricas e comparação entre modelos."""

import numpy as np
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


def bootstrap_ci(y_true, y_proba, threshold: float = 0.5, n_boot: int = 1000,
                 random_state: int = 42) -> pd.DataFrame:
    """Intervalo de 95% (bootstrap do teste) para ROC-AUC, recall e precisão.

    Com poucos maus pagadores no teste, uma métrica pontual esconde quanto ela
    varia; o intervalo mostra se a diferença entre modelos é maior que o ruído.
    """
    rng = np.random.default_rng(random_state)
    y = np.asarray(y_true)
    p = np.asarray(y_proba)
    amostras = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        pred = (p[i] >= threshold).astype(int)
        amostras.append({
            "roc_auc": roc_auc_score(y[i], p[i]),
            "recall": recall_score(y[i], pred, zero_division=0),
            "precision": precision_score(y[i], pred, zero_division=0),
        })
    df = pd.DataFrame(amostras)
    return pd.DataFrame({
        "ic95_inferior": df.quantile(0.025),
        "ic95_superior": df.quantile(0.975),
    }).round(4)


def recall_at_top(y_true, y_proba, fraction: float) -> float:
    """Fração dos maus pagadores capturados ao revisar os `fraction` mais arriscados."""
    ordem = pd.Series(y_proba).rank(ascending=False, method="first")
    corte = ordem <= len(ordem) * fraction
    y = pd.Series(y_true).reset_index(drop=True)
    return y[corte.values].sum() / y.sum()
