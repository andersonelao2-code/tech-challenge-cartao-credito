"""Definição e treino dos modelos."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.config import RANDOM_STATE

NOMES = {
    "regressao_logistica": "Regressão Logística",
    "arvore_decisao": "Árvore de Decisão",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Imputação + escala nas numéricas, imputação + one-hot nas categóricas.

    Fica dentro do Pipeline para ser ajustado só no treino (e só no fold de
    treino durante a validação cruzada) — sem vazamento do teste.
    """
    numericas = X.select_dtypes(include="number").columns
    categoricas = X.select_dtypes(exclude="number").columns
    return ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numericas),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categoricas),
    ])


def get_models(X_train: pd.DataFrame, y_train: pd.Series) -> dict[str, Pipeline]:
    """Os quatro candidatos, cada um encapsulado em um Pipeline.

    O desbalanceamento é tratado com pesos de classe (class_weight /
    scale_pos_weight), não com reamostragem: o teste continua com a proporção real.
    """
    peso_positivo = (y_train == 0).sum() / (y_train == 1).sum()
    classificadores = {
        "regressao_logistica": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE,
        ),
        "arvore_decisao": DecisionTreeClassifier(
            max_depth=5, class_weight="balanced", random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=10, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, eval_metric="logloss",
            scale_pos_weight=peso_positivo, random_state=RANDOM_STATE,
        ),
    }
    return {
        nome: Pipeline([("pre", build_preprocessor(X_train)), ("clf", clf)])
        for nome, clf in classificadores.items()
    }
