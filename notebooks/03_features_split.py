"""Etapas 11-13: feature engineering, balanceamento (diagnostico) e split treino/teste."""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier

pasta_dados = Path(__file__).resolve().parent.parent / "dados"
pasta_resultados = Path(__file__).resolve().parent.parent / "resultados"

application = pd.read_csv(pasta_dados / "application_record.csv").drop_duplicates(subset="ID", keep="first")
credit = pd.read_csv(pasta_dados / "credit_record.csv")
target = pd.read_csv(pasta_resultados / "target.csv")

status_de_risco = {"2", "3", "4", "5"}
log = []

# ---- Etapa 11: feature engineering a partir do historico ----
log.append("=== ETAPA 11: FEATURE ENGINEERING ===")
historico = credit.groupby("ID").agg(
    meses_observados=("MONTHS_BALANCE", "nunique"),
    registros_historico=("STATUS", "size"),
    eventos_risco=("STATUS", lambda valores: sum(str(v) in status_de_risco for v in valores)),
    meses_quitado=("STATUS", lambda valores: sum(str(v) == "C" for v in valores)),
    meses_sem_emprestimo=("STATUS", lambda valores: sum(str(v) == "X" for v in valores)),
).reset_index()
historico["proporcao_meses_risco"] = historico["eventos_risco"] / historico["registros_historico"]

log.append("Features criadas a partir do historico (por cliente):")
log.append("- meses_observados: quantos meses distintos de MONTHS_BALANCE o cliente tem")
log.append("- registros_historico: quantidade total de linhas de status no historico")
log.append("- eventos_risco: quantos meses tiveram status 2/3/4/5 (mesma regra do target,"
            " mas aqui como CONTAGEM, nao como flag binaria -- serve de feature, nao e o target)")
log.append("- meses_quitado: quantos meses o cliente teve status C (quitado)")
log.append("- meses_sem_emprestimo: quantos meses o cliente teve status X")
log.append("- proporcao_meses_risco: eventos_risco / registros_historico")
log.append("\nAmostra de 'historico':\n" + historico.head(5).to_string(index=False))

# --- checagem de leakage: feature quase igual ao target (regra do prompt-mestre, etapa 11) ---
_tmp = historico.merge(target, on="ID", how="inner")
_correlacao = _tmp["eventos_risco"].gt(0).astype(int).corr(_tmp["target"])
log.append(f"\nCHECAGEM DE LEAKAGE: correlacao entre (eventos_risco > 0) e o target = {_correlacao:.4f}")
if _correlacao > 0.95:
    log.append("LEAKAGE CONFIRMADO: eventos_risco foi calculado com a MESMA regra usada pra montar o "
                "target (status em {2,3,4,5}) -- e a resposta disfarcada de feature. "
                "eventos_risco e proporcao_meses_risco (que depende dele) foram REMOVIDAS do conjunto "
                "de features antes do treino. As demais features do historico (meses_observados, "
                "registros_historico, meses_quitado, meses_sem_emprestimo) NAO usam a contagem de "
                "status de risco e foram mantidas.")
historico = historico.drop(columns=["eventos_risco", "proporcao_meses_risco"])

base = application.merge(historico, on="ID", how="left", validate="one_to_one")
base = base.merge(target, on="ID", how="inner", validate="one_to_one")
log.append(f"\nbase.shape apos juntar cadastro + features de historico + target = {base.shape}")
log.append(f"Nulos em features de historico (deveria ser 0, pois so entraram IDs que tem credit_record): "
            f"{base[['meses_observados','registros_historico']].isna().sum().to_dict()}")

base.to_csv(pasta_resultados / "base_features.csv", index=False)

# ---- Etapa 12: balanceamento (diagnostico, sem alterar dados) ----
log.append("\n=== ETAPA 12: BALANCEAMENTO DE CLASSES ===")
contagem = base["target"].value_counts()
proporcao = base["target"].value_counts(normalize=True).round(4)
log.append("Contagem:\n" + contagem.to_string())
log.append("Proporcao:\n" + proporcao.to_string())
log.append(f"\nClasse minoritaria (risco=1) e {proporcao[1]*100:.2f}% da base -- desbalanceamento severo.")
log.append("Decisao: NAO usar SMOTE/oversampling. Usar class_weight='balanced' nos modelos que suportam, "
            "e avaliar com metricas que nao sejam so accuracy (precision/recall/F1/ROC-AUC). "
            "O teste sera mantido com a proporcao REAL (sem balancear), pois precisa representar a "
            "populacao real que o modelo vai encontrar em producao.")

baseline = DummyClassifier(strategy="most_frequent")
log.append(f"\nBaseline ingenuo (sempre prever a classe majoritaria) teria accuracy = "
            f"{proporcao[0]*100:.2f}% -- qualquer modelo real precisa ficar claramente acima disso "
            f"em metricas que enxergam a classe minoritaria (recall/F1/ROC-AUC), nao so accuracy.")

# ---- Etapa 13: split treino/teste ----
log.append("\n=== ETAPA 13: SPLIT TREINO/TESTE ===")
X = base.drop(columns=["target", "ID"])
y = base["target"]
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
log.append(f"X_treino.shape = {X_treino.shape}, X_teste.shape = {X_teste.shape}")
log.append("Proporcao de classes no treino:\n" + y_treino.value_counts(normalize=True).round(4).to_string())
log.append("Proporcao de classes no teste:\n" + y_teste.value_counts(normalize=True).round(4).to_string())

X_treino.to_csv(pasta_resultados / "X_treino.csv", index=False)
X_teste.to_csv(pasta_resultados / "X_teste.csv", index=False)
y_treino.to_csv(pasta_resultados / "y_treino.csv", index=False)
y_teste.to_csv(pasta_resultados / "y_teste.csv", index=False)

for texto in log:
    print(texto)

with open(pasta_resultados / "03_features_split.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(log))
