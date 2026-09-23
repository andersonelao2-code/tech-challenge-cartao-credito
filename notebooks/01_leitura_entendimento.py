"""Etapa 3-4: leitura dos CSVs e entendimento das colunas."""
import json
from pathlib import Path
import pandas as pd

pasta_dados = Path(__file__).resolve().parent.parent / "dados"
pasta_resultados = Path(__file__).resolve().parent.parent / "resultados"
pasta_resultados.mkdir(exist_ok=True)

application = pd.read_csv(pasta_dados / "application_record.csv")
credit = pd.read_csv(pasta_dados / "credit_record.csv")

log = []
log.append(f"application.shape = {application.shape}")
log.append(f"credit.shape = {credit.shape}")
log.append(f"application.columns = {application.columns.tolist()}")
log.append(f"credit.columns = {credit.columns.tolist()}")
log.append("\n--- application.dtypes ---\n" + str(application.dtypes))
log.append("\n--- credit.dtypes ---\n" + str(credit.dtypes))
log.append("\n--- application.nunique() ---\n" + str(application.nunique().sort_values().to_string()))
log.append("\n--- credit.nunique() ---\n" + str(credit.nunique().sort_values().to_string()))

for texto in log:
    print(texto)

with open(pasta_resultados / "01_leitura_entendimento.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(log))
