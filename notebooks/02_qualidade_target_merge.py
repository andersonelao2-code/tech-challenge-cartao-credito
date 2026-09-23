"""Etapas 6-10: dados faltantes, duplicidades, status de credito, target, merge."""
from pathlib import Path
import pandas as pd

pasta_dados = Path(__file__).resolve().parent.parent / "dados"
pasta_resultados = Path(__file__).resolve().parent.parent / "resultados"

application = pd.read_csv(pasta_dados / "application_record.csv")
credit = pd.read_csv(pasta_dados / "credit_record.csv")

log = []

# ---- Etapa 6: dados faltantes ----
log.append("=== ETAPA 6: DADOS FALTANTES ===")
for nome, tabela in {"application": application, "credit": credit}.items():
    nulos = tabela.isna().sum().sort_values(ascending=False)
    log.append(f"\n--- {nome} ---\n" + nulos[nulos > 0].to_string())
    if nulos[nulos > 0].empty:
        log.append(f"(nenhuma coluna com nulo em {nome}, exceto o que aparecer acima)")

# ---- Etapa 7: duplicidades ----
log.append("\n=== ETAPA 7: DUPLICIDADES ===")
log.append(f"application.duplicated().sum() = {application.duplicated().sum()}")
log.append(f"credit.duplicated().sum() = {credit.duplicated().sum()}")
log.append(f"application['ID'].duplicated().sum() = {application['ID'].duplicated().sum()}")
log.append("\ncredit.groupby('ID').size().describe():\n" + credit.groupby("ID").size().describe().to_string())

dup_ids = application[application.duplicated("ID", keep=False)].sort_values("ID")
log.append(f"\nLinhas de application com ID duplicado (todas as ocorrencias): {len(dup_ids)}")
log.append(dup_ids.head(10).to_string())

# Decisao: remover duplicatas EXATAS de application (linha inteira igual), manter so 1a ocorrencia de ID
# duplicado que nao seja exata (investigar antes de decidir)
dup_id_nao_exata = dup_ids[~dup_ids.duplicated(keep=False) | dup_ids.duplicated()]
log.append(f"\nApos remover duplicatas EXATAS de linha inteira, sobram com ID repetido: "
            f"{application.drop_duplicates().duplicated('ID').sum()} IDs")

application_limpo = application.drop_duplicates()  # remove linha 100% identica
log.append(f"application antes de dropar linha-duplicada-exata: {application.shape}")
log.append(f"application depois: {application_limpo.shape}")

# ---- Etapa 8: status de credito ----
log.append("\n=== ETAPA 8: STATUS DE CREDITO ===")
log.append(credit["STATUS"].value_counts(dropna=False).sort_index().to_string())
log.append(f"MONTHS_BALANCE min={credit['MONTHS_BALANCE'].min()}, max={credit['MONTHS_BALANCE'].max()}")
legenda = {
    "C": "quitado no mes", "X": "sem emprestimo no mes",
    "0": "regular (em dia)", "1": "atraso 1-29 dias", "2": "atraso 30-59 dias",
    "3": "atraso 60-89 dias", "4": "atraso 90-119 dias", "5": "atraso 120+ dias / prejuizo",
}
log.append(f"Legenda usada: {legenda}")

# ---- Etapa 9: definicao do target ----
log.append("\n=== ETAPA 9: DEFINICAO DO TARGET ===")
log.append("Regra: risco=1 se o cliente teve pelo menos 1 mes com STATUS em {2,3,4,5} "
            "(atraso >= 30 dias) em todo o historico observado. risco=0 caso contrario.")
status_de_risco = {"2", "3", "4", "5"}
target = (
    credit.groupby("ID")["STATUS"]
    .apply(lambda valores: int(any(str(v) in status_de_risco for v in valores)))
    .rename("target")
    .reset_index()
)
log.append("\nContagem de classes:\n" + target["target"].value_counts().to_string())
log.append("\nProporcao de classes:\n" + target["target"].value_counts(normalize=True).round(4).to_string())

# validacao manual: olhar alguns exemplos de cada classe
exemplo_risco = target[target["target"] == 1]["ID"].head(2).tolist()
exemplo_ok = target[target["target"] == 0]["ID"].head(2).tolist()
for id_exemplo in exemplo_risco + exemplo_ok:
    hist = credit[credit["ID"] == id_exemplo][["MONTHS_BALANCE", "STATUS"]].sort_values("MONTHS_BALANCE")
    log.append(f"\nHistorico do ID {id_exemplo} (target={target.set_index('ID').loc[id_exemplo, 'target']}):\n"
                + hist.to_string(index=False))

target.to_csv(pasta_resultados / "target.csv", index=False)

# ---- Etapa 10: merge ----
log.append("\n=== ETAPA 10: MERGE ===")
log.append(f"application_limpo['ID'].duplicated().sum() = {application_limpo['ID'].duplicated().sum()}")
log.append(f"target['ID'].duplicated().sum() = {target['ID'].duplicated().sum()}")

# ainda pode haver ID duplicado nao-exato em application_limpo; se houver, manter so a 1a ocorrencia
antes = application_limpo.shape[0]
application_unico = application_limpo.drop_duplicates(subset="ID", keep="first")
depois = application_unico.shape[0]
log.append(f"application: {antes} linhas antes de forcar ID unico -> {depois} depois "
            f"(removidas {antes - depois} linhas com ID repetido nao-exato)")

base = application_unico.merge(target, on="ID", how="inner", validate="one_to_one")
log.append(f"\nbase.shape (apos merge inner) = {base.shape}")
log.append(f"base['ID'].is_unique = {base['ID'].is_unique}")
log.append(f"base['target'].isna().sum() = {base['target'].isna().sum()}")
log.append(f"\nIDs em application_unico sem correspondencia em credit_record (ficaram de fora do merge): "
            f"{application_unico.shape[0] - base.shape[0]}")
log.append("(Esperado: application_record.csv traz TODOS os solicitantes de cadastro, mas so quem "
            "efetivamente tem conta de credito aparece em credit_record.csv — por isso o merge 'inner' "
            "reduz bastante a base. Essa e a populacao com dado suficiente pra treinar o modelo.)")

base.to_csv(pasta_resultados / "base_merged.csv", index=False)

for texto in log:
    print(texto)

with open(pasta_resultados / "02_qualidade_target_merge.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(log))
