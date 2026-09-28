# data/

**Nada aqui é versionado.** O `.gitignore` bloqueia o conteúdo destas pastas de propósito:
datasets em Git incham o repositório e frequentemente violam a licença da fonte.

| Pasta | Conteúdo |
|---|---|
| `raw/` | arquivos originais, exatamente como baixados da fonte — nunca editados |
| `processed/` | `base_modelagem.csv`, gerado pelo notebook `02_preprocessamento.ipynb` |

## Como obter

1. Baixe em: <https://drive.google.com/file/d/1z4yEyiCE_CGCWbvAAZQZSz-5-E5T5eYd/view?usp=sharing>
   (link oficial do Tech Challenge Fase 2 — mesma base pública do Kaggle
   "Credit Card Approval Prediction").
2. Salve os dois arquivos como:
   - `data/raw/application_record.csv` — cadastro dos solicitantes (438.557 linhas × 18 colunas)
   - `data/raw/credit_record.csv` — histórico mensal de crédito (1.048.575 linhas × 3 colunas)
3. Checksum SHA-256 dos arquivos usados neste projeto (`sha256sum data/raw/*.csv`):

```
4833f502d02ad94295de3ffe74f665e726a4b04342d2e94f8cec41dce951925b  application_record.csv
ba0006a4734f74422d68b0a7132ad591850be0a6affb535eb1042d207fe4b27e  credit_record.csv
```
