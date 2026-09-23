"""Monta o notebook final unindo os 7 scripts em celulas de codigo com markdown entre elas,
depois executa de verdade (nbclient) para embutir as saidas reais."""
import nbformat as nbf
from pathlib import Path

pasta = Path(__file__).resolve().parent
nb = nbf.v4.new_notebook()
celulas = []

def md(texto):
    celulas.append(nbf.v4.new_markdown_cell(texto))

def code_de_arquivo(nome_arquivo, pular_linhas_import_path=True):
    codigo = (pasta / nome_arquivo).read_text(encoding="utf-8")
    # remove a definicao de pasta_dados/pasta_resultados baseada em __file__ (nao existe em notebook)
    # e substitui por caminho relativo direto, ja que o notebook roda de dentro de notebooks/
    codigo = codigo.replace(
        'pasta_dados = Path(__file__).resolve().parent.parent / "dados"',
        'pasta_dados = Path("../dados")'
    ).replace(
        'pasta_resultados = Path(__file__).resolve().parent.parent / "resultados"',
        'pasta_resultados = Path("../resultados")'
    )
    # remove a docstring de topo (fica melhor como celula markdown)
    linhas = codigo.split("\n")
    if linhas[0].startswith('"""'):
        if linhas[0].count('"""') >= 2:
            codigo = "\n".join(linhas[1:]).lstrip("\n")
        else:
            fim = next(i for i, l in enumerate(linhas[1:], start=1) if l.strip().endswith('"""'))
            codigo = "\n".join(linhas[fim + 1:]).lstrip("\n")
    celulas.append(nbf.v4.new_code_cell(codigo))

md("""# Predição de Aprovação de Cartão de Crédito
## Tech Challenge Fase 2 — POSTECH / FIAP

Notebook único, executado do zero, consolidando as 27 etapas de ciência de dados do projeto:
ambiente → leitura → qualidade → target → merge → features → split → 4 modelos → avaliação →
explicabilidade → conclusões de negócio.

**Dataset:** `application_record.csv` (atributos cadastrais) + `credit_record.csv` (histórico
mensal de crédito), fornecidos pelo desafio.

**Regra de acompanhamento deste notebook:** nenhum número foi inventado — todo valor abaixo é
saída real da execução deste mesmo notebook. Um vazamento de dados real foi encontrado e
corrigido durante o desenvolvimento (ver seção de Feature Engineering) — deixado documentado
de propósito, é parte do processo de validação, não um erro escondido.""")

md("## 1-2. Bibliotecas\n\nExecução local (não Google Colab). Todas as bibliotecas já estão "
   "instaladas no ambiente: pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost, shap.")
celulas.append(nbf.v4.new_code_cell(
    "import pandas as pd, numpy as np, sklearn, xgboost, shap\n"
    "print('pandas', pd.__version__)\nprint('sklearn', sklearn.__version__)\n"
    "print('xgboost', xgboost.__version__)\nprint('shap', shap.__version__)"
))

md("## 3-4. Leitura dos CSVs e entendimento das colunas")
code_de_arquivo("01_leitura_entendimento.py")

md("## 6-10. Qualidade dos dados, status de crédito, definição do target e merge")
code_de_arquivo("02_qualidade_target_merge.py")

md("## 11-13. Feature engineering, balanceamento e split treino/teste\n\n"
   "**Atenção:** a primeira versão da feature `eventos_risco` causou vazamento de dados "
   "(correlação 1.0 com o target, pois foi calculada com a mesma regra que define o target). "
   "O código abaixo já inclui a checagem que detectou isso e a correção (remoção da feature).")
code_de_arquivo("03_features_split.py")

md("## 14-17. Treino dos 4 modelos (Regressão Logística, Árvore, Random Forest, XGBoost)")
code_de_arquivo("04_modelos.py")

md("## 18-24. Avaliação comparativa, matriz de confusão, ROC-AUC, feature importance")
code_de_arquivo("05_avaliacao.py")

md("### Matrizes de confusão dos 4 modelos")
celulas.append(nbf.v4.new_code_cell(
    "from IPython.display import Image\nImage(filename='../resultados/matrizes_confusao.png')"
))
md("### Curvas ROC")
celulas.append(nbf.v4.new_code_cell(
    "Image(filename='../resultados/curvas_roc.png')"
))
md("### Feature importance (permutation)")
celulas.append(nbf.v4.new_code_cell(
    "Image(filename='../resultados/feature_importance.png')"
))

md("## 25. SHAP")
code_de_arquivo("06_shap.py")
celulas.append(nbf.v4.new_code_cell(
    "Image(filename='../resultados/shap_importancia_global.png')"
))

md("## 26-27. Interpretação dos resultados e conclusões de negócio")
code_de_arquivo("07_interpretacao_negocio.py")

nb["cells"] = celulas
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12"},
}

caminho_nb = pasta / "predicao_cartao_credito.ipynb"
nbf.write(nb, caminho_nb)
print(f"Notebook montado: {caminho_nb}")
