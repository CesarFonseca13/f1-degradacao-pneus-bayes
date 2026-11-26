"""Gera um notebook .ipynb a partir de fontes em markdown (jupytext).
Uso: python construir_notebook.py <pasta_ou_arquivo_fonte> <saida.ipynb>"""
import pathlib, sys, jupytext
fonte, saida = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
partes = sorted(fonte.glob("*.md")) if fonte.is_dir() else [fonte]
nb = jupytext.reads("\n".join(p.read_text(encoding="utf-8") for p in partes), fmt="md")
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.metadata["language_info"] = {"name": "python"}
nb.metadata["title"] = "Quanto o pneu cai por volta? Bayes na F1 2026"
jupytext.write(nb, saida, fmt="ipynb")
print(f"{saida.name}: {len(nb.cells)} células ({sum(c.cell_type == 'code' for c in nb.cells)} de código)")
