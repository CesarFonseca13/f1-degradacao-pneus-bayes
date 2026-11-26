"""Gera o relatório em PDF a partir do notebook executado: notebook -> HTML (MathJax) -> PDF (Edge headless)."""
import pathlib, subprocess, sys
import nbformat
from nbconvert import HTMLExporter

RAIZ = pathlib.Path(__file__).resolve().parent.parent
NB = RAIZ / "trabalho_final_f1_bayes.ipynb"
HTML = RAIZ / "relatorio" / "trabalho_final_f1_bayes.html"
PDF = RAIZ / "trabalho_final_f1_bayes.pdf"
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

CSS = """
<style>
@page { size: A4; margin: 14mm 13mm 16mm 13mm; }
html, body { background: #ffffff !important; }
body { font-family: "Segoe UI", Arial, sans-serif; font-size: 10.5pt; line-height: 1.45; color: #111; }
.jp-InputPrompt, .jp-OutputPrompt, .prompt { display: none !important; }
.jp-Cell { padding: 2px 0 !important; break-inside: auto; }
.jp-InputArea-editor, .jp-CodeMirrorEditor, div.highlight { background: #f6f6f4 !important; border: none !important; border-left: 3px solid #2a78d6 !important; border-radius: 3px; }
pre, code, .highlight pre { font-family: Consolas, "Cascadia Mono", monospace !important; font-size: 7.9pt !important; line-height: 1.32 !important; white-space: pre-wrap !important; word-break: break-word; }
.jp-OutputArea-output pre { font-size: 7.9pt !important; background: transparent; }
.jp-RenderedImage img, .jp-OutputArea-output img { max-width: 100% !important; height: auto !important; break-inside: avoid; }
.jp-RenderedHTMLCommon table, .dataframe { font-size: 7.8pt !important; border-collapse: collapse; margin: 6px 0; }
.jp-RenderedHTMLCommon th, .jp-RenderedHTMLCommon td { padding: 2px 6px !important; }
.jp-RenderedHTMLCommon h1 { font-size: 21pt; margin-top: 0; color: #0b0b0b; }
.jp-RenderedHTMLCommon h2 { font-size: 15pt; border-bottom: 2px solid #e34948; padding-bottom: 3px; margin-top: 22px; break-after: avoid; }
.jp-RenderedHTMLCommon h3 { font-size: 12pt; margin-top: 16px; break-after: avoid; }
.jp-RenderedHTMLCommon blockquote { border-left: 3px solid #eda100; background: #fffaf0; margin: 8px 0; padding: 6px 12px; color: #222; }
.jp-RenderedHTMLCommon hr { border: none; border-top: 1px solid #ddd; margin: 18px 0; }
.jp-MarkdownOutput { padding-left: 0 !important; }
</style>
"""


def main():
    nb = nbformat.read(NB, as_version=4)
    exp = HTMLExporter(template_name="lab")
    exp.exclude_input_prompt = True
    exp.exclude_output_prompt = True
    corpo, _ = exp.from_notebook_node(nb)
    corpo = corpo.replace("</head>", CSS + "</head>", 1)
    HTML.parent.mkdir(exist_ok=True)
    HTML.write_text(corpo, encoding="utf-8")
    cmd = [EDGE, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--run-all-compositor-stages-before-draw",
           "--virtual-time-budget=60000", f"--print-to-pdf={PDF}", HTML.as_uri()]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    print(r.stdout[-500:], r.stderr[-800:])
    print("PDF:", PDF, PDF.exists() and f"{PDF.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    sys.exit(main())
