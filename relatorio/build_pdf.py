"""Gera relatorio/relatorio.pdf a partir de relatorio.md.

Uso: uv run python relatorio/build_pdf.py
Requer LibreOffice (comando `libreoffice` ou `soffice`) e as figuras de models/figures
(geradas por `python -m guardia.ml.train`).
"""
import shutil
import subprocess
from pathlib import Path

import markdown
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

CSS = """
body { font-family: 'DejaVu Sans', Arial, sans-serif; font-size: 10.5pt; line-height: 1.45; color: #222; }
h1 { color: #7b1fa2; font-size: 20pt; } h2 { color: #7b1fa2; border-bottom: 1px solid #ccc; margin-top: 18pt; }
table { border-collapse: collapse; width: 100%; margin: 8pt 0; font-size: 9pt; }
th, td { border: 1px solid #bbb; padding: 3pt 5pt; } th { background: #f3e5f5; }
code { font-family: 'DejaVu Sans Mono', monospace; font-size: 9pt; background: #f4f4f4; }
"""


def main() -> None:
    fig_dir = HERE / "figuras"
    fig_dir.mkdir(exist_ok=True)
    for name in ["confusion_random_forest.png", "shap_global.png"]:
        # LibreOffice ignora largura via CSS: reduzimos o PNG para caber na página (~16 cm a 96 dpi)
        img = Image.open(ROOT / "models" / "figures" / name)
        img.resize((580, round(img.height * 580 / img.width)), Image.LANCZOS).save(fig_dir / name)
    body = markdown.markdown((HERE / "relatorio.md").read_text(), extensions=["tables"])
    html = HERE / "relatorio.html"
    html.write_text(f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{body}</body></html>")
    office = shutil.which("libreoffice") or shutil.which("soffice")
    if not office:
        raise SystemExit("LibreOffice não encontrado")
    subprocess.run([office, "--headless", "--convert-to", "pdf:writer_web_pdf_Export", "--outdir", str(HERE), str(html)], check=True)
    print("gerado:", HERE / "relatorio.pdf")


if __name__ == "__main__":
    main()
