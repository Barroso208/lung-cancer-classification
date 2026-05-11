"""Convert .pptx → PDF (via PowerPoint COM) → JPGs (via pdf2image / pymupdf)."""
import sys, os
from pathlib import Path
import win32com.client
import pythoncom

src = Path(sys.argv[1]).resolve()
pdf = src.with_suffix(".pdf")

pythoncom.CoInitialize()
ppt = win32com.client.Dispatch("PowerPoint.Application")
ppt.Visible = 1
deck = ppt.Presentations.Open(str(src), WithWindow=False)
deck.SaveAs(str(pdf), 32)  # ppSaveAsPDF = 32
deck.Close()
ppt.Quit()
print(f"PDF: {pdf}")

# Convert PDF → JPGs using pymupdf
try:
    import fitz
except ImportError:
    os.system(f'"{sys.executable}" -m pip install --quiet pymupdf')
    import fitz

doc = fitz.open(str(pdf))
out_dir = src.parent / "_qa_slides"
out_dir.mkdir(exist_ok=True)
for i, page in enumerate(doc, 1):
    pm = page.get_pixmap(dpi=110)
    out = out_dir / f"slide-{i:02d}.jpg"
    pm.save(str(out))
    print(out)
doc.close()
