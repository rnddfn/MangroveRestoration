from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image, ImageDraw


ROOT = Path(r"C:\Users\lenov\Mangrove\skripsi")
OUT = Path(r"C:\Users\lenov\Mangrove\work\pdf_samples")
OUT.mkdir(parents=True, exist_ok=True)

samples = {
    "henny": (ROOT / "Naskah Skripsi.pdf", [0, 9, 16]),
    "pasha": (ROOT / "Skripsi_Muhammad Pasha Nabeel.pdf", [0, 7, 16]),
    "ryan": (ROOT / "Skripsi_Muhmmad Ryan Rizky Rahmadi.pdf", [0, 8, 16]),
}

thumbs = []
for label, (path, page_indexes) in samples.items():
    pdf = pdfium.PdfDocument(path)
    for page_index in page_indexes:
        bitmap = pdf[page_index].render(scale=1.25)
        image = bitmap.to_pil().convert("RGB")
        image.thumbnail((430, 610))
        canvas = Image.new("RGB", (450, 650), "white")
        canvas.paste(image, ((450 - image.width) // 2, 30))
        ImageDraw.Draw(canvas).text((10, 8), f"{label} - PDF page {page_index + 1}", fill="black")
        thumbs.append(canvas)

sheet = Image.new("RGB", (1350, 1950), "#dddddd")
for i, image in enumerate(thumbs):
    sheet.paste(image, ((i % 3) * 450, (i // 3) * 650))
sheet.save(OUT / "comparison_contact_sheet.png")
