from collections import Counter
from pathlib import Path
import re
import zipfile

from docx import Document
from docx.oxml.ns import qn


ROOT = Path(r"C:\Users\lenov\Mangrove\skripsi\draft")

for path in sorted(ROOT.glob("*.docx")):
    doc = Document(path)
    nonempty = [p for p in doc.paragraphs if p.text.strip()]
    text = "\n".join(p.text for p in nonempty)
    fonts = Counter()
    sizes = Counter()
    bold = 0
    numbered = 0
    for p in nonempty:
        ppr = p._p.pPr
        if ppr is not None and ppr.numPr is not None:
            numbered += 1
        for run in p.runs:
            if run.text.strip():
                name = run.font.name or run._r.rPr.rFonts.get(qn("w:eastAsia")) if run._r.rPr is not None and run._r.rPr.rFonts is not None else run.font.name
                fonts[name or "(style/default)"] += len(run.text)
                sizes[str(run.font.size.pt if run.font.size else "(style/default)")] += len(run.text)
                bold += bool(run.bold)
    print(f"\n=== {path.name} ===")
    print("words", len(re.findall(r"\b\w+\b", text)), "chars", len(text), "numbered_paragraphs", numbered)
    print("fonts_by_chars", fonts.most_common(10))
    print("sizes_by_chars", sizes.most_common(10), "bold_runs", bold)
    print("citations_square", len(re.findall(r"\[\d+(?:\s*[-,]\s*\d+)*\]", text)))
    print("author_year", len(re.findall(r"\([A-Z][^()]{1,60},\s*20\d{2}\)", text)))
    print("template_markers", sum(text.lower().count(x) for x in ["contoh", "misalnya", "format baku", "universitas xyz", "azkia informasia", "deteksi sel kanker"]))
    with zipfile.ZipFile(path) as archive:
        xml = archive.read("word/document.xml").decode("utf-8", errors="replace")
        names = set(archive.namelist())
        print(
            "ooxml",
            "fields", xml.count("w:instrText"),
            "zotero", xml.lower().count("zotero"),
            "mendeley", xml.lower().count("mendeley"),
            "insertions", xml.count("<w:ins"),
            "deletions", xml.count("<w:del"),
            "comments_part", "word/comments.xml" in names,
        )
    for i, section in enumerate(doc.sections):
        print(
            "section", i,
            "page_cm", round(section.page_width.cm, 2), round(section.page_height.cm, 2),
            "margins_cm", round(section.top_margin.cm, 2), round(section.right_margin.cm, 2),
            round(section.bottom_margin.cm, 2), round(section.left_margin.cm, 2),
            "header_footer_cm", round(section.header_distance.cm, 2), round(section.footer_distance.cm, 2),
            "start", section.start_type,
        )
