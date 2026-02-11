import sys
from docx import Document

in_path = sys.argv[1]
out_path = sys.argv[2]

doc = Document(in_path)
texts = []
for p in doc.paragraphs:
    texts.append(p.text)

with open(out_path, "w", encoding="utf-8") as f:
    f.write("\n\n".join(t for t in texts if t.strip()))

print(f"Extracted {len(texts)} paragraphs to {out_path}")
