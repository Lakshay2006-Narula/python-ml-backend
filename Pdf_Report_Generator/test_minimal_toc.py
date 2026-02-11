"""Test minimal TOC implementation"""
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, PageBreak
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.lib.styles import getSampleStyleSheet

# Create PDF
pdf_path = "data/processed/test_minimal_toc.pdf"
doc = SimpleDocTemplate(pdf_path, pagesize=letter)
story = []
styles = getSampleStyleSheet()

# Add TOC
toc = TableOfContents()
story.append(toc)
story.append(PageBreak())

# Add section with bookmark
p = Paragraph('1. First Section<a name="sec1"/>', styles['Heading1'])
story.append(p)

# Register TOC entry - try different signatures
print("Testing addEntry signatures...")
print(f"addEntry method: {TableOfContents.addEntry}")
print(f"Signature help:")
import inspect
print(inspect.signature(TableOfContents.addEntry))

# Try the correct call - pageNum is required even if placeholder
toc.addEntry(0, '1. First Section', 0, key='sec1')

# Build
doc.multiBuild(story)
print(f"✓ PDF generated: {pdf_path}")
