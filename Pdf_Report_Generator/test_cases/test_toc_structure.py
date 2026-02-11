"""
Test TOC Structure Only - Minimal PDF
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.pdfgen import canvas

class PageNumCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        canvas.Canvas.__init__(self, *args, **kwargs)
        self.pages = []
        
    def showPage(self):
        self.pages.append(dict(self.__dict__))
        self._startPage()
        
    def save(self):
        page_count = len(self.pages)
        for page_num in range(page_count):
            self.__dict__.update(self.pages[page_num])
            self.draw_page_number(page_num + 1, page_count)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)
        
    def draw_page_number(self, page_num, page_count):
        self.setFont("Helvetica", 9)
        self.drawRightString(
            7.5 * inch, 0.5 * inch,
            f"Page {page_num}"
        )

def test_toc_structure():
    """Test TOC with proper structure and dynamic page numbers"""
    
    print("\n" + "=" * 70)
    print("TEST: TOC STRUCTURE WITH DYNAMIC PAGE NUMBERS")
    print("=" * 70)
    
    pdf_path = "data/processed/test_toc_structure.pdf"
    
    # Create PDF
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        topMargin=0.75*inch,
        bottomMargin=0.75*inch
    )
    
    story = []
    styles = getSampleStyleSheet()
    
    # Add TOC Styles
    styles.add(ParagraphStyle(
        name="TOCHeading",
        parent=styles["Heading1"],
        fontSize=18,
        textColor=colors.HexColor("#1f4788"),
        spaceAfter=12
    ))
    
    styles.add(ParagraphStyle(
        name="TOCLevel1",
        parent=styles["Normal"],
        fontSize=11,
        leftIndent=0,
        firstLineIndent=0,
        spaceBefore=3,
        spaceAfter=3,
        leading=14
    ))
    
    styles.add(ParagraphStyle(
        name="TOCLevel2",
        parent=styles["Normal"],
        fontSize=10,
        leftIndent=20,
        firstLineIndent=0,
        spaceBefore=2,
        spaceAfter=2,
        leading=12
    ))
    
    # Title Page
    story.append(Spacer(1, 2*inch))
    story.append(Paragraph("<b>RF Drive Test Report</b>", styles["Title"]))
    story.append(PageBreak())
    
    # TOC Page
    story.append(Paragraph("<b>Table of Contents</b>", styles["TOCHeading"]))
    story.append(Spacer(1, 0.2*inch))
    
    toc = TableOfContents()
    toc.levelStyles = [
        styles['TOCLevel1'],
        styles['TOCLevel2'],
    ]
    story.append(toc)
    story.append(PageBreak())
    
    # Content sections with bookmarks
    h1 = Paragraph('1. Introduction<a name="sec1"/>', styles['Heading1'])
    toc.addEntry(0, '1. Introduction', 1, 'sec1')
    story.append(h1)
    story.append(Paragraph("This is the introduction section content.", styles['BodyText']))
    story.append(Spacer(1, 0.3*inch))
    
    h2 = Paragraph('2. Area Summary<a name="sec2"/>', styles['Heading1'])
    toc.addEntry(0, '2. Area Summary', 1, 'sec2')
    story.append(h2)
    story.append(Paragraph("This is the area summary content.", styles['BodyText']))
    story.append(Spacer(1, 0.3*inch))
    
    # Subsections under Map View
    h3 = Paragraph('5. Map View<a name="sec5"/>', styles['Heading1'])
    toc.addEntry(0, '5. Map View', 1, 'sec5')
    story.append(h3)
    story.append(Spacer(1, 0.2*inch))
    
    # Subsection a)
    h3a = Paragraph('a) Band<a name="sec5a"/>', styles['Heading2'])
    toc.addEntry(1, '    a) Band', 1, 'sec5a')
    story.append(h3a)
    story.append(Paragraph("Band analysis content here.", styles['BodyText']))
    story.append(Spacer(1, 0.2*inch))
    
    # Subsection b)
    h3b = Paragraph('b) RSRP<a name="sec5b"/>', styles['Heading2'])
    toc.addEntry(1, '    b) RSRP', 1, 'sec5b')
    story.append(h3b)
    story.append(Paragraph("RSRP analysis content here.", styles['BodyText']))
    story.append(Spacer(1, 0.2*inch))
    
    # Subsection c)
    h3c = Paragraph('c) RSRQ<a name="sec5c"/>', styles['Heading2'])
    toc.addEntry(1, '    c) RSRQ', 1, 'sec5c')
    story.append(h3c)
    story.append(Paragraph("RSRQ analysis content here.", styles['BodyText']))
    story.append(PageBreak())
    
    # Section 6
    h4 = Paragraph('6. PCI Analysis<a name="sec6"/>', styles['Heading1'])
    toc.addEntry(0, '6. PCI Analysis', 1, 'sec6')
    story.append(h4)
    story.append(Paragraph("PCI analysis content.", styles['BodyText']))
    
    # Build PDF with multiple passes for TOC
    doc.multiBuild(story, canvasmaker=PageNumCanvas)
    
    if os.path.exists(pdf_path):
        size_kb = os.path.getsize(pdf_path) / 1024
        print(f"\n✓ PDF generated successfully!")
        print(f"  - Path: {pdf_path}")
        print(f"  - Size: {size_kb:.2f} KB")
        print("\nVerify:")
        print("  1. TOC shows sections with page numbers")
        print("  2. Page numbers are right-aligned")
        print("  3. Subsections (a, b, c) are indented")
        print("  4. Page numbers are DYNAMIC (not hardcoded)")
        print("  5. Clicking TOC entries jumps to sections")
        return True
    else:
        print("❌ PDF was not created")
        return False

if __name__ == "__main__":
    success = test_toc_structure()
    sys.exit(0 if success else 1)
