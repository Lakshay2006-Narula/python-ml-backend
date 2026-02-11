"""
Test Dynamic TOC - Proper ReportLab Approach
Using bookmarks and automatic TOC population during multiBuild
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

def test_dynamic_toc():
    """Test TRULY dynamic TOC using bookmarks"""
    
    print("\n" + "=" * 70)
    print("TEST: DYNAMIC TOC WITH BOOKMARKS")
    print("=" * 70)
    
    pdf_path = "data/processed/test_dynamic_toc.pdf"
    
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        topMargin=0.75*inch,
        bottomMargin=0.75*inch
    )
    
    story = []
    styles = getSampleStyleSheet()
    
    # TOC Styles
    styles.add(ParagraphStyle(
        name="TOCHeading",
        parent=styles["Heading1"],
        fontSize=18,
        textColor=colors.HexColor("#1f4788"),
        spaceAfter=12
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
        ParagraphStyle(fontSize=11, name='TOCLevel1', leftIndent=0, leading=14),
        ParagraphStyle(fontSize=10, name='TOCLevel2', leftIndent=20, leading=12),
    ]
    story.append(toc)
    story.append(PageBreak())
    
    # Section 1 - bookmark in paragraph + register with TOC (pageNum=1 is placeholder)
    story.append(Paragraph('1. Introduction<bookmark name="intro" />', styles['Heading1']))
    toc.addEntry(0, '1. Introduction', 1, key='intro')
    story.append(Paragraph("Introduction content here.", styles['BodyText']))
    story.append(Spacer(1, 0.3*inch))
    
    # Section 2
    story.append(Paragraph('2. Area Summary<bookmark name="area" />', styles['Heading1']))
    toc.addEntry(0, '2. Area Summary', 1, key='area')
    story.append(Paragraph("Area summary content.", styles['BodyText']))
    story.append(Spacer(1, 0.3*inch))
    
    # Section 5 with subsections
    story.append(Paragraph('5. Map View<bookmark name="map" />', styles['Heading1']))
    toc.addEntry(0, '5. Map View', 1, key='map')
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph('a) Band<bookmark name="band" />', styles['Heading2']))
    toc.addEntry(1, 'a) Band', 1, key='band')
    story.append(Paragraph("Band content.", styles['BodyText']))
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph('b) RSRP<bookmark name="rsrp" />', styles['Heading2']))
    toc.addEntry(1, 'b) RSRP', 1, key='rsrp')
    story.append(Paragraph("RSRP content.", styles['BodyText']))
    story.append(PageBreak())
    
    story.append(Paragraph('6. PCI Summary<bookmark name="pci" />', styles['Heading1']))
    toc.addEntry(0, '6. PCI Summary', 1, key='pci')
    story.append(Paragraph("PCI content.", styles['BodyText']))
    
    # Build with multiBuild for TOC
    doc.multiBuild(story, canvasmaker=PageNumCanvas)
    
    if os.path.exists(pdf_path):
        size_kb = os.path.getsize(pdf_path) / 1024
        print(f"\n✓ PDF generated: {pdf_path}")
        print(f"  Size: {size_kb:.2f} KB")
        print("\n✓ TOC should be populated dynamically with page numbers")
        print("  Open PDF to verify TOC shows actual page numbers")
        return True
    else:
        print("❌ PDF was not created")
        return False

if __name__ == "__main__":
    success = test_dynamic_toc()
    sys.exit(0 if success else 1)
