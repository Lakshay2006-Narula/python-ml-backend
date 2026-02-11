# Drive Test Report Generator 

Automated telecom drive test report generation with AI-powered text generation and professional PDF output.

##  Overview

This project implements a comprehensive end-to-end pipeline for generating professional telecom drive test reports. It processes raw network measurement data from MySQL databases, performs spatial filtering using polygon boundaries, generates interactive KPI visualization maps, conducts statistical analysis, and produces publication-ready PDF reports with AI-generated narratives.

The system is designed for telecom network operators and engineers who need to analyze drive test data for network optimization, coverage assessment, and performance monitoring.

##  Architecture

### Core Pipeline Flow

```
Raw Data → Filtering → Maps → Analysis → Metadata → LLM Text → PDF Report
    ↓         ↓         ↓        ↓          ↓          ↓         ↓
MySQL DB → Polygon → Folium → Stats → Geocoding → Groq → ReportLab
```

### Module Architecture

- **`src/main.py`**: Pipeline orchestrator coordinating all processing steps
- **`src/load_data_db.py`**: Database connectivity and spatial data filtering
- **`src/map_generator.py`**: Interactive map generation using Folium
- **`src/kpi_analysis.py`**: Statistical KPI analysis and visualization
- **`src/metadata_generator.py`**: Structured metadata building with geocoding
- **`src/llm_integration.py`**: AI-powered report text generation
- **`src/pdf_generator.py`**: Professional PDF report assembly
- **`src/threshold_resolver.py`**: Dynamic KPI threshold resolution
- **`src/playwright_utils.py`**: HTML-to-PNG conversion utilities

##  Project Structure

```
Pdf_Report/
├── src/
│   ├── main.py                 # Main pipeline orchestrator
│   ├── load_data_db.py         # MySQL data loading with polygon filtering
│   ├── threshold_resolver.py   # User-specific KPI threshold management
│   ├── map_generator.py        # Folium-based interactive map generation
│   ├── kpi_analysis.py         # Statistical analysis and CDF plotting
│   ├── metadata_generator.py   # Metadata building with reverse geocoding
│   ├── cdf_kpi.py              # Cumulative distribution function plots
│   ├── llm_integration.py      # Groq API integration for text generation
│   ├── pdf_generator.py        # ReportLab PDF generation with TOC
│   ├── playwright_utils.py     # Browser automation for map screenshots
│   ├── db.py                   # Database connection utilities
│   └── kpi_config.py           # KPI definitions and color schemes
├── data/
│   ├── images/                 # Generated visualizations
│   │   ├── kpi_maps/           # KPI coverage maps
│   │   └── kpi_analysis/       # Statistical plots and tables
│   ├── processed/              # Output files
│   │   ├── filtered_data.csv   # Polygon-filtered measurement data
│   │   ├── report_metadata.json # Structured analysis metadata
│   │   └── report_text.json    # AI-generated report narratives
│   └── tmp/                    # Temporary HTML files
├── test_cases/                 # Comprehensive test suite
├── requirements.txt            # Python dependencies
├── examples_usage.py           # Usage examples and API demos
└── .env                        # Environment configuration
```

##  Quick Start

### Prerequisites

- Python 3.10+
- MySQL database with drive test data
- Groq API key for AI text generation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file in the project root:

```env
# Database Configuration
DB_HOST=your_mysql_host
DB_PORT=3306
DB_USER=your_username
DB_PASSWORD=your_password
DB_NAME=your_database

# LLM API Configuration
GROQ_API_KEY=your_groq_api_key_here
```

### 3. Run Complete Pipeline

```bash
python -m src.main
```

This executes the full pipeline:
1.  Load and filter drive test data from database
2.  Generate interactive KPI visualization maps
3.  Perform statistical KPI analysis
4.  Build structured metadata with location intelligence
5.  Generate professional report narratives using AI
6.  Create publication-ready PDF report with dynamic TOC

##  Data Processing Pipeline

### 1. Data Loading (`load_data_db.py`)

**Purpose**: Load raw drive test measurements from MySQL and apply spatial filtering.

**Key Features**:
- Multi-session data aggregation
- Polygon-based geographical filtering using Shapely
- Automatic data validation and cleaning
- Session metadata extraction

**Database Schema**:
- `tbl_project`: Project definitions with session references
- `tbl_network_logs`: Raw measurement data (RSRP, RSRQ, SINR, throughput, etc.)
- `tbl_regions`: Geographical boundaries as WKT polygons

### 2. Map Generation (`map_generator.py`)

**Purpose**: Create interactive geographical visualizations of network performance.

**Technologies**: Folium, Playwright, Shapely

**Map Types Generated**:
- **Base Route Map**: Drive path with polygon boundaries
- **KPI Range Maps**: Color-coded performance visualization (RSRP, RSRQ, SINR, DL/UL throughput, MOS)
- **Poor Region Maps**: Highlighted areas with poor coverage
- **Handover Maps**: Cell transition visualization
- **PCI Distribution Maps**: Physical Cell ID analysis

**Features**:
- Dynamic color schemes based on KPI thresholds
- Interactive legends with sample counts
- Polygon boundary overlays
- Responsive design for PDF embedding

### 3. KPI Analysis (`kpi_analysis.py`)

**Purpose**: Perform statistical analysis of network performance metrics.

**Analysis Types**:
- **Descriptive Statistics**: Mean, median, percentiles, min/max
- **Distribution Analysis**: CDF plots for each KPI
- **Threshold Analysis**: Poor performance identification
- **Band Distribution**: Frequency band usage statistics

**KPI Metrics Supported**:
- **RSRP** (Reference Signal Received Power): Signal strength
- **RSRQ** (Reference Signal Received Quality): Signal quality
- **SINR** (Signal-to-Interference-plus-Noise Ratio): Interference levels
- **DL/UL Throughput**: Data transfer speeds
- **MOS** (Mean Opinion Score): Voice quality
- **PCI** (Physical Cell ID): Cell identity distribution

### 4. Metadata Generation (`metadata_generator.py`)

**Purpose**: Build structured metadata for report generation.

**Components**:
- **Location Intelligence**: Reverse geocoding using Nominatim API
- **Area Summary**: Spatial clustering and location naming
- **Drive Statistics**: Temporal and spatial coverage analysis
- **KPI Summaries**: Performance metric aggregation
- **Band Analysis**: Frequency usage patterns

**Geocoding Features**:
- City/country identification from coordinates
- Spatial grid analysis for hotspot detection
- Road/street level location naming
- Haversine distance calculations for clustering

### 5. LLM Integration (`llm_integration.py`)

**Purpose**: Generate professional report narratives using AI.

**AI Provider**: Groq API with Llama models

**Text Generation**:
- **Introduction**: Executive summary with location context
- **Area Summary**: Structured geographical coverage description
- **Drive Summary**: Temporal and spatial test coverage
- **KPI Analysis**: Performance interpretation with technical insights
- **Map View Descriptions**: KPI-specific analysis paragraphs

**Features**:
- Structured JSON prompts for consistent output
- Markdown parsing and cleaning
- Fallback synthesis for missing sections
- Location token replacement
- Source tracking and diagnostics

### 6. PDF Generation (`pdf_generator.py`)

**Purpose**: Assemble professional PDF reports with dynamic table of contents.

**PDF Features**:
- **Dynamic TOC**: Automatic page number generation
- **Professional Styling**: Custom fonts, colors, and layouts
- **Image Integration**: Automatic map and chart embedding
- **Structured Sections**: Hierarchical content organization
- **Page Numbering**: Footer page counters

**Report Structure**:
1. Cover Page (Title, location, date)
2. Table of Contents
3. Introduction
4. Area Summary + Route Map
5. Drive Summary + Statistics
6. KPI Summary + Analysis Tables
7. Band Distribution
8. Coverage Analysis (RSRP, RSRQ, SINR maps)
9. Throughput Analysis (DL/UL maps)
10. PCI Summary + Distribution Charts
11. App Analytics (if available)
12. Indoor/Outdoor Analysis
13. Performance Summary

##  Advanced Usage

### Running Individual Modules

#### Generate Maps Only

```python
from src.map_generator import generate_kpi_map
import pandas as pd

df = pd.read_csv("data/processed/filtered_data.csv")
generate_kpi_map(
    df=df,
    kpi_column="rsrp",
    ranges=kpi_ranges,
    output_html="maps/rsrp_map.html",
    polygon_wkt=boundary
)
```

#### Generate Report Text Only

```python
from src.llm_integration import generate_report_text

metadata = load_metadata("data/processed/report_metadata.json")
report_text = generate_report_text(
    metadata=metadata,
    output_path="data/processed/report_text.json",
    model="llama-3.1-8b-instant",
    temperature=0.0,
    verbose=True
)
```

#### Custom PDF Generation

```python
from src.pdf_generator import PDFReportGenerator

generator = PDFReportGenerator(
    output_path="custom_report.pdf",
    images_dir="data/images"
)

# Load data
with open("data/processed/report_metadata.json") as f:
    metadata = json.load(f)
with open("data/processed/report_text.json") as f:
    report_text = json.load(f)

generator.generate_report(report_text, metadata, verbose=True)
```

### Configuration Options

#### KPI Thresholds (`src/threshold_resolver.py`)

Dynamic threshold resolution based on user preferences:
- Database-stored custom ranges
- Automatic range calculation from data distribution
- User-specific configurations

#### Map Styling (`src/map_generator.py`)

Customizable map appearance:
- Color schemes per KPI type
- Legend positioning and styling
- Zoom levels and center coordinates
- Boundary polygon styling

##  Output Files

### Generated Artifacts

```
data/
├── processed/
│   ├── filtered_data.csv         # Polygon-filtered measurements
│   ├── report_metadata.json      # Structured analysis data
│   ├── report_text.json          # AI-generated narratives
│   └── drive_test_report_toc.pdf # Final PDF report
└── images/
    ├── kpi_maps/
    │   ├── rsrp_map.png          # Signal strength visualization
    │   ├── rsrq_map.png          # Signal quality visualization
    │   ├── sinr_map.png          # Interference analysis
    │   ├── dl_map.png            # Download speed map
    │   ├── ul_map.png            # Upload speed map
    │   ├── mos_map.png           # Voice quality map
    │   ├── base_route_map.png    # Drive route with boundaries
    │   ├── handover_map.png      # Cell transition analysis
    │   └── pci_map.png           # Cell identity distribution
    └── kpi_analysis/
        ├── kpi_summary.png       # KPI statistics table
        ├── band_table.png        # Frequency usage table
        ├── band_pie.png          # Band distribution chart
        ├── cdf_rsrp.png          # RSRP distribution plot
        ├── cdf_rsrq.png          # RSRQ distribution plot
        ├── cdf_sinr.png          # SINR distribution plot
        ├── cdf_dl_tpt.png        # DL throughput distribution
        ├── cdf_ul_tpt.png        # UL throughput distribution
        └── cdf_mos.png           # MOS distribution plot
```

### Report Metadata Structure

```json
{
  "location": {
    "city": "Mumbai",
    "country": "India"
  },
  "area_summary": {
    "Overview": "Drive route covers key operational areas...",
    "Hotspots & Marked Locations": "Location A, Location B",
    "Major Areas Covered": "Areas include Road X, Area Y..."
  },
  "drive_summary": {
    "distance_covered": 45.2,
    "total_samples": 125000,
    "total_sessions": 3,
    "start_date": "2024-01-15",
    "end_date": "2024-01-17"
  },
  "kpi_summary": {
    "RSRP": {
      "average": -85.3,
      "min": -120.0,
      "max": -45.0,
      "poor_count": 1250,
      "poor_percentage": 1.2
    }
  },
  "band_summary": [
    {"band": "B8", "sample_percentage": 45.2},
    {"band": "B40", "sample_percentage": 34.8}
  ]
}
```

##  Technical Details

### Dependencies

**Core Libraries**:
- `pandas`: Data manipulation and analysis
- `folium`: Interactive map generation
- `matplotlib`: Statistical plotting
- `shapely`: Geometric operations and spatial filtering
- `playwright`: Browser automation for map rendering
- `reportlab`: PDF generation and layout
- `groq`: AI text generation API
- `geopy`: Geocoding and location services
- `mysql-connector-python`: Database connectivity

**Development Tools**:
- `python-dotenv`: Environment configuration
- `openpyxl`: Excel file processing
- `numpy`: Numerical computations

### Performance Considerations

- **Memory Usage**: Large datasets processed in chunks
- **API Limits**: Geocoding requests rate-limited with sleep intervals
- **File I/O**: Efficient streaming for large CSV exports
- **Image Optimization**: PNG compression for PDF embedding

### Error Handling

- Database connection failures with retry logic
- Missing data graceful degradation
- API timeout handling with fallbacks
- File system permission checks

##  Report Customization

### Styling Options

The PDF generator supports extensive customization:

```python
# Custom styles in pdf_generator.py
styles.add(ParagraphStyle(
    name="CustomSection",
    parent=styles["Heading2"],
    fontSize=18,
    textColor=colors.HexColor("#1f4788")
))
```

### Content Sections

Reports can be customized by modifying the `generate_report` method in `PDFReportGenerator`:

- Add/remove sections
- Reorder content
- Include additional images
- Modify table formats

### LLM Prompts

Text generation can be customized by editing prompts in `llm_integration.py`:

- Modify tone and style
- Add domain-specific terminology
- Include additional KPIs
- Change output structure

##  Troubleshooting

### Common Issues

#### Database Connection Issues

```python
# Test connection
from src.db import get_connection
cn = get_connection()
print("Connection successful!")
cn.close()
```

#### LLM API Issues

```python
# Check API key
import os
print("GROQ_API_KEY:", os.getenv("GROQ_API_KEY"))

# Test with minimal prompt
from src.llm_integration import generate_report_text
report_text = generate_report_text(metadata, verbose=True)
```

#### PDF Generation Issues

```python
# Verify image paths
import os
images_dir = "data/images/kpi_maps"
print("Available images:", os.listdir(images_dir))

# Check image file integrity
from PIL import Image
img = Image.open("data/images/kpi_maps/rsrp_map.png")
print("Image size:", img.size)
```

#### Map Generation Issues

```python
# Test Folium map creation
import folium
m = folium.Map(location=[19.0760, 72.8777], zoom_start=12)
print("Folium working")

# Test Playwright conversion
from src.playwright_utils import html_to_png
html_to_png("test.html", "test.png")
```

### Debug Mode

Enable verbose logging:

```python
# In main.py
main(project_id=149, user_id=13, verbose=True)

# Individual modules
generate_report_text(metadata, verbose=True)
generate_pdf_report(verbose=True)
```

##  API Reference

### Main Pipeline

```python
def main(project_id: int, user_id: int | None = None):
    """
    Execute complete report generation pipeline.
    
    Args:
        project_id: Database project identifier
        user_id: Optional user ID for threshold resolution
    """
```

### Data Loading

```python
def load_project_data(project_id: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Load and filter project data from database.
    
    Returns:
        raw_df: Unfiltered measurement data
        filtered_df: Polygon-filtered data
        project_meta: Project metadata dictionary
    """
```

### Map Generation

```python
def generate_kpi_map(
    df: pd.DataFrame,
    kpi_column: str,
    ranges: list,
    output_html: str,
    polygon_wkt: str = None
) -> None:
    """Generate interactive KPI visualization map."""
```

### LLM Integration

```python
def generate_report_text(
    metadata: dict,
    output_path: str,
    model: str = "llama-3.1-8b-instant",
    temperature: float = 0.0,
    max_tokens: int = 2500,
    verbose: bool = False
) -> dict:
    """Generate AI-powered report narratives."""
```

### PDF Generation

```python
def generate_pdf_report(
    metadata_path: str,
    report_text_path: str,
    output_path: str,
    images_dir: str,
    verbose: bool = False
) -> str:
    """Generate professional PDF report with TOC."""
```



### Development Setup

1. Fork the repository
2. Create feature branch: `git checkout -b feature/new-feature`
3. Install development dependencies
4. Run tests: `python -m pytest test_cases/`
5. Submit pull request

### Code Standards

- Type hints for all function parameters
- Docstrings following Google style
- Unit tests for new functionality
- Error handling with appropriate exceptions

### Testing

Run the comprehensive test suite:

```bash
# LLM integration tests
python test_cases/test_llm_json_validation.py

# PDF generation tests
python test_cases/test_full_pdf_with_toc.py

# Map generation tests
python test_cases/test_map_data_validation.py
```







### Run Individual Modules

#### Generate LLM Report Text Only

```python
from src.llm_integration import generate_report_text, load_metadata

metadata = load_metadata("data/processed/report_metadata.json")
report_text = generate_report_text(
    metadata=metadata,
    output_path="data/processed/report_text.json",
    model="llama-3.1-8b-instant",
    temperature=0.0,
    verbose=True
)
```

#### Generate PDF Report Only

```python
from src.pdf_generator import generate_pdf_report

pdf_path = generate_pdf_report(
    metadata_path="data/processed/report_metadata.json",
    report_text_path="data/processed/report_text.json",
    output_path="data/processed/drive_test_report.pdf",
    images_dir="data/images",
    verbose=True
)
```

### Customize Report Generation

```python
from src.pdf_generator import PDFReportGenerator

# Create custom generator
generator = PDFReportGenerator(
    output_path="custom_report.pdf",
    images_dir="data/images"
)

# Load data
import json
with open("data/processed/report_metadata.json") as f:
    metadata = json.load(f)
with open("data/processed/report_text.json") as f:
    report_text = json.load(f)

# Generate with custom settings
generator.generate_report(
    report_text=report_text,
    metadata=metadata,
    verbose=True
)
```

##  Module Documentation

### LLM Integration (`src/llm_integration.py`)

**Key Functions:**

- `generate_report_text(metadata, output_path, model, temperature, max_tokens, verbose)`
  - Generates structured report narratives using LLM
  - Returns: Dict with report sections

- `load_metadata(metadata_path)`
  - Loads metadata JSON file
  - Returns: Dict containing metadata

- `extract_json_from_text(text)`
  - Parses JSON from LLM response
  - Handles markdown code blocks

- `validate_report_text(report_text)`
  - Validates report structure
  - Returns: (is_valid, bad_key)

**Configuration:**
- Model: `llama-3.1-8b-instant` (default)
- Temperature: `0.0` (deterministic)
- Max Tokens: `1500`

### PDF Generator (`src/pdf_generator.py`)

**Key Classes:**

`PDFReportGenerator`
- Main class for PDF generation
- Handles layout, styling, and content

**Key Methods:**

- `add_cover_page(metadata)` - Generate cover page
- `add_section(title, text, add_pagebreak)` - Add text section
- `add_image(filename, width, height, caption, subdir)` - Add image
- `add_kpi_summary_table(metadata)` - Add KPI table
- `add_band_summary_table(metadata)` - Add band table
- `generate_report(report_text, metadata, verbose)` - Build complete report

**Styling:**
- Page Size: A4
- Margins: 40pt
- Custom fonts and colors
- Professional table formatting

##  Report Structure

The generated PDF includes:

1. **Cover Page**
   - Report title
   - Location (city, country)
   - Generation date

2. **Introduction**
   - Purpose and scope
   - Test methodology

3. **Area Summary**
   - Geographic coverage
   - Drive test route map

4. **Drive Summary**
   - Test statistics (if available)

5. **KPI Summary**
   - Overall metrics narrative
   - KPI statistics table

6. **Band Distribution**
   - Band usage analysis
   - Distribution table

7. **Coverage Analysis**
   - RSRP analysis with map
   - RSRQ analysis with map
   - SINR analysis with map

8. **Throughput Analysis**
   - DL throughput with map
   - UL throughput with map
   - MOS analysis with map

9. **PCI Summary**
   - Network diversity analysis

##  Troubleshooting

### LLM API Issues

```python
# Check API key
import os
print(os.getenv("GROQ_API_KEY"))

# Test with verbose mode
report_text = generate_report_text(metadata, verbose=True)
```

### PDF Generation Issues

```python
# Check image paths
import os
images_dir = "data/images/kpi_maps"
print(os.listdir(images_dir))

# Generate with verbose mode
pdf_path = generate_pdf_report(verbose=True)
```

### Database Connection Issues

```python
# Test connection
from src.db import get_connection
cn = get_connection()
print("Connection successful!")
cn.close()
```

##  Dependencies

Core libraries:
- `pandas` - Data processing
- `folium` - Map generation
- `matplotlib` - Plotting
- `shapely` - Geometric operations
- `playwright` - HTML to PNG conversion
- `groq` - LLM API
- `reportlab` - PDF generation
- `mysql-connector-python` - Database connection



##  License

Proprietary - Internal Use Only



---
**Developed By:** Vineeth Raja Banala

