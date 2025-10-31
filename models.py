# In models.py

from extensions import db
from datetime import datetime
from sqlalchemy.dialects.mysql import BIGINT # Import the specific type for 'bigint unsigned'

# This is your existing table for logs
class Prediction(db.Model):
    __tablename__ = 'predictions'
    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    output_dir = db.Column(db.String(255), nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    method = db.Column(db.String(50))
    min_samples = db.Column(db.Integer)

    def __repr__(self):
        return f'<Prediction {self.id}: {self.filename}>'

# === ADD THIS NEW CLASS ===
# This defines the 'site_noMl' table
class SiteNoMl(db.Model):
    __tablename__ = 'site_noMl'

    # These columns match the 'DESCRIBE' output you sent
    id = db.Column(BIGINT(unsigned=True), primary_key=True, autoincrement=True)
    network = db.Column(db.String(100), index=True)
    earfcn_or_narfcn = db.Column(db.Float)
    site_key_inferred = db.Column(db.Integer, index=True)
    pci_or_psi = db.Column(db.Float, index=True)
    samples = db.Column(db.Integer)
    lat_pred = db.Column(db.Float)
    lon_pred = db.Column(db.Float)
    azimuth_deg_5 = db.Column(db.Integer)
    azimuth_deg_5_soft = db.Column(db.Integer)
    azimuth_deg_label_soft = db.Column(db.String(100))
    azimuth_adjustment_deg = db.Column(db.Float)
    template_spacing_deg = db.Column(db.Float)
    beamwidth_deg_est = db.Column(db.Integer)
    median_sample_distance_m = db.Column(db.Float)
    cell_id_representative = db.Column(db.Integer)
    sector_count = db.Column(db.Float)
    azimuth_reliability = db.Column(db.Float)
    spacing_used = db.Column(db.String(100))

# === ADD THIS NEW CLASS ===
# This defines the 'site_Ml' table
class SiteMl(db.Model):
    __tablename__ = 'site_Ml'

    # The columns are identical
    id = db.Column(BIGINT(unsigned=True), primary_key=True, autoincrement=True)
    network = db.Column(db.String(100), index=True)
    earfcn_or_narfcn = db.Column(db.Float)
    site_key_inferred = db.Column(db.Integer, index=True)
    pci_or_psi = db.Column(db.Float, index=True)
    samples = db.Column(db.Integer)
    lat_pred = db.Column(db.Float)
    lon_pred = db.Column(db.Float)
    azimuth_deg_5 = db.Column(db.Integer)
    azimuth_deg_5_soft = db.Column(db.Integer)
    azimuth_deg_label_soft = db.Column(db.String(100))
    azimuth_adjustment_deg = db.Column(db.Float)
    template_spacing_deg = db.Column(db.Float)
    beamwidth_deg_est = db.Column(db.Integer)
    median_sample_distance_m = db.Column(db.Float)
    cell_id_representative = db.Column(db.Integer)
    sector_count = db.Column(db.Float)
    azimuth_reliability = db.Column(db.Float)
    spacing_used = db.Column(db.String(100))