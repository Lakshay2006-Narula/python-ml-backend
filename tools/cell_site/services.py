from flask import current_app
from werkzeug.utils import secure_filename
import os
import time
import pandas as pd

from . import cell_site_processing as site
from extensions import db

# === NEW CODE ===
# Import the models to get their column names
from models import SiteNoMl, SiteMl
# === END NEW CODE ===


class CellSiteService:
    
    ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}
    
    def allowed_file(self, filename):
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in self.ALLOWED_EXTENSIONS
    
    def process_file(self, file, params):
        """Process uploaded cell site file"""
        
        # ... (File saving and directory creation code is unchanged) ...
        filename = secure_filename(file.filename)
        filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        current_app.logger.info(f"File saved: {filepath}")
        timestamp = str(int(time.time()))
        outdir = os.path.join(
            current_app.config['OUTPUT_FOLDER'],
            f'cellsite_{timestamp}'
        )
        os.makedirs(outdir, exist_ok=True)
        current_app.logger.info(f"Output directory: {outdir}")
        site.setup_logger(outdir, tag=params['method'])
        
        try:
            if params['method'] == 'noml':
                results = site.run_noml(
                    input_path=filepath,
                    outdir=outdir,
                    min_samples=params.get('min_samples', 30),
                    bin_size=params.get('bin_size', 5),
                    soft_spacing=params.get('soft_spacing', False),
                    use_ta=params.get('use_ta', False),
                    make_map=params.get('make_map', False),
                    merge_sites=params.get('soft_spacing', False)
                )
            else:  # ML method
                results = site.run_ml(
                    train_path=params.get('train_path'),
                    model_path=params.get('model_path'),
                    input_path=filepath,
                    outdir=outdir,
                    min_samples=params.get('min_samples', 30),
                    bin_size=params.get('bin_size', 5),
                    soft_spacing=params.get('soft_spacing', False),
                    make_map=params.get('make_map', False)
                )

            # === MODIFIED: SAVE DATAFRAME TO DB ===
            # Put the try/except block back for safety
            try:
                df_to_save = results.pop('dataframe', None)
                
                if df_to_save is not None and not df_to_save.empty:
                    
                    if params['method'] == 'noml':
                        table_name = 'site_noMl'
                        # Get the list of VALID columns from the SiteNoMl model
                        db_columns = SiteNoMl.__table__.columns.keys()
                    else:
                        table_name = 'site_Ml'
                        # Get the list of VALID columns from the SiteMl model
                        db_columns = SiteMl.__table__.columns.keys()

                    # === NEW CODE ===
                    # Filter the DataFrame to ONLY include columns that
                    # exist in the database table.
                    df_filtered = df_to_save[df_to_save.columns.intersection(db_columns)]
                    # === END NEW CODE ===

                    current_app.logger.info(f"Saving {len(df_filtered)} rows to table: {table_name}")
                    
                    # Save the new FILTERED DataFrame
                    df_filtered.to_sql(
                        table_name,
                        con=db.engine,
                        if_exists='append',
                        index=False
                    )
                    
                    current_app.logger.info(f"Successfully saved data to {table_name}.")
                    
            except Exception as db_e:
                current_app.logger.error(f"Failed to save data to database: {db_e}")
            # === END MODIFIED BLOCK ===

            
            # Local storage - convert results to relative paths
            relative_results = {}
            for key, path in results.items():
                if path and isinstance(path, str) and os.path.exists(path):
                    relative_results[key] = os.path.basename(path)
            
            return {
                'success': True,
                'results': relative_results,
                'output_dir': os.path.basename(outdir),
                'message': 'File processed successfully',
                'storage': 'local'
            }
        
        except Exception as e:
            current_app.logger.error(f"Processing error: {str(e)}", exc_info=True)
            raise
        
        finally:
            # Cleanup uploaded file
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                    current_app.logger.info(f"Cleaned up: {filepath}")
                except Exception as e:
                    current_app.logger.warning(f"Cleanup failed: {e}")