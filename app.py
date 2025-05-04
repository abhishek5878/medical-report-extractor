from flask import Flask, render_template, request, send_file, flash, redirect, url_for, jsonify
import os
import tempfile
import uuid
import pandas as pd
from werkzeug.utils import secure_filename
import time
import json
import logging
from processors import process_report, create_validation_report
from flask_talisman import Talisman

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = "medical_report_extractor_secret_key"
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024  # 8MB max upload size
app.config['UPLOAD_EXTENSIONS'] = ['.pdf']

# Add nonce generation function
@app.context_processor
def inject_nonce():
    return {'csp_nonce': lambda: str(uuid.uuid4())}

# Configure Content Security Policy
csp = {
    'default-src': '\'self\'',
    'script-src': [
        '\'self\'',
        '\'unsafe-inline\'',
        'https://cdn.jsdelivr.net',
        'https://code.jquery.com'
    ],
    'style-src': [
        '\'self\'',
        '\'unsafe-inline\'',
        'https://cdn.jsdelivr.net',
        'https://fonts.googleapis.com'
    ],
    'img-src': [
        '\'self\'',
        'data:',
        'https:'
    ],
    'font-src': [
        '\'self\'',
        'https://fonts.gstatic.com',
        'https://cdn.jsdelivr.net'
    ]
}

# Initialize Talisman with CSP
talisman = Talisman(
    app,
    content_security_policy=csp,
    content_security_policy_nonce_in=['script-src'],
    force_https=True,
    strict_transport_security=True,
    session_cookie_secure=True,
    session_cookie_http_only=True
)

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Path to data files
MAPPING_FILE = os.path.join('data', 'marker_mapping.csv')
TRUE_DATA_FILE = os.path.join('data', 'true_data.csv')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'pdf'}

@app.errorhandler(413)
def request_entity_too_large(error):
    flash('File too large. Maximum size is 8MB.')
    return redirect(url_for('index'))

@app.route('/')
def index():
    logger.info("Index page accessed")
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        logger.debug("Upload request received")
        logger.debug(f"Request files: {request.files}")
        
        if 'file' not in request.files:
            logger.error("No file part in request")
            flash('No file part in the request')
            return redirect(url_for('index'))
        
        file = request.files['file']
        logger.debug(f"File received: {file.filename}")
        
        if file.filename == '':
            logger.error("No file selected")
            flash('No file selected')
            return redirect(url_for('index'))
        
        if not allowed_file(file.filename):
            logger.error(f"Invalid file type: {file.filename}")
            flash('Only PDF files are allowed')
            return redirect(url_for('index'))
        
        # Check file size before saving
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        logger.debug(f"File size: {file_size} bytes")
        
        if file_size > app.config['MAX_CONTENT_LENGTH']:
            logger.error(f"File too large: {file_size} bytes")
            flash('File too large. Maximum size is 8MB.')
            return redirect(url_for('index'))
        
        # Generate unique filename
        unique_filename = str(uuid.uuid4()) + '.pdf'
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        logger.debug(f"Saving file to: {filepath}")
        
        try:
            # Save file in chunks to reduce memory usage
            chunk_size = 8192  # 8KB chunks
            with open(filepath, 'wb') as f:
                while True:
                    chunk = file.stream.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
            
            # Verify file was saved correctly
            if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
                raise ValueError("File was not saved correctly")
            
            logger.debug("File saved successfully")
            
            # Load mapping to get test names
            try:
                mapping_df = pd.read_csv(MAPPING_FILE)
                thyrocare_tests = mapping_df['Thyrocare Test Name'].dropna().tolist()
                if not thyrocare_tests:
                    raise ValueError("No test names found in mapping file")
            except Exception as e:
                logger.error(f"Error loading test names: {str(e)}")
                flash(f'Error loading test names: {str(e)}')
                os.remove(filepath)
                return redirect(url_for('index'))
            
            # Process the PDF with timeout
            try:
                results = process_report(filepath, thyrocare_tests, batch_size=10)
                if not results:
                    raise ValueError("No values could be extracted from the PDF")
            except Exception as e:
                logger.error(f"Error processing PDF: {str(e)}")
                flash(f'Error processing PDF: {str(e)}')
                os.remove(filepath)
                return redirect(url_for('index'))
            
            # Create result DataFrame
            result_df = pd.DataFrame({
                "Test Name": list(results.keys()),
                "Value": list(results.values())
            })
            
            # Generate output filename
            output_filename = f"results_{int(time.time())}.xlsx"
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
            
            # Generate validation report
            try:
                create_validation_report(
                    true_data_path=TRUE_DATA_FILE,
                    result_df=result_df,
                    mapping_path=MAPPING_FILE,
                    output_excel_path=output_path
                )
            except Exception as e:
                logger.error(f"Error creating validation report: {str(e)}")
                flash(f'Error creating validation report: {str(e)}')
                os.remove(filepath)
                return redirect(url_for('index'))
            
            # Clean up the uploaded PDF
            os.remove(filepath)
            
            # Return the Excel file
            return send_file(
                output_path,
                as_attachment=True,
                download_name='medical_report_results.xlsx',
                mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )
            
        except Exception as e:
            logger.error(f"Error saving file: {str(e)}")
            flash(f'Error saving file: {str(e)}')
            if os.path.exists(filepath):
                os.remove(filepath)
            return redirect(url_for('index'))
            
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        flash(f'Unexpected error: {str(e)}')
        if 'filepath' in locals() and os.path.exists(filepath):
            try:
                os.remove(filepath)
            except:
                pass
        return redirect(url_for('index'))

@app.route('/health')
def health_check():
    logger.info("Health check endpoint called")
    return jsonify({"status": "healthy"}), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    logger.info(f"Starting application on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)