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
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024  # 8MB max file size
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
        logger.info("Upload request received")
        
        if 'file' not in request.files:
            logger.error("No file part in request")
            return jsonify({'success': False, 'error': 'No file provided'}), 400
        
        file = request.files['file']
        logger.info(f"File received: {file.filename}")
        
        if file.filename == '':
            logger.error("No file selected")
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        if not file.filename.lower().endswith('.pdf'):
            logger.error("Invalid file type")
            return jsonify({'success': False, 'error': 'Please upload a PDF file'}), 400
        
        # Check file size before saving
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        logger.info(f"File size: {file_size} bytes")
        
        if file_size > app.config['MAX_CONTENT_LENGTH']:
            logger.error(f"File too large: {file_size} bytes")
            return jsonify({'error': 'File too large. Maximum size is 8MB.'}), 413
        
        # Save the file temporarily
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(temp_path)
        
        try:
            # Process the PDF
            result = process_report(temp_path)
            
            # Save the result to a temporary file
            output_filename = f"results_{os.path.splitext(file.filename)[0]}.xlsx"
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
            result.to_excel(output_path, index=False)
            
            return jsonify({
                'success': True,
                'message': 'File processed successfully',
                'filename': output_filename
            })
            
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            return jsonify({'success': False, 'error': str(e)}), 500
            
        finally:
            # Clean up the uploaded file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    except Exception as e:
        logger.error(f"Error handling upload: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/download/<filename>')
def download_file(filename):
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(file_path):
            return jsonify({'success': False, 'error': 'File not found'}), 404
            
        return send_file(
            file_path,
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        logger.error(f"Error downloading file: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/health')
def health_check():
    logger.info("Health check endpoint called")
    return jsonify({"status": "healthy"}), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    logger.info(f"Starting application on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)