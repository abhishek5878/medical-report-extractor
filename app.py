from flask import Flask, render_template, request, send_file, flash, redirect, url_for, jsonify, g
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
from flask_cors import CORS
import signal
from datetime import timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define paths
MAPPING_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'true_data.csv')

app = Flask(__name__, static_folder='docs', static_url_path='')
app.secret_key = "medical_report_extractor_secret_key"
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
app.config['UPLOAD_EXTENSIONS'] = ['.pdf']
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=15)
app.config['TIMEOUT'] = 900  # 15 minutes timeout

# Configure CORS with more permissive settings
CORS(app, resources={
    r"/*": {
        "origins": ["https://abhishek5878.github.io", "http://localhost:5000"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Accept", "Origin", "X-Requested-With"],
        "expose_headers": ["Content-Type", "Content-Length"],
        "supports_credentials": False,
        "max_age": 3600
    }
})

# Configure Talisman with more permissive CSP
Talisman(app, content_security_policy={
    'default-src': "'self'",
    'script-src': ["'self'", "'unsafe-inline'", "'unsafe-eval'", "https://cdn.jsdelivr.net"],
    'style-src': ["'self'", "'unsafe-inline'", "https://cdn.jsdelivr.net"],
    'img-src': ["'self'", "data:", "https://cdn.jsdelivr.net"],
    'connect-src': ["'self'", "https://medical-report-extractor.onrender.com", "https://abhishek5878.github.io", "http://localhost:5000"]
})

@app.context_processor
def inject_nonce():
    return {'csp_nonce': os.urandom(16).hex()}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'pdf'

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'error': 'File too large. Maximum size is 8MB.'}), 413

@app.route('/')
def index():
    return app.send_static_file('index.html')

def get_test_names():
    """Load test names from the mapping file"""
    try:
        if not os.path.exists(MAPPING_FILE):
            logger.error(f"Mapping file not found: {MAPPING_FILE}")
            return None
            
        mapping_df = pd.read_csv(MAPPING_FILE)
        test_names = mapping_df['Name'].dropna().tolist()
        
        if not test_names:
            logger.error("No test names found in mapping file")
            return None
            
        logger.info(f"Loaded {len(test_names)} test names from mapping file")
        return test_names
        
    except Exception as e:
        logger.error(f"Error loading test names: {str(e)}")
        return None

def save_results_to_excel(results, filename):
    """Save results to Excel file"""
    try:
        # Create result DataFrame
        result_df = pd.DataFrame({
            "Test Name": list(results.keys()),
            "Value": list(results.values())
        })
        
        # Create result filename
        result_filename = f"results_{filename}"
        result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
        
        # Save to Excel
        result_df.to_excel(result_path, index=False)
        
        # Verify file was saved
        if not os.path.exists(result_path):
            logger.error(f"Failed to save results file: {result_path}")
            return None
            
        return result_path
        
    except Exception as e:
        logger.error(f"Error saving results to Excel: {str(e)}")
        return None

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
            
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed'}), 400
            
        # Create upload directory if it doesn't exist
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        # Save the file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Get test names
        test_names = get_test_names()
        if not test_names:
            return jsonify({'error': 'Could not load test names'}), 500
            
        # Process the PDF
        try:
            results = process_report(filepath, test_names)
            if not results:
                return jsonify({'error': 'No results could be extracted from the PDF'}), 400
                
            # Create validation report
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'validation_report.xlsx')
            create_validation_report(
                app.config['MAPPING_FILE'],
                pd.DataFrame([{'Test Name': k, 'Value': v} for k, v in results.items()]),
                app.config['MAPPING_FILE'],
                output_path
            )
            
            # Return success response
            return jsonify({
                'message': 'File processed successfully',
                'results': results,
                'download_url': f'/download/{os.path.basename(output_path)}'
            })
            
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            return jsonify({'error': f'Error processing PDF: {str(e)}'}), 500
            
        finally:
            # Clean up uploaded file
            try:
                os.remove(filepath)
            except Exception as e:
                logger.error(f"Error cleaning up file: {str(e)}")
                
    except Exception as e:
        logger.error(f"Error in upload_file: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/download/<filename>')
def download_file(filename):
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404
            
        return send_file(
            filepath,
            as_attachment=True,
            download_name=filename,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
    except Exception as e:
        logger.error(f"Error downloading file: {str(e)}")
        return jsonify({'error': f'Error downloading file: {str(e)}'}), 500

@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy'})

@app.before_request
def before_request():
    g.start = time.time()

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', 'https://abhishek5878.github.io')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Accept')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    response.headers.add('Access-Control-Allow-Credentials', 'false')
    
    # Add timeout header
    response.headers.add('X-Timeout', str(app.config['TIMEOUT']))
    
    # Log request time
    if hasattr(g, 'start'):
        elapsed = time.time() - g.start
        logger.info(f"Request took {elapsed:.2f} seconds")
    
    return response

if __name__ == '__main__':
    app.run(debug=True)