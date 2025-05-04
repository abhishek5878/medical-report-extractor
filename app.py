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
from flask_cors import CORS
import signal

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = "medical_report_extractor_secret_key"
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024  # 8MB max upload size
app.config['UPLOAD_EXTENSIONS'] = ['.pdf']

# Configure CORS with more permissive settings
CORS(app, resources={
    r"/*": {
        "origins": ["https://abhishek5878.github.io", "http://localhost:5000"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Accept", "Origin", "X-Requested-With"],
        "expose_headers": ["Content-Type", "Content-Length"],
        "supports_credentials": True,
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
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            logger.error("No file part in request")
            return jsonify({'error': 'No file part in the request'}), 400
        
        file = request.files['file']
        if file.filename == '':
            logger.error("No file selected")
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            logger.error(f"Invalid file type: {file.filename}")
            return jsonify({'error': 'Only PDF files are allowed'}), 400
        
        # Check file size before saving
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        
        if file_size > app.config['MAX_CONTENT_LENGTH']:
            return jsonify({'error': 'File too large. Maximum size is 8MB.'}), 413
        
        # Generate unique filename
        unique_filename = str(uuid.uuid4()) + '.pdf'
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        logger.info(f"Saving file to: {filepath}")
        
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
            
            logger.info("File saved successfully")
            
            # Load test names from mapping file
            try:
                if not os.path.exists(MAPPING_FILE):
                    raise ValueError(f"Mapping file not found: {MAPPING_FILE}")
                    
                mapping_df = pd.read_csv(MAPPING_FILE)
                test_names = mapping_df['Thyrocare Test Name'].dropna().tolist()
                if not test_names:
                    raise ValueError("No test names found in mapping file")
                logger.info(f"Loaded {len(test_names)} test names from mapping file")
                
            except Exception as e:
                logger.error(f"Error loading test names: {str(e)}")
                if os.path.exists(filepath):
                    os.remove(filepath)
                return jsonify({'error': f'Error loading test names: {str(e)}'}), 500
            
            # Process the file
            try:
                # Set a timeout for the processing
                def timeout_handler(signum, frame):
                    raise TimeoutError("Processing timed out")
                
                # Set the timeout to 5 minutes
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(300)  # 5 minutes timeout
                
                try:
                    results = process_report(filepath, test_names, batch_size=10)
                    signal.alarm(0)  # Disable the alarm
                    
                    if not results:
                        raise ValueError("No values could be extracted from the PDF")
                    
                    # Create result DataFrame
                    result_df = pd.DataFrame({
                        "Test Name": list(results.keys()),
                        "Value": list(results.values())
                    })
                    
                    # Create result filename
                    result_filename = f"results_{int(time.time())}.xlsx"
                    result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
                    
                    # Save results to Excel
                    result_df.to_excel(result_path, index=False)
                    
                    # Clean up the uploaded PDF
                    os.remove(filepath)
                    
                    response_data = {
                        'success': True,
                        'filename': result_filename,
                        'message': 'File processed successfully'
                    }
                    response = jsonify(response_data)
                    response.headers['Content-Type'] = 'application/json'
                    response.headers['Content-Length'] = len(json.dumps(response_data))
                    return response
                    
                except TimeoutError:
                    logger.error("Processing timed out")
                    if os.path.exists(filepath):
                        os.remove(filepath)
                    response_data = {'error': 'Processing timed out. Please try again with a smaller file.'}
                    response = jsonify(response_data)
                    response.headers['Content-Type'] = 'application/json'
                    response.headers['Content-Length'] = len(json.dumps(response_data))
                    return response, 500
                    
            except ValueError as e:
                logger.error(f"Error processing PDF: {str(e)}")
                if os.path.exists(filepath):
                    os.remove(filepath)
                response_data = {'error': str(e)}
                response = jsonify(response_data)
                response.headers['Content-Type'] = 'application/json'
                response.headers['Content-Length'] = len(json.dumps(response_data))
                return response, 500
            except Exception as e:
                logger.error(f"Error processing PDF: {str(e)}")
                if os.path.exists(filepath):
                    os.remove(filepath)
                response_data = {'error': f'Error processing PDF: {str(e)}'}
                response = jsonify(response_data)
                response.headers['Content-Length'] = len(json.dumps(response_data))
                return response, 500
                
        except Exception as e:
            logger.error(f"Error saving file: {str(e)}")
            if os.path.exists(filepath):
                os.remove(filepath)
            response_data = {'error': f'Error saving file: {str(e)}'}
            response = jsonify(response_data)
            response.headers['Content-Type'] = 'application/json'
            response.headers['Content-Length'] = len(json.dumps(response_data))
            return response, 500
            
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        if 'filepath' in locals() and os.path.exists(filepath):
            try:
                os.remove(filepath)
            except:
                pass
        response_data = {'error': f'Unexpected error: {str(e)}'}
        response = jsonify(response_data)
        response.headers['Content-Type'] = 'application/json'
        response.headers['Content-Length'] = len(json.dumps(response_data))
        return response, 500

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

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', 'https://abhishek5878.github.io')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Accept')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response

if __name__ == '__main__':
    app.run(debug=True)