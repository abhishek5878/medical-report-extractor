from flask import Flask, render_template, request, send_file, flash, redirect, url_for
import os
import tempfile
import uuid
import pandas as pd
from werkzeug.utils import secure_filename
import time
import json
from processors import process_report, create_validation_report

app = Flask(__name__)
app.secret_key = "medical_report_extractor_secret_key"
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Path to data files
MAPPING_FILE = os.path.join('data', 'marker_mapping.csv')
TRUE_DATA_FILE = os.path.join('data', 'true_data.csv')

# Allowed extensions for upload
ALLOWED_EXTENSIONS = {'pdf'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        # Generate unique filename
        unique_filename = str(uuid.uuid4()) + '.pdf'
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        try:
            # Load mapping to get test names
            mapping_df = pd.read_csv(MAPPING_FILE)
            thyrocare_tests = mapping_df['Thyrocare Test Name'].dropna().tolist()
            
            # Process the PDF
            results = process_report(filepath, thyrocare_tests, batch_size=50)
            
            # Create result DataFrame
            result_df = pd.DataFrame({
                "Test Name": list(results.keys()),
                "Value": list(results.values())
            })
            
            # Generate output filename
            output_filename = f"results_{int(time.time())}.xlsx"
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
            
            # Generate validation report
            create_validation_report(
                true_data_path=TRUE_DATA_FILE,
                result_df=result_df,
                mapping_path=MAPPING_FILE,
                output_excel_path=output_path
            )
            
            # Clean up the uploaded PDF
            os.remove(filepath)
            
            # Return the Excel file
            return send_file(output_path, as_attachment=True, download_name='medical_report_results.xlsx')
            
        except Exception as e:
            flash(f'Error processing file: {str(e)}')
            return redirect(url_for('index'))
    else:
        flash('Only PDF files are allowed')
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)