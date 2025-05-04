document.addEventListener('DOMContentLoaded', function() {
    const dropZone = document.querySelector('.drop-zone');
    const fileInput = document.getElementById('fileInput');
    const uploadForm = document.getElementById('uploadForm');
    const progressContainer = document.querySelector('.progress-container');
    const progressBar = document.querySelector('.progress-bar');
    const statusMessage = document.getElementById('statusMessage');
    const resultContainer = document.getElementById('resultContainer');
    const downloadBtn = document.getElementById('downloadBtn');
    const validationReportBtn = document.getElementById('validationReportBtn');

    // Initialize status message
    statusMessage.style.display = 'block';

    // Drag and drop handlers
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('active');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('active');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('active');
        const files = e.dataTransfer.files;
        if (files.length) {
            fileInput.files = files;
            handleFileSelect(files[0]);
        }
    });

    // File input change handler
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleFileSelect(e.target.files[0]);
        }
    });

    function handleFileSelect(file) {
        if (file.type !== 'application/pdf') {
            showError('Please upload a PDF file');
            return;
        }
        statusMessage.textContent = `Selected file: ${file.name}`;
        statusMessage.className = 'alert alert-info';
    }

    // Form submission handler
    uploadForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const formData = new FormData(uploadForm);
        
        if (!fileInput.files.length) {
            showError('Please select a PDF file first');
            return;
        }

        try {
            progressContainer.style.display = 'block';
            progressBar.style.width = '0%';
            progressBar.setAttribute('aria-valuenow', 0);
            statusMessage.textContent = 'Uploading file...';
            statusMessage.className = 'alert alert-info';

            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            
            if (result.error) {
                throw new Error(result.error);
            }

            progressBar.style.width = '100%';
            progressBar.setAttribute('aria-valuenow', 100);
            statusMessage.textContent = 'File processed successfully!';
            statusMessage.className = 'alert alert-success';
            
            // Show result container
            resultContainer.style.display = 'block';
            downloadBtn.href = `/download/${result.filename}`;
            if (result.validation_report) {
                validationReportBtn.href = `/download/${result.validation_report}`;
                validationReportBtn.style.display = 'inline-block';
            } else {
                validationReportBtn.style.display = 'none';
            }
        } catch (error) {
            console.error('Error:', error);
            showError(error.message || 'An error occurred while processing the file');
        }
    });

    function showError(message) {
        statusMessage.textContent = message;
        statusMessage.className = 'alert alert-danger';
        progressContainer.style.display = 'none';
        resultContainer.style.display = 'none';
    }
}); 