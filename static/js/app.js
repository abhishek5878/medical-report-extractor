document.addEventListener('DOMContentLoaded', function() {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const statusMessage = document.getElementById('statusMessage');
    const resultContainer = document.getElementById('resultContainer');
    const API_URL = 'https://medical-report-extractor.onrender.com'; // Your Render backend URL

    // Click handler for drop zone
    dropZone.addEventListener('click', () => {
        fileInput.click();
    });

    // Drag and drop handlers
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.add('active');
    });

    dropZone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.remove('active');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        e.stopPropagation();
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
        statusMessage.className = 'status-message';
        statusMessage.style.display = 'block';
        resultContainer.style.display = 'none';
    }

    function handleFileUpload(formData) {
        const maxRetries = 3;
        let retryCount = 0;

        // Show loading state
        statusMessage.textContent = 'Processing file...';
        statusMessage.className = 'status-message processing';
        statusMessage.style.display = 'block';
        resultContainer.style.display = 'none';

        function attemptUpload() {
            fetch(`${API_URL}/upload`, {
                method: 'POST',
                body: formData
            })
            .then(response => {
                if (!response.ok) {
                    return response.text().then(text => {
                        try {
                            const data = JSON.parse(text);
                            throw new Error(data.error || 'Upload failed');
                        } catch (e) {
                            throw new Error(`Server error: ${response.status} ${response.statusText}`);
                        }
                    });
                }
                return response.text().then(text => {
                    try {
                        return JSON.parse(text);
                    } catch (e) {
                        throw new Error('Invalid server response');
                    }
                });
            })
            .then(data => {
                if (data.success) {
                    statusMessage.textContent = 'File processed successfully!';
                    statusMessage.className = 'status-message success';
                    
                    // Show download link
                    const downloadLink = document.createElement('a');
                    downloadLink.href = `${API_URL}/download/${data.filename}`;
                    downloadLink.textContent = 'Download Results';
                    downloadLink.className = 'download-button';
                    downloadLink.target = '_blank'; // Open in new tab
                    
                    // Add click handler for direct download
                    downloadLink.addEventListener('click', function(e) {
                        e.preventDefault();
                        fetch(this.href)
                            .then(response => {
                                if (!response.ok) throw new Error('Download failed');
                                return response.blob();
                            })
                            .then(blob => {
                                const url = window.URL.createObjectURL(blob);
                                const a = document.createElement('a');
                                a.href = url;
                                a.download = data.filename;
                                document.body.appendChild(a);
                                a.click();
                                window.URL.revokeObjectURL(url);
                                a.remove();
                            })
                            .catch(error => {
                                console.error('Download error:', error);
                                showError('Failed to download the file. Please try again.');
                            });
                    });
                    
                    resultContainer.innerHTML = '';
                    resultContainer.appendChild(downloadLink);
                    resultContainer.style.display = 'block';
                } else {
                    throw new Error(data.error || 'Processing failed');
                }
            })
            .catch(error => {
                console.error('Error:', error);
                if (retryCount < maxRetries && error.message.includes('Server error')) {
                    retryCount++;
                    statusMessage.textContent = `Retrying upload (${retryCount}/${maxRetries})...`;
                    setTimeout(attemptUpload, 2000 * retryCount); // Exponential backoff
                } else {
                    showError(error.message);
                }
            });
        }

        attemptUpload();
    }

    function showError(message) {
        statusMessage.textContent = `Error: ${message}`;
        statusMessage.className = 'status-message error';
        statusMessage.style.display = 'block';
        resultContainer.style.display = 'none';
    }

    // Form submission handler
    document.getElementById('uploadForm').addEventListener('submit', (e) => {
        e.preventDefault();
        const formData = new FormData();
        const file = fileInput.files[0];
        
        if (!file) {
            showError('Please select a PDF file first');
            return;
        }
        
        formData.append('file', file);
        handleFileUpload(formData);
    });
}); 