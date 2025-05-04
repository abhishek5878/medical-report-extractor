document.addEventListener('DOMContentLoaded', function() {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const statusMessage = document.getElementById('statusMessage');
    const resultContainer = document.getElementById('resultContainer');
    const processButton = document.getElementById('processButton');
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
            processButton.disabled = true;
            return;
        }
        statusMessage.textContent = `Selected file: ${file.name}`;
        statusMessage.className = 'status-message';
        statusMessage.style.display = 'block';
        resultContainer.style.display = 'none';
        processButton.disabled = false;
    }

    async function attemptUpload(file) {
        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('https://medical-report-extractor.onrender.com/upload', {
                method: 'POST',
                body: formData,
                mode: 'cors',
                credentials: 'include',
                headers: {
                    'Accept': 'application/json',
                    'Origin': 'https://abhishek5878.github.io'
                }
            });

            if (!response.ok) {
                let errorMessage = `Server error: ${response.status}`;
                try {
                    const errorData = await response.json();
                    errorMessage = errorData.error || errorMessage;
                } catch (e) {
                    console.error('Error parsing error response:', e);
                }
                throw new Error(errorMessage);
            }

            const data = await response.json();
            return data;
        } catch (error) {
            console.error('Upload error:', error);
            throw error;
        }
    }

    function handleFileUpload(formData) {
        const maxRetries = 3;
        let retryCount = 0;

        // Show loading state
        statusMessage.textContent = 'Processing file...';
        statusMessage.className = 'status-message processing';
        statusMessage.style.display = 'block';
        resultContainer.style.display = 'none';
        processButton.disabled = true;

        async function attemptUpload() {
            try {
                const response = await fetch('https://medical-report-extractor.onrender.com/upload', {
                    method: 'POST',
                    body: formData,
                    mode: 'cors',
                    credentials: 'include',
                    headers: {
                        'Accept': 'application/json',
                        'Origin': 'https://abhishek5878.github.io'
                    }
                });

                if (!response.ok) {
                    let errorMessage = `Server error: ${response.status}`;
                    try {
                        const errorData = await response.json();
                        errorMessage = errorData.error || errorMessage;
                    } catch (e) {
                        console.error('Error parsing error response:', e);
                    }
                    throw new Error(errorMessage);
                }

                const data = await response.json();
                if (data.success) {
                    statusMessage.textContent = 'File processed successfully!';
                    statusMessage.className = 'status-message success';
                    
                    // Show download link
                    const downloadLink = document.createElement('a');
                    downloadLink.href = `https://medical-report-extractor.onrender.com/download/${data.filename}`;
                    downloadLink.textContent = 'Download Results';
                    downloadLink.className = 'download-button';
                    downloadLink.target = '_blank';
                    
                    resultContainer.innerHTML = '';
                    resultContainer.appendChild(downloadLink);
                    resultContainer.style.display = 'block';
                    processButton.disabled = false;
                } else {
                    throw new Error(data.error || 'Processing failed');
                }
            } catch (error) {
                console.error('Error:', error);
                if (retryCount < maxRetries && error.message.includes('Server error')) {
                    retryCount++;
                    statusMessage.textContent = `Retrying upload (${retryCount}/${maxRetries})...`;
                    setTimeout(attemptUpload, 2000 * retryCount); // Exponential backoff
                } else {
                    showError(error.message);
                    processButton.disabled = false;
                }
            }
        }

        attemptUpload();
    }

    function showError(message) {
        statusMessage.textContent = `Error: ${message}`;
        statusMessage.className = 'status-message error';
        statusMessage.style.display = 'block';
        resultContainer.style.display = 'none';
        processButton.disabled = false;
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