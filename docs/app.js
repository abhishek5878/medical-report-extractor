document.addEventListener('DOMContentLoaded', function() {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const statusMessage = document.getElementById('statusMessage');
    const resultContainer = document.getElementById('resultContainer');
    const processButton = document.getElementById('processButton');
    const uploadForm = document.getElementById('uploadForm');

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
        if (!file.name.toLowerCase().endsWith('.pdf')) {
            showError('Please upload a PDF file');
            processButton.disabled = true;
            return;
        }
        processButton.disabled = false;
        statusMessage.textContent = `Selected file: ${file.name}`;
        statusMessage.className = 'status-message';
        statusMessage.style.display = 'block';
        resultContainer.style.display = 'none';
    }

    function handleFileUpload(formData) {
        const maxRetries = 3;
        let retryCount = 0;

        // Show loading state with more detailed message
        statusMessage.textContent = 'Processing large PDF file. This may take several minutes...';
        statusMessage.className = 'status-message processing';
        statusMessage.style.display = 'block';
        resultContainer.style.display = 'none';
        processButton.disabled = true;

        async function uploadWithRetry() {
            try {
                const controller = new AbortController();
                const timeoutId = setTimeout(() => controller.abort(), 900000); // 15 minute timeout

                const response = await fetch('/upload', {
                    method: 'POST',
                    body: formData,
                    signal: controller.signal,
                    headers: {
                        'Accept': 'application/json'
                    }
                });

                clearTimeout(timeoutId);

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
                if (!data) {
                    throw new Error('No response data received from server');
                }

                if (data.error) {
                    throw new Error(data.error);
                }

                if (data.success) {
                    statusMessage.textContent = 'File processed successfully!';
                    statusMessage.className = 'status-message success';
                    
                    // Show download link
                    const downloadLink = document.createElement('a');
                    downloadLink.href = `/download/${data.filename}`;
                    downloadLink.textContent = 'Download Results';
                    downloadLink.className = 'download-button';
                    downloadLink.target = '_blank';
                    
                    resultContainer.innerHTML = '';
                    resultContainer.appendChild(downloadLink);
                    resultContainer.style.display = 'block';
                    processButton.disabled = false;
                } else {
                    throw new Error(data.message || 'Processing failed');
                }
            } catch (error) {
                console.error('Error:', error);
                if (retryCount < maxRetries && error.message.includes('Server error')) {
                    retryCount++;
                    statusMessage.textContent = `Retrying upload (${retryCount}/${maxRetries})... This may take a while for large files.`;
                    setTimeout(uploadWithRetry, 5000 * retryCount); // Longer delay between retries
                } else {
                    showError(error.message);
                    processButton.disabled = false;
                }
            }
        }

        uploadWithRetry();
    }

    function showError(message) {
        statusMessage.textContent = `Error: ${message}`;
        statusMessage.className = 'status-message error';
        statusMessage.style.display = 'block';
        resultContainer.style.display = 'none';
        processButton.disabled = false;
    }

    // Form submission handler
    uploadForm.addEventListener('submit', (e) => {
        e.preventDefault();
        const file = fileInput.files[0];
        
        if (!file) {
            showError('Please select a PDF file first');
            return;
        }
        
        const formData = new FormData();
        formData.append('file', file);
        handleFileUpload(formData);
    });
}); 