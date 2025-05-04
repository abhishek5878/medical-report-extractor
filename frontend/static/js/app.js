async function uploadWithRetry(file, maxRetries = 3) {
    let retryCount = 0;
    while (retryCount < maxRetries) {
        try {
            const formData = new FormData();
            formData.append('file', file);

            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                let errorMessage = 'Server error';
                try {
                    const errorData = await response.json();
                    errorMessage = errorData.error || errorMessage;
                } catch (e) {
                    // If we can't parse the error JSON, use the status text
                    errorMessage = `Server error: ${response.status} ${response.statusText}`;
                }
                throw new Error(errorMessage);
            }

            return await response.json();
        } catch (error) {
            retryCount++;
            if (retryCount === maxRetries) {
                throw error;
            }
            // Wait before retrying (exponential backoff)
            await new Promise(resolve => setTimeout(resolve, Math.pow(2, retryCount) * 1000));
        }
    }
} 