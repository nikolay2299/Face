document.addEventListener('DOMContentLoaded', function() {
    // DOM elements
    const imageUpload = document.getElementById('imageUpload');
    const captureButton = document.getElementById('captureButton');
    const cameraContainer = document.getElementById('cameraContainer');
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const capturePhotoButton = document.getElementById('capturePhoto');
    const cancelCaptureButton = document.getElementById('cancelCapture');
    const resultContainer = document.getElementById('resultContainer');
    const loadingSpinner = document.getElementById('loadingSpinner');
    const errorMessage = document.getElementById('errorMessage');
    
    // Track if the camera is currently active
    let isCameraActive = false;
    let stream = null;
    
    // Listen for file uploads
    imageUpload.addEventListener('change', function(e) {
        if (e.target.files.length > 0) {
            const file = e.target.files[0];
            
            // Check if the file is an image
            if (!file.type.match('image.*')) {
                showError('Please select an image file (jpg, png, etc.)');
                return;
            }
            
            // Display loading state
            showLoading();
            
            // Process the image
            processImageFile(file);
        }
    });
    
    // Handle camera button click
    if (captureButton) {
        captureButton.addEventListener('click', function() {
            if (isCameraActive) {
                stopCamera();
            } else {
                startCamera();
            }
        });
    }
    
    // Capture photo from camera
    if (capturePhotoButton) {
        capturePhotoButton.addEventListener('click', function() {
            if (isCameraActive) {
                const photo = capturePhoto();
                stopCamera();
                
                // Display loading state
                showLoading();
                
                // Process the captured photo
                processImageData(photo);
            }
        });
    }
    
    // Cancel camera capture
    if (cancelCaptureButton) {
        cancelCaptureButton.addEventListener('click', function() {
            stopCamera();
        });
    }
    
    // Function to start the camera
    function startCamera() {
        if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
            navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' } })
                .then(function(mediaStream) {
                    stream = mediaStream;
                    video.srcObject = mediaStream;
                    video.play();
                    isCameraActive = true;
                    
                    // Show camera interface
                    cameraContainer.classList.remove('d-none');
                    captureButton.textContent = 'Cancel';
                    captureButton.classList.remove('btn-primary');
                    captureButton.classList.add('btn-danger');
                })
                .catch(function(err) {
                    console.error('Error accessing camera: ', err);
                    showError('Could not access the camera. Please check permissions or use the file upload option.');
                });
        } else {
            showError('Your browser does not support camera access. Please use the file upload option.');
        }
    }
    
    // Function to stop the camera
    function stopCamera() {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            video.srcObject = null;
            stream = null;
        }
        
        isCameraActive = false;
        
        // Hide camera interface
        cameraContainer.classList.add('d-none');
        captureButton.textContent = 'Use Camera';
        captureButton.classList.remove('btn-danger');
        captureButton.classList.add('btn-primary');
    }
    
    // Function to capture a photo from the video
    function capturePhoto() {
        const context = canvas.getContext('2d');
        
        // Set canvas dimensions to match video
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        
        // Draw the current video frame to the canvas
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        
        // Convert canvas to data URL
        return canvas.toDataURL('image/jpeg');
    }
    
    // Function to process the uploaded image file
    function processImageFile(file) {
        const formData = new FormData();
        formData.append('image', file);
        
        fetch('/analyze', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => handleAnalysisResult(data))
        .catch(error => {
            console.error('Error:', error);
            showError('An error occurred while processing the image. Please try again.');
        });
    }
    
    // Function to process image data from camera
    function processImageData(imageData) {
        fetch('/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: 'image_data=' + encodeURIComponent(imageData)
        })
        .then(response => response.json())
        .then(data => handleAnalysisResult(data))
        .catch(error => {
            console.error('Error:', error);
            showError('An error occurred while processing the image. Please try again.');
        });
    }
    
    // Function to handle the analysis result
    function handleAnalysisResult(data) {
        // Hide loading spinner
        hideLoading();
        
        if (data.error) {
            showError(data.error);
            return;
        }
        
        if (!data.success) {
            showError(data.message || 'An unknown error occurred during face analysis.');
            return;
        }
        
        // Update result container with analysis results
        resultContainer.innerHTML = `
            <div class="card mb-4">
                <div class="card-header">
                    <h3 class="mb-0">Face Shape Analysis Result</h3>
                </div>
                <div class="card-body">
                    <div class="row">
                        <div class="col-md-6">
                            <img src="${data.image_with_landmarks}" alt="Face with landmarks" class="img-fluid mb-3">
                        </div>
                        <div class="col-md-6">
                            <h4 class="text-primary">Your Face Shape: <span class="badge bg-primary">${data.face_shape.toUpperCase()}</span></h4>
                            <p>${data.description}</p>
                            
                            <h5 class="mt-4">Face Measurements:</h5>
                            <ul class="list-group">
                                <li class="list-group-item d-flex justify-content-between align-items-center">
                                    Face Height
                                    <span class="badge bg-secondary rounded-pill">${data.measurements.face_height}</span>
                                </li>
                                <li class="list-group-item d-flex justify-content-between align-items-center">
                                    Forehead Width
                                    <span class="badge bg-secondary rounded-pill">${data.measurements.face_width_top}</span>
                                </li>
                                <li class="list-group-item d-flex justify-content-between align-items-center">
                                    Cheekbone Width
                                    <span class="badge bg-secondary rounded-pill">${data.measurements.face_width_middle}</span>
                                </li>
                                <li class="list-group-item d-flex justify-content-between align-items-center">
                                    Jawline Width
                                    <span class="badge bg-secondary rounded-pill">${data.measurements.face_width_bottom}</span>
                                </li>
                                <li class="list-group-item d-flex justify-content-between align-items-center">
                                    Width-to-Height Ratio
                                    <span class="badge bg-secondary rounded-pill">${data.measurements.width_to_height_ratio}</span>
                                </li>
                            </ul>
                            
                            <div class="alert alert-info mt-3">
                                <strong>Analysis Confidence:</strong> ${Math.round(data.confidence * 100)}%
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        // Show the result container
        resultContainer.classList.remove('d-none');
        
        // Scroll to results
        resultContainer.scrollIntoView({ behavior: 'smooth' });
    }
    
    // Show loading spinner
    function showLoading() {
        loadingSpinner.classList.remove('d-none');
        errorMessage.classList.add('d-none');
        resultContainer.classList.add('d-none');
    }
    
    // Hide loading spinner
    function hideLoading() {
        loadingSpinner.classList.add('d-none');
    }
    
    // Show error message
    function showError(message) {
        hideLoading();
        errorMessage.textContent = message;
        errorMessage.classList.remove('d-none');
    }
});
