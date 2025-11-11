class PoseDetectionApp {
    constructor() {
        this.video = document.getElementById('video');
        this.canvas = document.getElementById('canvas');
        this.ctx = this.canvas.getContext('2d');
        this.overlay = document.getElementById('overlay');
        
        this.startCameraBtn = document.getElementById('startCamera');
        this.capturePoseBtn = document.getElementById('capturePose');
        this.stopCameraBtn = document.getElementById('stopCamera');
        
        this.resultsSection = document.getElementById('resultsSection');
        this.statsDiv = document.getElementById('stats');
        this.annotatedImage = document.getElementById('annotatedImage');
        this.landmarksList = document.getElementById('landmarksList');
        this.statusDiv = document.getElementById('status');
        
        this.stream = null;
        this.isCameraOn = false;
        
        this.initEventListeners();
    }
    
    initEventListeners() {
        this.startCameraBtn.addEventListener('click', () => this.startCamera());
        this.capturePoseBtn.addEventListener('click', () => this.capturePose());
        this.stopCameraBtn.addEventListener('click', () => this.stopCamera());
    }
    
    async startCamera() {
        try {
            this.updateStatus('🔄 Starting camera...', 'loading');
            
            this.stream = await navigator.mediaDevices.getUserMedia({ 
                video: { 
                    width: { ideal: 640 },
                    height: { ideal: 480 } 
                } 
            });
            
            this.video.srcObject = this.stream;
            this.isCameraOn = true;
            
            this.video.onloadedmetadata = () => {
                this.canvas.width = this.video.videoWidth;
                this.canvas.height = this.video.videoHeight;
                this.updateUI();
                this.updateStatus('✅ Camera ready! Click "Detect Pose"', 'success');
            };
            
        } catch (error) {
            console.error('Camera error:', error);
            this.updateStatus('❌ Camera access denied: ' + error.message, 'error');
        }
    }
    
    async capturePose() {
        if (!this.isCameraOn) {
            this.updateStatus('❌ Please start camera first', 'error');
            return;
        }
        
        try {
            this.updateStatus('🔍 Analyzing pose...', 'loading');
            this.capturePoseBtn.disabled = true;
            
            // Capture frame
            this.ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
            const imageData = this.canvas.toDataURL('image/jpeg', 0.8);
            
            // Send to server
            const response = await fetch('/detect', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ image: imageData })
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.displayResults(result);
                this.updateStatus('✅ Pose detected successfully!', 'success');
            } else {
                throw new Error(result.error || 'Detection failed');
            }
            
        } catch (error) {
            console.error('Detection error:', error);
            this.updateStatus('❌ Detection failed: ' + error.message, 'error');
        } finally {
            this.capturePoseBtn.disabled = false;
        }
    }
    
    stopCamera() {
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }
        
        this.isCameraOn = false;
        this.video.srcObject = null;
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.updateUI();
        this.updateStatus('⏹️ Camera stopped', 'loading');
    }
    
    displayResults(result) {
        this.resultsSection.style.display = 'block';
        
        // Statistics
        this.statsDiv.innerHTML = `
            <p>🎯 Pose Type: <strong>${this.getPoseTypeName(result.pose_type)}</strong></p>
            <p>📍 Landmarks Detected: <strong>${result.landmarks.length}</strong></p>
            <p>🔗 Connections: <strong>${result.connections.length}</strong></p>
        `;
        
        // Annotated Image
        if (result.annotated_image) {
            this.annotatedImage.src = result.annotated_image;
        }
        
        // Landmarks List
        this.landmarksList.innerHTML = result.landmarks
            .map(landmark => `
                <div class="landmark-item">
                    <div>Point ${landmark.index}</div>
                    <div>(${landmark.x.toFixed(2)}, ${landmark.y.toFixed(2)})</div>
                    <div>${Math.round(landmark.visibility * 100)}%</div>
                </div>
            `)
            .join('');
        
        // Scroll to results
        this.resultsSection.scrollIntoView({ behavior: 'smooth' });
    }
    
    getPoseTypeName(poseType) {
        const names = {
            'hands_up': '🙌 Hands Up',
            'left_hand_up': '👈 Left Hand Up',
            'right_hand_up': '👉 Right Hand Up',
            'standing': '🧍 Standing',
            'unknown': '❓ Unknown'
        };
        return names[poseType] || poseType;
    }
    
    updateUI() {
        this.startCameraBtn.disabled = this.isCameraOn;
        this.capturePoseBtn.disabled = !this.isCameraOn;
        this.stopCameraBtn.disabled = !this.isCameraOn;
    }
    
    updateStatus(message, type = 'loading') {
        this.statusDiv.textContent = message;
        this.statusDiv.className = `status ${type}`;
    }
}

// Инициализация приложения когда страница загружена
document.addEventListener('DOMContentLoaded', () => {
    new PoseDetectionApp();
});
