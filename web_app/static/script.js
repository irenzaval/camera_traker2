class PoseDetectionApp {
    constructor() {
        this.video = document.getElementById('video');
        this.canvas = document.getElementById('canvas');
        this.ctx = this.canvas.getContext('2d');

        this.startCameraBtn = document.getElementById('startCamera');
        this.capturePoseBtn = document.getElementById('capturePose');
        this.stopCameraBtn = document.getElementById('stopCamera');
        this.uploadImageBtn = document.getElementById('uploadImage');

        this.resultsSection = document.getElementById('resultsSection');
        this.statsDiv = document.getElementById('stats');
        this.annotatedImage = document.getElementById('annotatedImage');
        this.landmarksList = document.getElementById('landmarksList');
        this.statusDiv = document.getElementById('status');
        this.fileInput = document.getElementById('fileInput');

        this.stream = null;
        this.isCameraOn = false;

        this.initEventListeners();
        this.checkCameraSupport();
    }

    checkCameraSupport() {
        // Проверяем поддержку камеры
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            this.updateStatus('❌ Ваш браузер не поддерживает доступ к камере', 'error');
            this.startCameraBtn.disabled = true;
            this.startCameraBtn.innerHTML = '📷 Камера не поддерживается';
            return false;
        }

        // Проверяем HTTPS
        if (location.protocol !== 'https:' && location.hostname !== 'localhost' && location.hostname !== '127.0.0.1') {
            this.updateStatus('⚠️ Для доступа к камере нужен HTTPS', 'loading');
        }

        return true;
    }

    initEventListeners() {
        this.startCameraBtn.addEventListener('click', () => this.startCamera());
        this.capturePoseBtn.addEventListener('click', () => this.capturePose());
        this.stopCameraBtn.addEventListener('click', () => this.stopCamera());
        this.uploadImageBtn.addEventListener('click', () => this.fileInput.click());
        this.fileInput.addEventListener('change', (e) => this.handleFileUpload(e));
    }

    async startCamera() {
        if (!this.checkCameraSupport()) return;

        try {
            this.updateStatus('🔄 Запускаем камеру...', 'loading');

            // Параметры камеры
            const constraints = {
                video: {
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    facingMode: 'user' // Фронтальная камера
                },
                audio: false
            };

            this.stream = await navigator.mediaDevices.getUserMedia(constraints);
            this.video.srcObject = this.stream;
            this.isCameraOn = true;

            this.video.onloadedmetadata = () => {
                this.canvas.width = this.video.videoWidth;
                this.canvas.height = this.video.videoHeight;
                this.updateUI();
                this.updateStatus('✅ Камера готова! Нажмите "Определить позу"', 'success');
            };

            this.video.onerror = () => {
                this.updateStatus('❌ Ошибка видеопотока', 'error');
            };

        } catch (error) {
            console.error('Camera error:', error);
            this.handleCameraError(error);
        }
    }

    handleCameraError(error) {
        let errorMessage = 'Неизвестная ошибка камеры';

        switch(error.name) {
            case 'NotAllowedError':
                errorMessage = '❌ Доступ к камере запрещен. Разрешите доступ в настройках браузера';
                break;
            case 'NotFoundError':
                errorMessage = '❌ Камера не найдена';
                break;
            case 'NotSupportedError':
                errorMessage = '❌ Ваш браузер не поддерживает доступ к камере';
                break;
            case 'NotReadableError':
                errorMessage = '❌ Камера уже используется другим приложением';
                break;
            case 'OverconstrainedError':
                errorMessage = '❌ Запрошенные параметры камеры не поддерживаются';
                break;
            default:
                errorMessage = `❌ Ошибка камеры: ${error.message}`;
        }

        this.updateStatus(errorMessage, 'error');
    }

    async capturePose() {
        if (!this.isCameraOn) {
            this.updateStatus('❌ Сначала запустите камеру', 'error');
            return;
        }

        try {
            this.updateStatus('🔍 Анализируем позу...', 'loading');
            this.capturePoseBtn.disabled = true;

            // Захватываем кадр
            this.ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
            const imageData = this.canvas.toDataURL('image/jpeg', 0.8);

            // Отправляем на сервер
            const result = await this.sendToServer(imageData);
            this.displayResults(result);
            this.updateStatus('✅ Поза определена успешно!', 'success');

        } catch (error) {
            console.error('Detection error:', error);
            this.updateStatus('❌ Ошибка определения: ' + error.message, 'error');
        } finally {
            this.capturePoseBtn.disabled = false;
        }
    }

    handleFileUpload(event) {
        const file = event.target.files[0];
        if (!file) return;

        if (!file.type.match('image.*')) {
            this.updateStatus('❌ Пожалуйста, выберите файл изображения', 'error');
            return;
        }

        const reader = new FileReader();

        reader.onload = async (e) => {
            try {
                this.updateStatus('🔍 Анализируем загруженное изображение...', 'loading');
                const imageData = e.target.result;
                const result = await this.sendToServer(imageData);
                this.displayResults(result);
                this.updateStatus('✅ Анализ завершен!', 'success');
            } catch (error) {
                this.updateStatus('❌ Ошибка анализа: ' + error.message, 'error');
            }
        };

        reader.readAsDataURL(file);
    }

    async sendToServer(imageData) {
        const response = await fetch('/detect', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ image: imageData })
        });

        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }

        const result = await response.json();

        if (!result.success) {
            throw new Error(result.error || 'Detection failed');
        }

        return result;
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
        this.updateStatus('⏹️ Камера остановлена', 'loading');
    }

    displayResults(result) {
        this.resultsSection.style.display = 'block';

        // Статистика
        this.statsDiv.innerHTML = `
            <p>🎯 Тип позы: <strong>${this.getPoseTypeName(result.pose_type)}</strong></p>
            <p>📍 Обнаружено точек: <strong>${result.landmarks.length}</strong></p>
            <p>🔗 Соединений: <strong>${result.connections ? result.connections.length : 0}</strong></p>
        `;

        // Аннотированное изображение
        if (result.annotated_image) {
            this.annotatedImage.src = result.annotated_image;
            this.annotatedImage.style.display = 'block';
        } else {
            this.annotatedImage.style.display = 'none';
        }

        // Список landmarks
        if (result.landmarks && result.landmarks.length > 0) {
            this.landmarksList.innerHTML = result.landmarks
                .map(landmark => `
                    <div class="landmark-item">
                        <div>Точка ${landmark.index}</div>
                        <div>X: ${landmark.x.toFixed(2)}</div>
                        <div>Y: ${landmark.y.toFixed(2)}</div>
                        <div>Видимость: ${Math.round(landmark.visibility * 100)}%</div>
                    </div>
                `)
                .join('');
        } else {
            this.landmarksList.innerHTML = '<p>Точки не обнаружены</p>';
        }

        // Прокрутка к результатам
        this.resultsSection.scrollIntoView({ behavior: 'smooth' });
    }

    getPoseTypeName(poseType) {
        const names = {
            'hands_up': '🙌 Руки вверх',
            'left_hand_up': '👈 Левая рука вверх',
            'right_hand_up': '👉 Правая рука вверх',
            'standing': '🧍 Стоя',
            'unknown': '❓ Неизвестно'
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

// Инициализация при загрузке страницы
document.addEventListener('DOMContentLoaded', () => {
    new PoseDetectionApp();
});