from flask import Flask, render_template, request, jsonify, send_from_directory
import cv2
import mediapipe as mp
import numpy as np
import base64
import io
import os
from PIL import Image
import logging

app = Flask(__name__)


# Включим CORS вручную
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response


# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WebBodyPoseDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )

    def process_image(self, image_data):
        try:
            # Декодируем base64 изображение
            if ',' in image_data:
                image_data = image_data.split(',')[1]

            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            image_np = np.array(image)

            # Конвертируем BGR to RGB
            image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

            # Обрабатываем позу
            results = self.pose.process(image_rgb)

            response_data = {
                'success': True,
                'landmarks': [],
                'connections': [],
                'pose_type': 'unknown'
            }

            if results.pose_landmarks:
                # Извлекаем landmarks
                landmarks_data = []
                for idx, landmark in enumerate(results.pose_landmarks.landmark):
                    if landmark.visibility > 0.5:
                        landmarks_data.append({
                            'x': float(landmark.x),
                            'y': float(landmark.y),
                            'z': float(landmark.z),
                            'visibility': float(landmark.visibility),
                            'index': idx
                        })

                response_data['landmarks'] = landmarks_data
                response_data['connections'] = list(self.mp_pose.POSE_CONNECTIONS)
                response_data['pose_type'] = self.classify_pose(results.pose_landmarks.landmark)

                # Создаем изображение с landmarks
                annotated_image = self.draw_landmarks(image_np.copy(), results)
                response_data['annotated_image'] = self.image_to_base64(annotated_image)

            return response_data

        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            return {'success': False, 'error': str(e)}

    def classify_pose(self, landmarks):
        """Определяет тип позы"""
        if len(landmarks) < 25:
            return "unknown"

        # Проверяем поднятые руки
        left_hand_up = landmarks[15].y < landmarks[11].y
        right_hand_up = landmarks[16].y < landmarks[12].y

        if left_hand_up and right_hand_up:
            return "hands_up"
        elif left_hand_up:
            return "left_hand_up"
        elif right_hand_up:
            return "right_hand_up"
        else:
            return "standing"

    def draw_landmarks(self, image, results):
        """Рисует landmarks на изображении"""
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )
        return image

    def image_to_base64(self, image):
        """Конвертирует изображение в base64"""
        try:
            _, buffer = cv2.imencode('.jpg', image)
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            return f"data:image/jpeg;base64,{image_base64}"
        except Exception as e:
            logger.error(f"Error converting image: {str(e)}")
            return None


# Инициализация детектора
detector = WebBodyPoseDetector()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/detect', methods=['POST', 'OPTIONS'])
def detect_pose():
    if request.method == 'OPTIONS':
        return '', 200

    try:
        data = request.json

        if not data or 'image' not in data:
            return jsonify({'success': False, 'error': 'No image data provided'}), 400

        image_data = data['image']
        result = detector.process_image(image_data)

        return jsonify(result)

    except Exception as e:
        logger.error(f"API error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy', 'service': 'Body Pose Detection Web'})


# Статические файлы
@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)


if __name__ == '__main__':
    # Создаем папки если их нет
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)

    app.run(host='0.0.0.0', port=5000, debug=True)