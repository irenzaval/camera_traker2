from flask import Flask, render_template, request, jsonify, send_from_directory
import cv2
import mediapipe as mp
import numpy as np
import base64
import io
import os
from PIL import Image
import logging
import math

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


class SimpleBodyTracker:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,  # Изменено на True для обработки изображений
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def calculate_distance(self, point1, point2, image_shape):
        """Вычисляет расстояние между двумя точками"""
        h, w = image_shape[:2]
        x1, y1 = int(point1.x * w), int(point1.y * h)
        x2, y2 = int(point2.x * w), int(point2.y * h)
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return distance, (x1, y1), (x2, y2)

    def process_image(self, image_data):
        """Обрабатывает одно изображение и возвращает результаты"""
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
                'distances': [],
                'annotated_image': None,
                'stats': {}
            }

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # Извлекаем landmarks
                landmarks_data = []
                for idx, landmark in enumerate(landmarks):
                    if landmark.visibility > 0.3:  # Более низкий порог для веб-версии
                        landmarks_data.append({
                            'x': float(landmark.x),
                            'y': float(landmark.y),
                            'z': float(landmark.z),
                            'visibility': float(landmark.visibility),
                            'index': idx,
                            'name': self.get_landmark_name(idx)
                        })

                response_data['landmarks'] = landmarks_data

                # Вычисляем расстояния
                distances_data = self.calculate_body_distances(landmarks, image_np.shape)
                response_data['distances'] = distances_data

                # Создаем аннотированное изображение
                annotated_image = self.create_annotated_image(image_np.copy(), results, distances_data)
                response_data['annotated_image'] = self.image_to_base64(annotated_image)

                # Статистика
                response_data['stats'] = {
                    'total_landmarks': len(landmarks_data),
                    'total_distances': len(distances_data),
                    'detection_quality': self.calculate_detection_quality(landmarks_data)
                }

            return response_data

        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            return {'success': False, 'error': str(e)}

    def calculate_body_distances(self, landmarks, image_shape):
        """Вычисляет расстояния между ключевыми точками тела"""
        distances = []

        # Важные пары точек для измерения (как в вашем коде)
        key_pairs = [
            (self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.RIGHT_SHOULDER, "Shoulders"),
            (self.mp_pose.PoseLandmark.LEFT_WRIST, self.mp_pose.PoseLandmark.RIGHT_WRIST, "Hands"),
            (self.mp_pose.PoseLandmark.LEFT_HIP, self.mp_pose.PoseLandmark.RIGHT_HIP, "Hips"),
            (self.mp_pose.PoseLandmark.LEFT_ANKLE, self.mp_pose.PoseLandmark.RIGHT_ANKLE, "Feet"),
            # Добавляем дополнительные расстояния
            (self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.LEFT_ELBOW, "Left Upper Arm"),
            (self.mp_pose.PoseLandmark.LEFT_ELBOW, self.mp_pose.PoseLandmark.LEFT_WRIST, "Left Lower Arm"),
            (self.mp_pose.PoseLandmark.RIGHT_SHOULDER, self.mp_pose.PoseLandmark.RIGHT_ELBOW, "Right Upper Arm"),
            (self.mp_pose.PoseLandmark.RIGHT_ELBOW, self.mp_pose.PoseLandmark.RIGHT_WRIST, "Right Lower Arm"),
        ]

        for point1, point2, label in key_pairs:
            if (point1.value < len(landmarks) and point2.value < len(landmarks) and
                    landmarks[point1.value].visibility > 0.3 and landmarks[point2.value].visibility > 0.3):
                distance, coord1, coord2 = self.calculate_distance(
                    landmarks[point1.value], landmarks[point2.value], image_shape
                )

                distances.append({
                    'label': label,
                    'distance': float(distance),
                    'point1': point1.value,
                    'point2': point2.value,
                    'point1_name': self.get_landmark_name(point1.value),
                    'point2_name': self.get_landmark_name(point2.value),
                    'coordinates': {
                        'point1': {'x': coord1[0], 'y': coord1[1]},
                        'point2': {'x': coord2[0], 'y': coord2[1]}
                    }
                })

        return distances

    def create_annotated_image(self, image, results, distances):
        """Создает изображение с landmarks и расстояниями"""
        # Рисуем landmarks тела (как в вашем коде)
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )

        # Рисуем линии расстояний
        for distance_info in distances:
            coord1 = distance_info['coordinates']['point1']
            coord2 = distance_info['coordinates']['point2']

            # Рисуем линию
            cv2.line(image,
                     (coord1['x'], coord1['y']),
                     (coord2['x'], coord2['y']),
                     (255, 0, 255), 2)

            # Подпись расстояния
            mid_x = (coord1['x'] + coord2['x']) // 2
            mid_y = (coord1['y'] + coord2['y']) // 2

            cv2.putText(image, f"{distance_info['distance']:.0f}px",
                        (mid_x, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Общая информация
        cv2.putText(image, f"Landmarks: {len([l for l in results.pose_landmarks.landmark if l.visibility > 0.3])}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(image, f"Distances: {len(distances)}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return image

    def get_landmark_name(self, index):
        """Возвращает название для каждой ключевой точки"""
        names = {
            0: "Nose", 1: "Left Eye Inner", 2: "Left Eye", 3: "Left Eye Outer",
            4: "Right Eye Inner", 5: "Right Eye", 6: "Right Eye Outer",
            7: "Left Ear", 8: "Right Ear", 9: "Mouth Left", 10: "Mouth Right",
            11: "Left Shoulder", 12: "Right Shoulder", 13: "Left Elbow",
            14: "Right Elbow", 15: "Left Wrist", 16: "Right Wrist",
            17: "Left Pinky", 18: "Right Pinky", 19: "Left Index",
            20: "Right Index", 21: "Left Thumb", 22: "Right Thumb",
            23: "Left Hip", 24: "Right Hip", 25: "Left Knee", 26: "Right Knee",
            27: "Left Ankle", 28: "Right Ankle", 29: "Left Heel",
            30: "Right Heel", 31: "Left Foot Index", 32: "Right Foot Index"
        }
        return names.get(index, f"Point_{index}")

    def calculate_detection_quality(self, landmarks):
        """Рассчитывает качество обнаружения"""
        if not landmarks:
            return "poor"

        visible_count = len(landmarks)
        high_confidence_count = len([l for l in landmarks if l['visibility'] > 0.7])

        ratio = high_confidence_count / visible_count if visible_count > 0 else 0

        if ratio > 0.8:
            return "excellent"
        elif ratio > 0.6:
            return "good"
        elif ratio > 0.4:
            return "fair"
        else:
            return "poor"

    def image_to_base64(self, image):
        """Конвертирует изображение в base64"""
        try:
            _, buffer = cv2.imencode('.jpg', image)
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            return f"data:image/jpeg;base64,{image_base64}"
        except Exception as e:
            logger.error(f"Error converting image: {str(e)}")
            return None


# Инициализация трекера
tracker = SimpleBodyTracker()


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
        result = tracker.process_image(image_data)

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

    print("🚀 Body Pose Detection Web App Started!")
    print("📍 Access: http://localhost:5000")
    print("📸 Upload photos to analyze body pose and distances")

    app.run(host='0.0.0.0', port=5000, debug=True)