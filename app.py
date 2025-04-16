import os
import logging
import base64
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
from face_analyzer import FaceShapeAnalyzer
from feather_integration import TelegramBotAPI

# Configure Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "default_secret_key")

# Initialize face analyzer and Telegram Bot API client
face_analyzer = FaceShapeAnalyzer()
telegram_bot = TelegramBotAPI()

@app.route('/')
def index():
    """Render the main page of the application."""
    return render_template('index.html')

@app.route('/ping')
def ping():
    """Simple endpoint to check if the application is running."""
    return jsonify({"status": "ok", "message": "Service is running"}), 200

@app.route('/analyze', methods=['POST'])
def analyze_face():
    """Analyze the uploaded face image and return the results."""
    try:
        # Get the image from the request
        if 'image' not in request.files:
            if 'image_data' not in request.form:
                return jsonify({'error': 'No image provided'}), 400
            
            # Handle base64 encoded image
            image_b64 = request.form['image_data'].split(',')[1]
            image_data = base64.b64decode(image_b64)
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            # Handle regular file upload
            file = request.files['image']
            nparr = np.frombuffer(file.read(), np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Analyze the face
        result = face_analyzer.analyze(image)
        
        if not result['success']:
            return jsonify({'error': result['message']}), 400
        
        # Добавляем возможность отправки результатов через Telegram
        if telegram_bot.is_configured() and 'telegram_chat_id' in request.form:
            # Получаем ID чата из запроса
            chat_id = request.form['telegram_chat_id']
            
            # Отправляем результаты анализа в Telegram
            telegram_response = telegram_bot.send_analysis_result(
                chat_id=chat_id,
                face_shape=result['face_shape'],
                description=result['description'],
                confidence=result['confidence'],
                image_data=result['image_with_landmarks']
            )
            
            # Добавляем информацию об отправке в результат
            result['telegram_message_sent'] = telegram_response.get('success', False)
            
            # Отправляем рекомендации по стилю
            recommendations_response = telegram_bot.send_recommendations(
                chat_id=chat_id,
                face_shape=result['face_shape']
            )
            
            # Добавляем информацию об отправке рекомендаций в результат
            result['telegram_recommendations_sent'] = recommendations_response.get('success', False)
        
        return jsonify(result)
    
    except Exception as e:
        logging.error(f"Error analyzing face: {str(e)}")
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500
