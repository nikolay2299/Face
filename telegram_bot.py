#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import logging
import json
import requests
import base64
import time
from typing import Dict, Any, Optional, List, Union
from io import BytesIO
import tempfile

import cv2
import numpy as np

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Получаем токен из переменных окружения
TOKEN = os.environ.get("BOT_FEATHER_TOKEN")
if not TOKEN:
    logger.error("Не найден токен Telegram Bot API. Установите переменную окружения BOT_FEATHER_TOKEN.")
    sys.exit(1)

# URL Telegram Bot API
API_URL = f"https://api.telegram.org/bot{TOKEN}"

# URL для анализа лица (API нашего приложения)
FACE_ANALYZER_URL = "http://localhost:5000/analyze"  # gunicorn сервер работает на порту 5000

# Настройки бота
HELP_MESSAGE = """
🤖 *Анализатор формы лица Бот*

Этот бот поможет определить форму вашего лица и предоставит рекомендации по стилю, прическам и аксессуарам.

*Как пользоваться:*
1. Отправьте фотографию своего лица
2. Подождите пока бот проанализирует ваше фото
3. Получите результат с определением формы лица и рекомендациями

*Рекомендации для лучшего результата:*
• Используйте фото с хорошим освещением
• Смотрите прямо в камеру
• Уберите волосы от лица
• Держите нейтральное выражение лица

*Команды:*
/start - Начать работу с ботом
/help - Показать это сообщение помощи
/info - Информация о боте
"""

INFO_MESSAGE = """
📊 *Информация о боте*

Этот бот анализирует форму лица на основе технологий компьютерного зрения:
• Использует MediaPipe для определения 468 ключевых точек лица
• Анализирует соотношения между различными частями лица
• Определяет 7 основных форм лица (овальное, круглое, квадратное, сердцевидное, ромбовидное, продолговатое, треугольное)
• Предоставляет персонализированные рекомендации по стилю

Версия: 1.1.0
Создан на платформе Replit
"""

START_MESSAGE = """
👋 *Привет! Я бот для анализа формы лица.*

Я могу определить форму вашего лица и дать рекомендации по прическам, макияжу и аксессуарам, которые подойдут именно вам.

Просто отправьте мне свою фотографию, и я проведу анализ. Для лучшего результата убедитесь, что ваше лицо хорошо видно, свет равномерный, а волосы не закрывают контур лица.

Отправьте /help для получения более подробной информации.
"""

WAIT_MESSAGE = "⏳ Анализирую вашу фотографию... Пожалуйста, подождите."

ERROR_MESSAGE = """
😕 К сожалению, не удалось проанализировать эту фотографию.

Пожалуйста, убедитесь, что:
• На фото четко видно лицо
• Хорошее освещение
• Нет сильных теней
• Лицо не закрыто волосами или другими предметами

Попробуйте сделать новое фото и отправить его снова.
"""

def send_message(chat_id: Union[int, str], text: str, parse_mode: str = "Markdown") -> Dict[str, Any]:
    """Отправляет текстовое сообщение пользователю."""
    url = f"{API_URL}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": parse_mode
    }
    
    response = requests.post(url, json=payload)
    return response.json()

def send_photo(chat_id: Union[int, str], photo, caption: str = None) -> Dict[str, Any]:
    """Отправляет фото пользователю."""
    url = f"{API_URL}/sendPhoto"
    
    files = None
    data = {"chat_id": chat_id}
    
    if caption:
        data["caption"] = caption
    
    try:
        # Если фото передано как путь к файлу
        if isinstance(photo, str) and os.path.isfile(photo):
            files = {"photo": open(photo, "rb")}
            response = requests.post(url, data=data, files=files)
        # Если фото передано как байты
        elif isinstance(photo, bytes):
            photo_io = BytesIO(photo)
            files = {"photo": ("photo.jpg", photo_io, "image/jpeg")}
            response = requests.post(url, data=data, files=files)
        # Если фото передано как BytesIO
        elif isinstance(photo, BytesIO):
            photo.seek(0)
            files = {"photo": ("photo.jpg", photo, "image/jpeg")}
            response = requests.post(url, data=data, files=files)
        # Если это URL или file_id
        else:
            data["photo"] = photo
            response = requests.post(url, data=data)
        
        return response.json()
    except Exception as e:
        logger.error(f"Ошибка отправки фото: {str(e)}")
        return {"ok": False, "error": str(e)}

def get_file_path(file_id: str) -> Optional[str]:
    """Получает путь к файлу в Telegram."""
    url = f"{API_URL}/getFile"
    response = requests.get(url, params={"file_id": file_id})
    result = response.json()
    
    if result.get("ok") and "result" in result:
        return result["result"]["file_path"]
    return None

def download_file(file_path: str) -> Optional[bytes]:
    """Скачивает файл из Telegram."""
    url = f"https://api.telegram.org/file/bot{TOKEN}/{file_path}"
    response = requests.get(url)
    
    if response.status_code == 200:
        return response.content
    return None

def process_photo(file_id: str, chat_id: Union[int, str]) -> None:
    """Обрабатывает фотографию и отправляет результаты анализа."""
    try:
        # Получаем путь к файлу
        file_path = get_file_path(file_id)
        if not file_path:
            send_message(chat_id, "Не удалось получить доступ к фотографии. Пожалуйста, попробуйте другое фото.")
            return
        
        # Скачиваем файл
        photo_data = download_file(file_path)
        if not photo_data:
            send_message(chat_id, "Не удалось скачать фотографию. Пожалуйста, попробуйте другое фото.")
            return
        
        # Предварительно уведомляем пользователя, что начался анализ
        send_message(chat_id, WAIT_MESSAGE)
        
        # Преобразуем изображение для анализа
        nparr = np.frombuffer(photo_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Сохраняем изображение во временный файл
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            temp_file_path = temp_file.name
            temp_file.write(photo_data)
        
        # Отправляем изображение на анализ в наше основное приложение
        with open(temp_file_path, 'rb') as f:
            files = {'image': f}
            data = {'telegram_chat_id': str(chat_id)}
            response = requests.post(FACE_ANALYZER_URL, files=files, data=data)
        
        # Удаляем временный файл
        os.unlink(temp_file_path)
        
        # Обрабатываем ответ
        if response.status_code == 200:
            result = response.json()
            
            # Результаты обрабатываются в API анализатора лица и отправляются напрямую через Telegram API
            # Так что здесь нам не нужно отправлять ответ пользователю
            # Но на всякий случай проверим, был ли отправлен ответ
            if not result.get('telegram_message_sent', False):
                # Если сообщение не было отправлено напрямую из API, отправим базовое сообщение
                basic_result = (
                    f"🔍 *Результаты анализа лица*\n\n"
                    f"👤 *Форма лица*: {result['face_shape'].upper()}\n\n"
                    f"📝 *Описание*: {result['description']}\n\n"
                    f"🎯 *Уверенность*: {int(result['confidence'] * 100)}%"
                )
                send_message(chat_id, basic_result)
        else:
            # Если что-то пошло не так, отправляем сообщение об ошибке
            send_message(chat_id, ERROR_MESSAGE)
    
    except Exception as e:
        logger.error(f"Ошибка при обработке фотографии: {str(e)}")
        send_message(chat_id, f"Произошла ошибка при анализе: {str(e)}")

def process_update(update: Dict[str, Any]) -> None:
    """Обрабатывает обновление от Telegram."""
    # Проверяем, что сообщение содержит текст или фото
    if 'message' not in update:
        return
    
    message = update['message']
    chat_id = message['chat']['id']
    
    # Обрабатываем команды
    if 'text' in message:
        if message['text'] == '/start':
            send_message(chat_id, START_MESSAGE)
        elif message['text'] == '/help':
            send_message(chat_id, HELP_MESSAGE)
        elif message['text'] == '/info':
            send_message(chat_id, INFO_MESSAGE)
        else:
            send_message(chat_id, "Отправьте фотографию своего лица для анализа.")
    
    # Обрабатываем фотографии
    elif 'photo' in message:
        # Берем самую большую версию фото (последний элемент в массиве)
        file_id = message['photo'][-1]['file_id']
        process_photo(file_id, chat_id)

def get_updates(offset: int = 0) -> List[Dict[str, Any]]:
    """Получает обновления от Telegram Bot API."""
    url = f"{API_URL}/getUpdates"
    params = {
        "offset": offset,
        "timeout": 10  # Уменьшаем таймаут для более частых проверок
    }
    
    try:
        response = requests.get(url, params=params, timeout=15)  # Добавляем таймаут для запроса
        result = response.json()
        if result.get("ok") and "result" in result:
            return result["result"]
        elif not result.get("ok"):
            logger.error(f"API вернул ошибку: {result.get('description', 'Неизвестная ошибка')}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Ошибка сети при получении обновлений: {str(e)}")
    except Exception as e:
        logger.error(f"Неожиданная ошибка при получении обновлений: {str(e)}")
    
    return []

def set_webhook(webhook_url: str) -> bool:
    """Устанавливает вебхук для бота."""
    url = f"{API_URL}/setWebhook"
    params = {"url": webhook_url}
    
    try:
        response = requests.get(url, params=params)
        result = response.json()
        if result.get("ok", False):
            logger.info(f"Вебхук успешно установлен: {webhook_url}")
            return True
        else:
            logger.error(f"Ошибка установки вебхука: {result}")
    except Exception as e:
        logger.error(f"Ошибка при установке вебхука: {str(e)}")
    
    return False

def polling_loop() -> None:
    """Запускает цикл опроса обновлений от Telegram."""
    logger.info("Запущен цикл опроса обновлений")
    
    # Проверяем соединение с Telegram API
    try:
        bot_info = requests.get(f"{API_URL}/getMe", timeout=10).json()
        if bot_info.get("ok", False):
            bot_username = bot_info["result"]["username"]
            logger.info(f"Успешное соединение с API. Бот @{bot_username} активен")
        else:
            logger.error(f"Ошибка при проверке API: {bot_info.get('description', 'Неизвестная ошибка')}")
    except Exception as e:
        logger.error(f"Не удалось проверить соединение с Telegram API: {str(e)}")
    
    offset = 0
    consecutive_errors = 0
    max_consecutive_errors = 5
    
    while True:
        try:
            # Получаем обновления
            updates = get_updates(offset)
            
            # Если успешно получили обновления, сбрасываем счетчик ошибок
            if updates is not None:
                consecutive_errors = 0
            
            # Обрабатываем каждое обновление
            for update in updates:
                try:
                    process_update(update)
                    # Увеличиваем смещение для следующего запроса
                    offset = update['update_id'] + 1
                except Exception as e:
                    logger.error(f"Ошибка при обработке обновления {update.get('update_id')}: {str(e)}")
            
            # Небольшая пауза между запросами
            time.sleep(0.5)
            
        except KeyboardInterrupt:
            logger.info("Бот остановлен вручную")
            break
            
        except Exception as e:
            # Увеличиваем счетчик последовательных ошибок
            consecutive_errors += 1
            logger.error(f"Ошибка в цикле опроса ({consecutive_errors}/{max_consecutive_errors}): {str(e)}")
            
            # Если слишком много ошибок подряд, делаем более длительную паузу
            if consecutive_errors >= max_consecutive_errors:
                logger.warning(f"Слишком много ошибок подряд. Пауза на 30 секунд...")
                time.sleep(30)
                consecutive_errors = 0
            else:
                # Короткая пауза перед повторной попыткой
                time.sleep(5)

if __name__ == "__main__":
    # Выводим информацию о запуске бота
    bot_info = requests.get(f"{API_URL}/getMe").json()
    if bot_info.get("ok", False):
        bot_username = bot_info["result"]["username"]
        logger.info(f"Бот @{bot_username} запущен")
    else:
        logger.error("Не удалось получить информацию о боте. Проверьте токен.")
        sys.exit(1)
    
    # Запускаем цикл опроса обновлений
    polling_loop()