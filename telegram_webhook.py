#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import flask
import logging
import requests
import json
from telegram_bot import process_update

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_app():
    """Создает Flask приложение для обработки вебхуков."""
    app = flask.Flask(__name__)
    
    @app.route('/webhook', methods=['POST'])
    def webhook():
        """Обрабатывает входящие вебхуки от Telegram."""
        try:
            update = flask.request.json
            logger.info(f"Получено обновление: {update.get('update_id')}")
            
            # Обрабатываем обновление
            process_update(update)
            
            return {'status': 'success'}, 200
        except Exception as e:
            logger.error(f"Ошибка при обработке вебхука: {str(e)}")
            return {'status': 'error', 'message': str(e)}, 500
    
    @app.route('/ping', methods=['GET'])
    def ping():
        """Проверка работоспособности вебхука."""
        return {'status': 'ok', 'message': 'Webhook server is working!'}, 200
    
    return app

if __name__ == "__main__":
    app = create_app()
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)