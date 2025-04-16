#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import logging
import threading
import requests
from telegram_bot import polling_loop, set_webhook

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_polling():
    """Запускает режим long polling для бота."""
    logger.info("Запуск бота в режиме long polling...")
    
    # Проверяем, включен ли режим вебхука, и если да, то отключаем его
    TOKEN = os.environ.get("BOT_FEATHER_TOKEN")
    API_URL = f"https://api.telegram.org/bot{TOKEN}"
    
    try:
        webhook_info = requests.get(f"{API_URL}/getWebhookInfo", timeout=10).json()
        if webhook_info.get("ok") and webhook_info["result"].get("url"):
            logger.info(f"Отключение вебхука: {webhook_info['result']['url']}")
            delete_result = requests.get(f"{API_URL}/deleteWebhook", timeout=10).json()
            if delete_result.get("ok"):
                logger.info("Вебхук успешно отключен")
            else:
                logger.error(f"Ошибка при отключении вебхука: {delete_result.get('description')}")
    except Exception as e:
        logger.error(f"Ошибка при проверке вебхука: {str(e)}")
    
    # Запускаем бота в режиме long polling
    try:
        polling_loop()
    except KeyboardInterrupt:
        logger.info("Бот остановлен вручную")
    except Exception as e:
        logger.error(f"Критическая ошибка в работе бота: {str(e)}")
        
    logger.info("Бот остановлен")

def run_webhook(webhook_url=None, port=8080):
    """Запускает режим webhook для бота."""
    from telegram_webhook import create_app
    
    if not webhook_url:
        # Определяем URL вебхука на основе домена Replit
        replit_domain = os.environ.get("REPL_SLUG")
        webhook_url = f"https://{replit_domain}.repl.co/webhook"
    
    logger.info(f"Запуск бота в режиме webhook на {webhook_url}...")
    
    # Устанавливаем вебхук
    success = set_webhook(webhook_url)
    if not success:
        logger.error("Не удалось установить вебхук. Завершение работы")
        return
    
    # Создаем Flask приложение для обработки вебхуков
    app = create_app()
    
    # Запускаем сервер
    logger.info(f"Запуск сервера на порту {port}")
    app.run(host="0.0.0.0", port=port)

if __name__ == "__main__":
    # По умолчанию запускаем режим long polling
    mode = os.environ.get("BOT_MODE", "polling").lower()
    
    if mode == "webhook":
        webhook_url = os.environ.get("WEBHOOK_URL")
        webhook_port = int(os.environ.get("WEBHOOK_PORT", 8080))
        run_webhook(webhook_url, webhook_port)
    else:
        run_polling()