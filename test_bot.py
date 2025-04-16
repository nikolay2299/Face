#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import logging
import requests
import time

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
    exit(1)

# URL Telegram Bot API
API_URL = f"https://api.telegram.org/bot{TOKEN}"

def test_bot_api():
    """Тестирует подключение к Telegram Bot API."""
    try:
        logger.info("Проверка соединения с Telegram Bot API...")
        response = requests.get(f"{API_URL}/getMe", timeout=10)
        result = response.json()
        
        if result.get("ok", False):
            bot_info = result["result"]
            logger.info(f"✅ Успешное соединение с API!")
            logger.info(f"Информация о боте:")
            logger.info(f"  - ID: {bot_info['id']}")
            logger.info(f"  - Имя: {bot_info['first_name']}")
            logger.info(f"  - Username: @{bot_info['username']}")
            logger.info(f"  - Может принимать запросы: {bot_info.get('can_join_groups', False)}")
            logger.info(f"  - Может читать сообщения группы: {bot_info.get('can_read_all_group_messages', False)}")
            
            # Проверяем настройки вебхука
            logger.info("Проверка настроек вебхука...")
            webhook_info = requests.get(f"{API_URL}/getWebhookInfo", timeout=10).json()
            
            if webhook_info.get("ok", False):
                webhook_url = webhook_info["result"].get("url", "")
                if webhook_url:
                    logger.info(f"⚠️ Установлен вебхук: {webhook_url}")
                    logger.info("Для работы в режиме long-polling рекомендуется удалить вебхук")
                    logger.info(f"Команда для удаления вебхука: {API_URL}/deleteWebhook")
                else:
                    logger.info("✅ Вебхук не установлен. Бот готов к работе в режиме long-polling")
            else:
                logger.error(f"❌ Ошибка при получении информации о вебхуке: {webhook_info.get('description', 'Неизвестная ошибка')}")
            
            return True
        else:
            logger.error(f"❌ Ошибка API: {result.get('description', 'Неизвестная ошибка')}")
            return False
            
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Ошибка сети при подключении к API: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"❌ Неожиданная ошибка: {str(e)}")
        return False

def send_test_message():
    """Отправляет тестовое сообщение самому себе (боту)."""
    try:
        # Получаем ID бота
        bot_info = requests.get(f"{API_URL}/getMe", timeout=10).json()
        if not bot_info.get("ok", False):
            logger.error("❌ Не удалось получить информацию о боте")
            return False
        
        bot_id = bot_info["result"]["id"]
        
        # Пробуем отправить сообщение самому боту
        # Это не всегда работает, так как боты не могут начинать диалог с другими пользователями
        # Но можно попробовать
        logger.info(f"Попытка отправить тестовое сообщение боту (ID: {bot_id})...")
        
        response = requests.post(
            f"{API_URL}/sendMessage",
            json={
                "chat_id": bot_id,
                "text": "Это тестовое сообщение. Бот работает корректно!"
            },
            timeout=10
        )
        
        result = response.json()
        if result.get("ok", False):
            logger.info("✅ Тестовое сообщение успешно отправлено!")
            return True
        else:
            logger.warning(f"⚠️ Не удалось отправить тестовое сообщение: {result.get('description', 'Неизвестная ошибка')}")
            logger.info("Это нормально, так как боты не могут начинать диалог с пользователями")
            return False
            
    except Exception as e:
        logger.error(f"❌ Ошибка при отправке тестового сообщения: {str(e)}")
        return False

if __name__ == "__main__":
    logger.info("Начало тестирования Telegram бота")
    
    if test_bot_api():
        logger.info("✅ Тест API прошел успешно")
        send_test_message()
    else:
        logger.error("❌ Тест API не пройден. Проверьте токен и соединение")
    
    logger.info("Тестирование завершено")
    logger.info("")
    logger.info("Инструкции по запуску бота:")
    logger.info("1. Используйте команду 'python run_telegram_bot.py' для запуска в режиме long-polling")
    logger.info("2. Бот доступен по адресу t.me/faceaishape_bot")
    logger.info("3. Отправьте боту фотографию лица для получения анализа формы лица")