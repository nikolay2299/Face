#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import logging
import subprocess
import signal
import requests

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# URL для проверки статуса веб-приложения
STATUS_URL = "http://localhost:5000/ping"

# Получаем токен из переменных окружения
TOKEN = os.environ.get("BOT_FEATHER_TOKEN")
if not TOKEN:
    logger.error("Не найден токен Telegram Bot API. Установите переменную окружения BOT_FEATHER_TOKEN.")
    sys.exit(1)

# URL Telegram Bot API
API_URL = f"https://api.telegram.org/bot{TOKEN}"

def check_flask_status():
    """Проверяет статус Flask приложения."""
    try:
        response = requests.get(STATUS_URL, timeout=5)
        return response.status_code == 200
    except:
        return False

def check_bot_status():
    """Проверяет статус соединения бота с Telegram API."""
    try:
        response = requests.get(f"{API_URL}/getMe", timeout=10)
        return response.json().get("ok", False)
    except:
        return False

def start_bot_process():
    """Запускает процесс бота."""
    try:
        # Запускаем бота как отдельный процесс
        process = subprocess.Popen(
            ["python", "bot_server.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        logger.info(f"Бот запущен с PID {process.pid}")
        return process
    except Exception as e:
        logger.error(f"Не удалось запустить бот: {str(e)}")
        return None

def main():
    """Основная функция для запуска и мониторинга бота."""
    logger.info("Запуск мониторинга бота...")
    
    # Проверяем статус Flask приложения
    if not check_flask_status():
        logger.warning("Flask приложение не доступно. Некоторые функции бота могут не работать.")
    
    # Запускаем процесс бота
    bot_process = start_bot_process()
    if not bot_process:
        return
    
    try:
        # Ждем некоторое время для запуска бота
        time.sleep(5)
        
        # Проверяем, работает ли бот
        if not check_bot_status():
            logger.error("Бот не смог подключиться к Telegram API.")
            bot_process.terminate()
            return
        
        # Основной цикл мониторинга
        consecutive_failures = 0
        max_consecutive_failures = 3
        
        while True:
            # Проверяем статус процесса
            if bot_process.poll() is not None:
                logger.warning(f"Процесс бота завершился с кодом {bot_process.returncode}. Перезапуск...")
                bot_process = start_bot_process()
                if not bot_process:
                    break
                time.sleep(5)
                consecutive_failures = 0
            
            # Проверяем соединение с Telegram API
            if not check_bot_status():
                consecutive_failures += 1
                logger.warning(f"Неудачная проверка соединения ({consecutive_failures}/{max_consecutive_failures})")
                
                if consecutive_failures >= max_consecutive_failures:
                    logger.error("Слишком много ошибок подряд. Перезапуск бота...")
                    bot_process.terminate()
                    time.sleep(2)
                    bot_process = start_bot_process()
                    if not bot_process:
                        break
                    time.sleep(5)
                    consecutive_failures = 0
            else:
                consecutive_failures = 0
            
            # Пауза между проверками
            time.sleep(60)
    
    except KeyboardInterrupt:
        logger.info("Получен сигнал завершения. Остановка бота...")
    finally:
        # Завершаем процесс бота при выходе
        if bot_process and bot_process.poll() is None:
            logger.info("Остановка процесса бота...")
            bot_process.terminate()
            try:
                bot_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("Процесс не завершился вовремя. Принудительное завершение...")
                bot_process.kill()

if __name__ == "__main__":
    main()