#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import requests
import logging
from typing import Dict, Any, Optional, List, Union

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TelegramBotAPI:
    """Класс для интеграции с Telegram Bot API для отправки результатов анализа."""
    
    def __init__(self):
        """Инициализация клиента Telegram Bot API с токеном из переменных окружения."""
        self.token = os.environ.get("BOT_FEATHER_TOKEN")
        if self.token:
            self.api_url = f"https://api.telegram.org/bot{self.token}"
        else:
            logger.warning("Токен Telegram Bot не найден в переменных окружения. Интеграция не будет работать.")
            self.api_url = None
    
    def is_configured(self) -> bool:
        """Проверка, настроен ли API клиент."""
        return self.token is not None and self.api_url is not None
    
    def send_analysis_result(self, chat_id: str, face_shape: str, 
                          description: str, confidence: float, 
                          image_data: str = None) -> Dict[str, Any]:
        """
        Отправляет результаты анализа формы лица пользователю в Telegram.
        
        Args:
            chat_id: ID чата для отправки сообщения
            face_shape: Определенная форма лица
            description: Описание формы лица
            confidence: Уверенность в определении (от 0 до 1)
            image_data: Base64-закодированное изображение с разметкой лица (опционально)
            
        Returns:
            Словарь с результатом отправки сообщения
        """
        if not self.is_configured():
            logger.error("API клиент не настроен. Невозможно отправить результат анализа.")
            return {"ok": False, "error": "API клиент не настроен"}
        
        # Формируем сообщение с результатом анализа
        message_text = (
            f"🔍 *Результаты анализа лица*\n\n"
            f"👤 *Форма лица*: {face_shape.upper()}\n\n"
            f"📝 *Описание*: {description}\n\n"
            f"🎯 *Уверенность*: {int(confidence * 100)}%"
        )
        
        try:
            # Отправляем текстовое сообщение
            text_response = requests.post(
                f"{self.api_url}/sendMessage",
                json={
                    "chat_id": chat_id,
                    "text": message_text,
                    "parse_mode": "Markdown"
                },
                timeout=10
            )
            
            text_result = text_response.json()
            
            # Если есть изображение с разметкой, отправляем его
            if image_data and text_result.get("ok"):
                # Проверяем, начинается ли строка с data:image
                if image_data.startswith("data:image"):
                    # Извлекаем только данные base64 (удаляем префикс data:image/jpeg;base64,)
                    import base64
                    from io import BytesIO
                    
                    # Получаем только закодированные данные
                    img_data = image_data.split(',')[1]
                    img_binary = base64.b64decode(img_data)
                    
                    # Создаем объект BytesIO для отправки
                    img_io = BytesIO(img_binary)
                    img_io.name = 'face_analysis.jpg'
                    
                    # Отправляем изображение
                    files = {"photo": img_io}
                    photo_response = requests.post(
                        f"{self.api_url}/sendPhoto",
                        data={"chat_id": chat_id, "caption": "Анализ лица с разметкой ключевых точек"},
                        files=files,
                        timeout=10
                    )
                    
                    photo_result = photo_response.json()
                    if not photo_result.get("ok"):
                        logger.error(f"Ошибка при отправке изображения: {photo_result}")
            
            # Отправляем рекомендации по стилю
            self.send_recommendations(chat_id, face_shape)
            
            return text_result
            
        except Exception as e:
            logger.error(f"Ошибка при отправке результата анализа: {str(e)}")
            return {"ok": False, "error": str(e)}
    
    def send_recommendations(self, chat_id: str, face_shape: str) -> Dict[str, Any]:
        """
        Отправляет рекомендации по прическам и стилю для данной формы лица.
        
        Args:
            chat_id: ID чата для отправки сообщения
            face_shape: Форма лица
            
        Returns:
            Словарь с результатом отправки сообщения
        """
        if not self.is_configured():
            logger.error("API клиент не настроен. Невозможно отправить рекомендации.")
            return {"ok": False, "error": "API клиент не настроен"}
        
        # Получаем рекомендации для данной формы лица
        recommendations = self._get_recommendations_for_face_shape(face_shape)
        
        # Формируем сообщение с рекомендациями
        message_text = (
            f"✨ *Рекомендации для {face_shape.upper()} формы лица*\n\n"
            f"💇 *Прически*:\n"
        )
        
        # Добавляем рекомендации по прическам
        for haircut in recommendations.get("haircuts", []):
            message_text += f"• {haircut}\n"
        
        message_text += "\n👓 *Очки и аксессуары*:\n"
        
        # Добавляем рекомендации по аксессуарам
        for accessory in recommendations.get("accessories", []):
            message_text += f"• {accessory}\n"
        
        message_text += "\n💄 *Макияж*:\n"
        
        # Добавляем рекомендации по макияжу
        for makeup in recommendations.get("makeup", []):
            message_text += f"• {makeup}\n"
        
        try:
            # Отправляем сообщение
            response = requests.post(
                f"{self.api_url}/sendMessage",
                json={
                    "chat_id": chat_id,
                    "text": message_text,
                    "parse_mode": "Markdown"
                },
                timeout=10
            )
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Ошибка при отправке рекомендаций: {str(e)}")
            return {"ok": False, "error": str(e)}
    
    def _get_recommendations_for_face_shape(self, face_shape: str) -> Dict[str, List[str]]:
        """
        Возвращает рекомендации по прическам и аксессуарам для данной формы лица.
        
        Args:
            face_shape: Форма лица
            
        Returns:
            Словарь с рекомендациями
        """
        recommendations = {
            "oval": {
                "haircuts": [
                    "Практически любые прически и длина волос",
                    "Длинные волосы с легкими волнами",
                    "Короткие стрижки типа пикси",
                    "Средняя длина с многослойными стрижками"
                ],
                "accessories": [
                    "Подходят практически любые формы очков",
                    "Круглые, квадратные или кошачьи очки",
                    "Серьги любой формы и размера"
                ],
                "makeup": [
                    "Не требуется сильная коррекция, используйте легкое контурирование",
                    "Выделяйте глаза или губы по желанию",
                    "Румяна наносите на яблочки щек"
                ]
            },
            "round": {
                "haircuts": [
                    "Асимметричные стрижки для визуального удлинения лица",
                    "Длинные многослойные стрижки",
                    "Прямые или волнистые волосы ниже подбородка",
                    "Избегайте очень коротких и пышных причесок"
                ],
                "accessories": [
                    "Прямоугольные или квадратные очки",
                    "Избегайте круглых оправ, так как они подчеркнут округлость",
                    "Длинные серьги, визуально удлиняющие лицо"
                ],
                "makeup": [
                    "Контурирование скул для придания им более выраженной формы",
                    "Удлиняющее контурирование по линии роста волос и под скулами",
                    "Румяна наносите ниже скул, а не на яблочки щек"
                ]
            },
            "square": {
                "haircuts": [
                    "Мягкие волны и кудри для смягчения углов",
                    "Длинная челка набок",
                    "Объем в области висков",
                    "Многослойные стрижки средней длины"
                ],
                "accessories": [
                    "Круглые или овальные очки для смягчения углов",
                    "Избегайте квадратных и прямоугольных оправ",
                    "Круглые серьги или серьги с мягкими изгибами"
                ],
                "makeup": [
                    "Смягчающее контурирование углов челюсти",
                    "Румяна на яблочках щек для придания мягкости",
                    "Округлые формы в макияже бровей"
                ]
            },
            "heart": {
                "haircuts": [
                    "Объемные прически от средней линии ушей и ниже",
                    "Прически с пробором посередине",
                    "Удлиненный боб или лоб длиной до подбородка",
                    "Многослойные стрижки с акцентом на нижнюю часть лица"
                ],
                "accessories": [
                    "Очки нижней оправой или без оправы",
                    "Очки кошачий глаз, сбалансируют верхнюю и нижнюю части лица",
                    "Объемные серьги, привлекающие внимание к нижней части лица"
                ],
                "makeup": [
                    "Контурирование висков и лба для визуального сужения",
                    "Хайлайтер на подбородок для визуального расширения",
                    "Румяна на средней линии щек"
                ]
            },
            "diamond": {
                "haircuts": [
                    "Прически с объемом в области подбородка",
                    "Длинные многослойные стрижки",
                    "Челки для визуального сокращения высоты лба",
                    "Боб с удлинением к подбородку"
                ],
                "accessories": [
                    "Очки с закругленными краями или овальные",
                    "Очки с верхней оправой или без оправы",
                    "Объемные серьги, привлекающие внимание к нижней части лица"
                ],
                "makeup": [
                    "Контурирование скул для их смягчения",
                    "Румяна на яблочках щек",
                    "Акцент на глаза и губы"
                ]
            },
            "oblong": {
                "haircuts": [
                    "Прически с объемом по бокам",
                    "Многослойные стрижки средней длины",
                    "Прямая или косая челка для визуального уменьшения длины лица",
                    "Избегайте чрезмерно длинных и прямых причесок"
                ],
                "accessories": [
                    "Широкие очки с горизонтальными акцентами",
                    "Круглые очки для смягчения вытянутости",
                    "Короткие, объемные серьги"
                ],
                "makeup": [
                    "Контурирование лба и подбородка для визуального укорочения",
                    "Использование румян на скулах по горизонтали",
                    "Широкие формы в макияже бровей"
                ]
            },
            "triangle": {
                "haircuts": [
                    "Объем в верхней части головы и у висков",
                    "Короткие или средней длины стрижки",
                    "Многослойные прически с объемом на макушке",
                    "Челка для балансировки пропорций"
                ],
                "accessories": [
                    "Очки с акцентом на верхнюю часть оправы",
                    "Оправы кошачий глаз или очки-авиаторы",
                    "Объемные серьги, акцентирующие верхнюю часть лица"
                ],
                "makeup": [
                    "Акцент на глаза и брови",
                    "Контурирование линии челюсти для её визуального сужения",
                    "Румяна на скулах и чуть выше"
                ]
            }
        }
        
        # Возвращаем рекомендации для заданной формы лица
        # Если форма не распознана, возвращаем для овального лица
        return recommendations.get(face_shape.lower(), recommendations["oval"])