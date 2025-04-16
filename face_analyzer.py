#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import mediapipe as mp
import base64
import logging
from typing import Dict, Any, Tuple, List, Optional

class FaceShapeAnalyzer:
    """Class for analyzing face shapes using MediaPipe and OpenCV."""
    
    def __init__(self):
        """Initialize the face mesh detector from MediaPipe."""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Инициализируем детектор лицевых точек с высокой точностью
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        
        # Ключевые индексы для определения формы лица
        # Эти индексы соответствуют точкам лица в MediaPipe FaceMesh
        self.FACE_OVAL_INDICES = [
            10,  # Подбородок
            152,  # Правый край лица (у уха)
            234,  # Верхняя точка лба
            454,  # Левый край лица (у уха)
        ]
        
        self.CHEEKBONE_INDICES = [
            117,  # Правая скула
            346   # Левая скула
        ]
        
        self.JAW_INDICES = [
            172,  # Правый угол челюсти
            397   # Левый угол челюсти
        ]
        
        self.FOREHEAD_INDICES = [
            67,  # Правый край лба
            296  # Левый край лба
        ]
        
        self.FACE_SHAPE_DESCRIPTIONS = {
            "oval": """
                У вас овальная форма лица, которая считается идеальной и наиболее универсальной в мире стиля. 
                Ваше лицо имеет сбалансированные пропорции с мягкими контурами, плавно сужающимися к подбородку. 
                Скулы слегка выражены, а подбородок мягко закруглен, что создает гармоничный силуэт.
                
                Овальная форма лица отличается хорошей симметрией и отсутствием каких-либо доминирующих черт.
                Это наиболее универсальная форма, позволяющая экспериментировать с разнообразными прическами и стилями.
            """,
            "round": """
                У вас круглая форма лица, которая характеризуется мягкими линиями и отсутствием резких углов.
                Ширина и высота вашего лица примерно одинаковы, что создает эффект округлости.
                
                Ваши скулы широкие и являются самой широкой частью лица, придавая ему мягкий, юный вид.
                Подбородок короткий и скругленный, а линия челюсти мягкая и менее выраженная.
                Основная черта круглого лица — это полные щеки и отсутствие резко выраженных углов.
            """,
            "square": """
                У вас квадратная форма лица, которая говорит о сильном характере и выразительных чертах.
                Ваше лицо имеет широкую линию челюсти с четко выраженными углами, что придает внешности структурность.
                
                Ширина лба, скул и челюсти примерно одинаковая, что создает эффект прямоугольности или квадрата.
                Подбородок широкий и скорее плоский, чем заостренный, с четкими линиями.
                Квадратная форма лица подчеркивает решительность и характер, придавая облику уверенность и стойкость.
            """,
            "heart": """
                У вас сердцевидная форма лица, которая считается одной из самых романтичных и женственных.
                Ваше лицо характеризуется широким лбом и выраженными скулами, элегантно сужающимися к подбородку.
                
                Подбородок заостренный и является самой узкой частью лица, напоминая кончик сердца.
                Линия роста волос часто имеет V-образную форму или явный пик, дополняя сердцевидный эффект.
                Эта форма лица придает образу утонченность и женственность, выделяя глаза и скулы.
            """,
            "diamond": """
                У вас ромбовидная форма лица, которая считается редкой и выразительной.
                Главная особенность вашего лица — выраженные скулы, которые являются самой широкой его частью.
                
                Лоб и линия подбородка заметно уже, чем скулы, что создает хорошо очерченный силуэт.
                Подбородок может быть заостренным или узким, дополняя общую структуру ромба.
                Такая форма лица делает взгляд особенно выразительным и подчеркивает уникальность черт.
            """,
            "oblong": """
                У вас продолговатая форма лица, которая отличается элегантной вытянутостью и утонченностью.
                Ваше лицо имеет значительно большую длину, чем ширину, с вытянутыми пропорциями.
                
                Лоб, скулы и линия челюсти имеют примерно одинаковую ширину, создавая ровный вертикальный силуэт.
                Подбородок скругленный, а лицо в целом выглядит удлиненным с прямыми боковыми линиями.
                Продолговатая форма придает облику аристократичность и изысканность.
            """,
            "triangle": """
                У вас треугольная форма лица (основание внизу), которая добавляет вашему облику характерной выразительности.
                Ваше лицо имеет узкий лоб и широкую линию челюсти, создавая интересный контраст верхней и нижней частей.
                
                Подбородок широкий и зачастую является самой широкой частью лица, придавая чертам силу и уверенность.
                Скулы могут быть широкими, но обычно не шире линии челюсти, плавно расширяясь от висков вниз.
                Эта форма лица подчеркивает сильный характер и выразительный профиль.
            """
        }
    
    def analyze(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Analyze the face in the provided image.
        
        Args:
            image: The input image as a numpy array (BGR format from OpenCV)
            
        Returns:
            A dictionary containing analysis results
        """
        # Проверка, что изображение не пустое
        if image is None or image.size == 0:
            return {
                "success": False,
                "message": "Предоставлено пустое изображение"
            }
        
        # Преобразуем BGR в RGB для MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, _ = image.shape
        
        # Обрабатываем изображение с помощью Face Mesh
        results = self.face_mesh.process(image_rgb)
        
        # Проверяем, обнаружено ли лицо
        if not results.multi_face_landmarks:
            return {
                "success": False,
                "message": "Лицо не обнаружено на изображении. Пожалуйста, убедитесь, что лицо хорошо освещено и находится в фокусе камеры."
            }
        
        # Получаем первое (и, предположительно, единственное) лицо
        face_landmarks = results.multi_face_landmarks[0]
        landmarks = [(lm.x * width, lm.y * height) for lm in face_landmarks.landmark]
        
        # Создаем копию изображения для отображения точек и измерений
        visualization_image = image.copy()
        
        # Настраиваем стиль отображения на более компактный и читаемый для Telegram
        connection_style = self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
        connection_style.color = (0, 180, 255)  # Оранжевый цвет для соединений 
        connection_style.thickness = 1  # Тонкие линии для соединений
        
        # Рисуем основную сетку
        self.mp_drawing.draw_landmarks(
            image=visualization_image,
            landmark_list=face_landmarks,
            connections=self.mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=connection_style
        )
        
        # Добавляем анализируемые точки более заметно
        # Обводим ключевые точки для измерений более крупными кругами
        for idx in self.FACE_OVAL_INDICES + self.CHEEKBONE_INDICES + self.JAW_INDICES + self.FOREHEAD_INDICES:
            # Сначала рисуем более широкую обводку
            cv2.circle(visualization_image, 
                      (int(landmarks[idx][0]), int(landmarks[idx][1])), 
                      5, (0, 0, 0), 2)  # Черная обводка
            
            # Затем рисуем цветную точку внутри
            cv2.circle(visualization_image, 
                      (int(landmarks[idx][0]), int(landmarks[idx][1])), 
                      3, (0, 255, 0), -1)  # Зеленая точка
        
        # Получаем измерения лица
        measurements = self._measure_face(landmarks)
        
        # Дополняем визуализацию основными измерениями
        self._draw_measurement_lines(visualization_image, landmarks, measurements)
        
        # Определяем форму лица
        face_shape, confidence = self._determine_face_shape(measurements)
        
        # Добавляем текст с результатом на изображение
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(visualization_image, 
                   f"Форма лица: {face_shape.upper()}", 
                   (10, 30), 
                   font, 0.7, (255, 255, 255), 4)  # Обводка текста
        
        cv2.putText(visualization_image, 
                   f"Форма лица: {face_shape.upper()}", 
                   (10, 30), 
                   font, 0.7, (0, 0, 255), 2)  # Основной текст
        
        # Добавляем показатель уверенности
        conf_text = f"Уверенность: {int(confidence * 100)}%"
        cv2.putText(visualization_image, 
                   conf_text, 
                   (10, 60), 
                   font, 0.6, (255, 255, 255), 3)  # Обводка
        
        cv2.putText(visualization_image, 
                   conf_text, 
                   (10, 60), 
                   font, 0.6, (0, 200, 0), 1)  # Основной текст
        
        # Преобразуем изображение с разметкой в base64 для отправки в веб-интерфейс
        _, buffer = cv2.imencode('.jpg', visualization_image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Создаем словарь с результатами анализа
        return {
            "success": True,
            "face_shape": face_shape,
            "confidence": confidence,
            "description": self.FACE_SHAPE_DESCRIPTIONS[face_shape].strip(),
            "measurements": measurements,
            "image_with_landmarks": f"data:image/jpeg;base64,{img_base64}"
        }
        
    def _draw_measurement_lines(self, image: np.ndarray, landmarks: List[Tuple[float, float]], measurements: Dict[str, float]) -> None:
        """
        Рисует линии основных измерений на изображении для лучшей визуализации.
        
        Args:
            image: Изображение для рисования
            landmarks: Список лицевых точек
            measurements: Словарь с измерениями
        """
        # Цвета для разных типов измерений
        forehead_color = (100, 180, 0)    # Зеленый
        cheekbone_color = (0, 165, 255)   # Оранжевый
        jaw_color = (0, 0, 255)           # Красный
        height_color = (255, 0, 0)        # Синий
        
        # Толщина линий
        line_thickness = 2
        
        # Рисуем линию ширины лба
        p1 = (int(landmarks[self.FOREHEAD_INDICES[0]][0]), int(landmarks[self.FOREHEAD_INDICES[0]][1]))
        p2 = (int(landmarks[self.FOREHEAD_INDICES[1]][0]), int(landmarks[self.FOREHEAD_INDICES[1]][1]))
        cv2.line(image, p1, p2, forehead_color, line_thickness)
        
        # Добавляем текст с измерением
        mid_point = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2 - 10)
        cv2.putText(image, f"{int(measurements['forehead_width'])}", mid_point, 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 3)
        cv2.putText(image, f"{int(measurements['forehead_width'])}", mid_point, 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, forehead_color, 1)
        
        # Рисуем линию ширины скул
        p1 = (int(landmarks[self.CHEEKBONE_INDICES[0]][0]), int(landmarks[self.CHEEKBONE_INDICES[0]][1]))
        p2 = (int(landmarks[self.CHEEKBONE_INDICES[1]][0]), int(landmarks[self.CHEEKBONE_INDICES[1]][1]))
        cv2.line(image, p1, p2, cheekbone_color, line_thickness)
        
        # Добавляем текст с измерением
        mid_point = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2 - 10)
        cv2.putText(image, f"{int(measurements['cheekbone_width'])}", mid_point, 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 3)
        cv2.putText(image, f"{int(measurements['cheekbone_width'])}", mid_point, 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, cheekbone_color, 1)
        
        # Рисуем линию ширины челюсти
        p1 = (int(landmarks[self.JAW_INDICES[0]][0]), int(landmarks[self.JAW_INDICES[0]][1]))
        p2 = (int(landmarks[self.JAW_INDICES[1]][0]), int(landmarks[self.JAW_INDICES[1]][1]))
        cv2.line(image, p1, p2, jaw_color, line_thickness)
        
        # Добавляем текст с измерением
        mid_point = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2 + 20)
        cv2.putText(image, f"{int(measurements['jaw_width'])}", mid_point, 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 3)
        cv2.putText(image, f"{int(measurements['jaw_width'])}", mid_point, 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, jaw_color, 1)
        
        # Рисуем линию высоты лица
        p1 = (int(landmarks[self.FACE_OVAL_INDICES[2]][0]), int(landmarks[self.FACE_OVAL_INDICES[2]][1]))
        p2 = (int(landmarks[self.FACE_OVAL_INDICES[0]][0]), int(landmarks[self.FACE_OVAL_INDICES[0]][1]))
        cv2.line(image, p1, p2, height_color, line_thickness)
        
        # Добавляем текст с измерением
        mid_point = (p1[0] + 30, (p1[1] + p2[1]) // 2)
        cv2.putText(image, f"{int(measurements['face_height'])}", mid_point, 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 3)
        cv2.putText(image, f"{int(measurements['face_height'])}", mid_point, 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, height_color, 1)
    
    def _measure_face(self, landmarks: List[Tuple[float, float]]) -> Dict[str, float]:
        """
        Perform measurements on the face to determine its shape.
        
        Args:
            landmarks: List of facial landmarks as (x, y) coordinates
            
        Returns:
            Dictionary with various face measurements
        """
        # Вычисляем характерные измерения лица
        
        # Ширина лица в разных точках
        forehead_width = self._distance(landmarks[self.FOREHEAD_INDICES[0]], landmarks[self.FOREHEAD_INDICES[1]])
        cheekbone_width = self._distance(landmarks[self.CHEEKBONE_INDICES[0]], landmarks[self.CHEEKBONE_INDICES[1]])
        jaw_width = self._distance(landmarks[self.JAW_INDICES[0]], landmarks[self.JAW_INDICES[1]])
        
        # Высота лица
        face_height = self._distance(landmarks[self.FACE_OVAL_INDICES[2]], landmarks[self.FACE_OVAL_INDICES[0]])
        
        # Высота нижней части лица (от скул до подбородка)
        lower_face_height = self._distance(
            ((landmarks[self.CHEEKBONE_INDICES[0]][0] + landmarks[self.CHEEKBONE_INDICES[1]][0]) / 2,
             (landmarks[self.CHEEKBONE_INDICES[0]][1] + landmarks[self.CHEEKBONE_INDICES[1]][1]) / 2),
            landmarks[self.FACE_OVAL_INDICES[0]]
        )
        
        # Расстояние от линии подбородка до линии скул
        chin_to_jaw_length = self._distance(
            landmarks[self.FACE_OVAL_INDICES[0]],
            ((landmarks[self.JAW_INDICES[0]][0] + landmarks[self.JAW_INDICES[1]][0]) / 2,
             (landmarks[self.JAW_INDICES[0]][1] + landmarks[self.JAW_INDICES[1]][1]) / 2)
        )
        
        # Соотношения различных измерений
        face_width_to_height_ratio = cheekbone_width / face_height
        forehead_to_jaw_ratio = forehead_width / jaw_width
        cheekbone_to_jaw_ratio = cheekbone_width / jaw_width
        
        return {
            "forehead_width": forehead_width,
            "cheekbone_width": cheekbone_width,
            "jaw_width": jaw_width,
            "face_height": face_height,
            "lower_face_height": lower_face_height,
            "chin_to_jaw_length": chin_to_jaw_length,
            "face_width_to_height_ratio": face_width_to_height_ratio,
            "forehead_to_jaw_ratio": forehead_to_jaw_ratio,
            "cheekbone_to_jaw_ratio": cheekbone_to_jaw_ratio
        }
    
    def _distance(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two points."""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def _determine_face_shape(self, measurements: Dict[str, float]) -> Tuple[str, float]:
        """
        Determine the face shape based on various measurements.
        
        Args:
            measurements: Dictionary of face measurements
            
        Returns:
            Tuple containing (face_shape, confidence)
        """
        # Создаем словарь для хранения "очков" для каждой формы лица
        shape_scores = {
            "oval": 0,
            "round": 0,
            "square": 0,
            "heart": 0,
            "diamond": 0,
            "oblong": 0,
            "triangle": 0
        }
        
        # Улучшенный алгоритм определения формы лица
        
        # Овальная форма лица
        # - Сбалансированные пропорции
        # - Соотношение ширины к высоте около 2/3
        # - Мягкая линия подбородка
        # - Нет выраженных "углов" или чрезмерных пропорций
        if 0.63 <= measurements["face_width_to_height_ratio"] <= 0.77:
            shape_scores["oval"] += 2
        
        if 0.87 <= measurements["forehead_to_jaw_ratio"] <= 1.13:
            shape_scores["oval"] += 1
        
        if 0.9 <= measurements["cheekbone_to_jaw_ratio"] <= 1.15:
            shape_scores["oval"] += 1
        
        # Круглая форма лица
        # - Примерно одинаковая ширина и высота
        # - Скулы - самая широкая часть лица
        # - Мягкая линия подбородка без выраженных углов
        if measurements["face_width_to_height_ratio"] >= 0.78:
            shape_scores["round"] += 3
        elif measurements["face_width_to_height_ratio"] >= 0.73:
            shape_scores["round"] += 1
        
        if measurements["cheekbone_width"] > measurements["forehead_width"] * 1.05 and measurements["cheekbone_width"] > measurements["jaw_width"] * 1.05:
            shape_scores["round"] += 2
        
        if measurements["chin_to_jaw_length"] < measurements["lower_face_height"] * 0.35:
            shape_scores["round"] += 1
        
        # Квадратная форма лица
        # - Ширина лба, скул и челюсти примерно одинаковая
        # - Выраженная линия челюсти с четкими углами
        if 0.9 <= measurements["forehead_to_jaw_ratio"] <= 1.1 and 0.9 <= measurements["cheekbone_to_jaw_ratio"] <= 1.1:
            shape_scores["square"] += 3
        
        if measurements["chin_to_jaw_length"] < measurements["lower_face_height"] * 0.28:
            shape_scores["square"] += 2
        
        # Сердцевидная форма лица
        # - Широкий лоб и узкий подбородок
        # - Сужение лица к подбородку
        if measurements["forehead_width"] > measurements["jaw_width"] * 1.15:
            shape_scores["heart"] += 3
        elif measurements["forehead_width"] > measurements["jaw_width"] * 1.08:
            shape_scores["heart"] += 1
        
        if measurements["forehead_width"] > measurements["cheekbone_width"] * 1.05:
            shape_scores["heart"] += 2
        
        if measurements["chin_to_jaw_length"] > measurements["lower_face_height"] * 0.38:
            shape_scores["heart"] += 1
        
        # Ромбовидная форма лица
        # - Скулы - самая широкая часть лица
        # - Лоб и подбородок уже чем скулы
        if measurements["cheekbone_width"] > measurements["forehead_width"] * 1.13 and measurements["cheekbone_width"] > measurements["jaw_width"] * 1.13:
            shape_scores["diamond"] += 3
        elif measurements["cheekbone_width"] > measurements["forehead_width"] * 1.05 and measurements["cheekbone_width"] > measurements["jaw_width"] * 1.05:
            shape_scores["diamond"] += 1
        
        # Проверяем, что лоб и подбородок сужены по сравнению со скулами
        narrow_top_bottom = (measurements["cheekbone_width"] - measurements["forehead_width"]) > 0 and (measurements["cheekbone_width"] - measurements["jaw_width"]) > 0
        if narrow_top_bottom:
            shape_scores["diamond"] += 2
        
        # Продолговатая форма лица
        # - Длинное лицо с примерно одинаковой шириной лба, скул и челюсти
        # - Соотношение высоты к ширине больше чем у овального
        if measurements["face_width_to_height_ratio"] < 0.62:
            shape_scores["oblong"] += 3
        elif measurements["face_width_to_height_ratio"] < 0.67:
            shape_scores["oblong"] += 1
        
        if 0.85 <= measurements["forehead_to_jaw_ratio"] <= 1.15 and 0.85 <= measurements["cheekbone_to_jaw_ratio"] <= 1.15:
            shape_scores["oblong"] += 2
        
        # Треугольная форма лица (основание вниз)
        # - Узкий лоб и широкая челюсть
        # - Обратная форма сердца
        if measurements["jaw_width"] > measurements["forehead_width"] * 1.15:
            shape_scores["triangle"] += 3
        elif measurements["jaw_width"] > measurements["forehead_width"] * 1.05:
            shape_scores["triangle"] += 1
        
        if measurements["jaw_width"] > measurements["cheekbone_width"] * 1.05:
            shape_scores["triangle"] += 2
        
        # Логируем очки для отладки
        logging.debug(f"Shape scores: {shape_scores}")
        logging.debug(f"Measurements: {measurements}")
        
        # Определяем форму с наибольшим количеством очков
        max_score = max(shape_scores.values())
        
        # Находим все формы с максимальным количеством очков
        top_shapes = [shape for shape, score in shape_scores.items() if score == max_score]
        
        # Если есть несколько форм с одинаковым количеством очков, выбираем приоритетную
        # Приоритет: oval > heart > diamond > round > square > oblong > triangle
        priority_order = ["oval", "heart", "diamond", "round", "square", "oblong", "triangle"]
        
        # Выбираем форму с наивысшим приоритетом из top_shapes
        face_shape = next((shape for shape in priority_order if shape in top_shapes), "oval")
        
        # Рассчитываем уверенность в определении формы
        # (от 0.55 до 1.0 в зависимости от количества очков)
        total_possible_score = 6  # Максимально возможное количество очков для каждой формы
        confidence = 0.55 + (max_score / total_possible_score) * 0.45
        
        return face_shape, confidence