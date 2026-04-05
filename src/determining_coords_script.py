import cv2 as cv
import numpy as np
import time
from shapely.geometry import LineString, Point
import torch

class YOLODetector:
    def __init__(self, model_path='yolov8n.pt'):
        """Инициализация YOLO модели"""
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=False)
        self.model.conf = 0.5  # Уверенность детекции
        self.model.iou = 0.45  # IoU порог
        
    def detect_work_area(self, img):
        """Детекция рабочей области с помощью YOLO"""
        results = self.model(img)
        detections = results.pandas().xyxy[0]
        
        # Ищем класс "work_area" (нужно обучить модель или использовать существующие классы)
        work_areas = detections[detections['name'] == 'work_area']
        
        if len(work_areas) > 0:
            # Берем первую найденную рабочую область
            area = work_areas.iloc[0]
            x1, y1, x2, y2 = int(area['xmin']), int(area['ymin']), int(area['xmax']), int(area['ymax'])
            return [(x1, y1), (x2, y2), (x2, y1), (x1, y2)]
        return None
    
    def detect_calibration_marker(self, img):
        """Детекция калибровочной метки (точки отсчета)"""
        results = self.model(img)
        detections = results.pandas().xyxy[0]
        
        # Ищем калибровочную метку
        markers = detections[detections['name'] == 'calibration_marker']
        
        if len(markers) > 0:
            marker = markers.iloc[0]
            center_x = int((marker['xmin'] + marker['xmax']) / 2)
            center_y = int((marker['ymin'] + marker['ymax']) / 2)
            return (center_x, center_y)
        return None

class ModernizedFrameWork:
    def __init__(self):
        self.yolo = YOLODetector()
        self.work_area_coords = None
        self.origin_point = None
        self.calibration_matrix = None
        
    def get_frames(self, id):
        """Получение кадров с камеры"""
        cam = cv.VideoCapture(id)
        assert cam.isOpened()
        cam.set(3, 1920)
        cam.set(4, 1080)
        out = np.zeros((int(cam.get(4)*2), int(cam.get(3)*2), 3))
        
        for i in range(10):
            ret, frame = cam.read()
            if ret:
                out[::2, ::2] = frame
            ret, frame = cam.read()
            if ret:
                out[::2, 1::2] = frame
            ret, frame = cam.read()
            if ret:
                out[1::2, ::2] = frame
            ret, frame = cam.read()
            if ret:
                out[1::2, 1::2] = frame
                
        cam.release()
        return out



"""Сюда добавляем функции из беседы"""
def transform_to_work_area_coordinates(self, board_coords):
        """Преобразует координаты заготовки в систему координат рабочей области"""
        if not self.origin_point:
            return None
        
        # Находим центр заготовки
        center_x = int(np.mean([p[0] for p in board_coords]))
        center_y = int(np.mean([p[1] for p in board_coords]))
        
        # Вычисляем относительные координаты (пиксели)
        rel_x = center_x - self.origin_point[0]
        rel_y = self.origin_point[1] - center_y  # Инвертируем Y для правильной ориентации
        
        # Преобразуем в реальные координаты (мм)
        # Здесь нужна калибровка: сколько пикселей в мм
        pixels_per_mm_x = 34.5  # Примерное значение, нужно откалибровать
        pixels_per_mm_y = 35    # Примерное значение, нужно откалибровать
        
        real_x = rel_x / pixels_per_mm_x
        real_y = rel_y / pixels_per_mm_y
        
        return (real_x, real_y)
    def get_diagonal(self, box):
        """Вычисляет диагональ прямоугольника"""
        points = box.reshape(-1, 2)
        distances = []
        for i in range(len(points)):
            for j in range(i+1, len(points)):
                dist = np.sqrt((points[i][0] - points[j][0])**2 + (points[i][1] - points[j][1])**2)
                distances.append(dist)
        return max(distances)
    
    def det_min_max_side(self, box):
        """Определяет минимальную и максимальную стороны"""
        points = box.reshape(-1, 2)
        sides = []
        for i in range(len(points)):
            j = (i + 1) % len(points)
            side = np.sqrt((points[i][0] - points[j][0])**2 + (points[i][1] - points[j][1])**2)
            sides.append(side)
        return min(sides), max(sides)
    
    def get_vertical_and_horizontal(self, lines, img):
        """Разделяет линии на вертикальные и горизонтальные"""
        vertical = []
        horizontal = []
           if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                
                if abs(angle) < 10 or abs(abs(angle) - 180) < 10:
                    horizontal.append([x1, y1, x2, y2])
                elif abs(abs(angle) - 90) < 10:
                    vertical.append([x1, y1, x2, y2])
        
        return vertical, horizontal, img
  def get_best_vertical(self, vertical_lines):
        """Выбирает лучшую вертикальную линию"""
        if not vertical_lines:
            return [0, 0, 0, 0]
        
        # Выбираем самую длинную вертикальную линию
        longest = max(vertical_lines, key=lambda line: np.sqrt((line[2]-line[0])**2 + (line[3]-line[1])**2))
        return longest

"""Сюда добавляем функции из беседы"""

def main():
    fwa = ModernizedFrameWork()
    
    # Получение кадра
    img = fwa.get_frames(2)
    timestamp = str(int(time.time()) % 100000)
    img_path = f'/tmp/out_2_{timestamp}.jpeg'
    cv.imwrite(img_path, img)
    
    # Определение рабочей области с помощью YOLO
    img_corrected = fwa.detect_work_area_with_yolo(img)
    
    # Поиск заготовки и получение относительных координат
    req_diagonal = 100  # Примерное значение
    min_side = 50       # Примерное значение
    max_side = 80       # Примерное значение
    
    board_coords, relative_coords = fwa.find_board_relative_to_work_area(
        img_path, req_diagonal, min_side, max_side
    )
    
    if relative_coords:
        print(f"Координаты заготовки относительно рабочей области: X={relative_coords[0]:.2f} мм, Y={relative_coords[1]:.2f} мм")
      
if __name__ == "__main__":
    main()




