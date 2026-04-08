import cv2 as cv
import numpy as np
import time
from shapely.geometry import LineString, Point
import torch

class YOLODetector:
    def __init__(self, model_path='yolov8n.pt'):
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=False)
        self.model.conf = 0.5  # Уверенность детекции
        self.model.iou = 0.45  # IoU порог
        
    def detect_work_area(self, img):
        results = self.model(img)
        detections = results.pandas().xyxy[0]
        work_areas = detections[detections['name'] == 'work_area']
        
        if len(work_areas) > 0:
            # Берем первую найденную рабочую область
            area = work_areas.iloc[0]
            x1, y1, x2, y2 = int(area['xmin']), int(area['ymin']), int(area['xmax']), int(area['ymax'])
            return [(x1, y1), (x2, y2), (x2, y1), (x1, y2)]
        return None
    
    def detect_calibration_marker(self, img):
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
def transform_to_work_area_coordinates(self, board_coords):
        # Преобразует координаты заготовки в систему координат рабочей области
        if not self.origin_point:
            return None
        # Находим центр заготовки
        center_x = int(np.mean([p[0] for p in board_coords]))
        center_y = int(np.mean([p[1] for p in board_coords]))
        # Вычисляем относительные координаты
        rel_x = center_x - self.origin_point[0]
        rel_y = self.origin_point[1] - center_y
        # Преобразуем в реальные координаты (мм)
        # Здесь нужна ручная калибровка: сколько пикселей в мм
        pixels_per_mm_x = 34.5 
        pixels_per_mm_y = 35 
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
def get_best_horizontal(self, horizontal_lines):
    """Выбирает лучшую горизонтальную линию"""
    if not horizontal_lines:
        return [0, 0, 0, 0]
    
    # Выбираем самую длинную горизонтальную линию
    longest = max(horizontal_lines, key=lambda line: np.sqrt((line[2]-line[0])**2 + (line[3]-line[1])**2))
    return longest

def detect_work_area_with_yolo(self, img):
        """Определение рабочей области с помощью YOLO"""
        work_area = self.yolo.detect_work_area(img)
        
        if work_area is None:
            # Fallback на старый метод, если YOLO не нашел
            return self.correcting_perspective_old(img)
        
        # Сохраняем координаты рабочей области
        self.work_area_coords = work_area
        
        # Находим точку отсчета (левый нижний угол рабочей области)
        self.origin_point = self.find_origin_point(work_area)
        
        return self.correct_perspective_by_yolo(img, work_area)
    
def find_origin_point(self, work_area):
    """Находит точку отсчета (левый нижний угол)"""
    # Сортируем точки по x и y
    points = sorted(work_area, key=lambda p: (p[0], p[1]))
    # Левая нижняя точка имеет минимальный x и максимальный y
    origin = min(points, key=lambda p: (p[0], -p[1]))
    return origin

def correct_perspective_by_yolo(self, img, work_area):
    """Корректирует перспективу на основе найденной YOLO области"""
    input_pts = np.float32(work_area)
    
    # Создаем выходные точки для прямоугольной области
    width = int(max([p[0] for p in work_area]) - min([p[0] for p in work_area]))
    height = int(max([p[1] for p in work_area]) - min([p[1] for p in work_area]))
    
    output_pts = np.float32(
        [[0, height],# левый нижний
        [0, 0],       # левый верхний
        [width, 0],   # правый верхний
        [width, height]  # правый нижний
    ])
    
    M = cv.getPerspectiveTransform(input_pts, output_pts)
    out = cv.warpPerspective(img, M, (width, height), flags=cv.INTER_LINEAR)
    
    return out

def correcting_perspective_old(self, img):
    """Старый метод коррекции перспективы (для обратной совместимости)"""
    pt_A = [212, 284]
    pt_B = [212, 2097]
    pt_C = [3294, 2033]
    pt_D = [3208, 207]
    
    input_pts = np.float32([pt_A, pt_B, pt_C, pt_D])
    output_pts = np.float32([[210, 204], [210, 2097], [3296, 2097], [3296, 204]])
    
    M = cv.getPerspectiveTransform(input_pts, output_pts)
    out = cv.warpPerspective(img, M, (img.shape[1], img.shape[0]), flags=cv.INTER_LINEAR)
    
    return out[204:2097, 210:3296]

def find_board_relative_to_work_area(self, img_path, req_diagonal, min_side, max_side):
    """Находит заготовку и определяет ее координаты относительно рабочей области"""
    # Загружаем и корректируем изображение
    img = cv.imread(img_path)
    
    # Если у нас есть рабочая область, используем YOLO для коррекции
    if self.work_area_coords is not None:
        img_corrected = self.correct_perspective_by_yolo(img, self.work_area_coords)
    else:
        img_corrected = self.correcting_perspective_old(img)
    
    # Определяем заготовку (используем существующий метод)
    board_coords = self.find_board_by_cam_two_enhanced(img_corrected, req_diagonal, min_side, max_side)
    
    if board_coords and self.origin_point:
        # Переводим координаты заготовки в систему координат рабочей области
        relative_coords = self.transform_to_work_area_coordinates(board_coords)
        return board_coords, relative_coords
    
    return board_coords, None

def find_board_by_cam_two_enhanced(self, img, req_diagonal, min_side, max_side):
    """Улучшенный метод поиска заготовки с использованием YOLO"""
    # Пробуем найти заготовку через YOLO
    yolo_board = self.yolo.detect_work_area(img)  # Используем ту же модель, но с другим классом
    
    if yolo_board:
        # Если YOLO нашел, проверяем размеры
        box = np.array(yolo_board, dtype=np.int0)
        diagonal_ = self.get_diagonal(box) / 10.286
        min_side_, max_side_ = self.det_min_max_side(box)



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
    # Значения ниже примерные
    req_diagonal = 100 
    min_side = 50
    max_side = 80 
    board_coords, relative_coords = fwa.find_board_relative_to_work_area(
        img_path, req_diagonal, min_side, max_side)
    if relative_coords:
        print(f"Координаты заготовки относительно рабочей области: X={relative_coords[0]:.2f} мм, Y={relative_coords[1]:.2f} мм")
if __name__ == "__main__":
    main()




