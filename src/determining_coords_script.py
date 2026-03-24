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




