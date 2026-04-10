import cv2
import numpy as np
from ultralytics import YOLO

try:
    obb_model = YOLO("yolo11n-obb.pt")
    print("YOLO-OBB model loaded successfully.")
except Exception as e:
    print(f"Failed to load YOLO-OBB model: {e}")
    obb_model = None

def detect_workspace(frame, confidence_threshold=0.5):

    # Возвращает координаты рабочей области (x, y, width, height, angle) в пикселях.
    if obb_model is None:
        return None

    # Выполняем предсказание
    results = obb_model(frame, conf=confidence_threshold, verbose=False)
    
    if len(results) == 0 or results[0].obb is None:
        return None

    # Получаем данные OBB 
    obb_data = results[0].obb.xywhr.cpu().numpy()  # [x_center, y_center, width, height, angle]
    confidences = results[0].obb.conf.cpu().numpy()
    classes = results[0].obb.cls.cpu().numpy()
    
    if len(obb_data) == 0:
        return None
        
    best_idx = np.argmax(confidences)
    workspace = obb_data[best_idx]
    
    # Преобразуем угол из радиан в градусы
    x, y, w, h, angle_rad = workspace
    angle_deg = np.degrees(angle_rad)
    
    # Возвращаем параметры рабочей области: центр (x, y), размеры (w, h), угол в градусах
    return {
        'center': (float(x), float(y)),
        'width': float(w),
        'height': float(h),
        'angle': float(angle_deg),
        'confidence': float(confidences[best_idx]),
        'class': int(classes[best_idx])
    }

def main():
    # Инициализация камеры
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Не удалось открыть камеру")
        return

    # КАЛИБРОВКА: Укажите реальную ширину объекта в мм
    # Например, если ваш объект шириной 50 мм, укажите 50.0
    REAL_OBJECT_WIDTH_MM = 50.0  # <-- ИЗМЕНИТЕ ПОД ВАШ ОБЪЕКТ
    # Если реальный размер неизвестен, поставьте None
    # REAL_OBJECT_WIDTH_MM = None

    while True:
        # Захват кадра
        ret, frame = cap.read()

        # Проверка успешности захвата кадра
        if not ret:
            print("Не удалось получить кадр. Выход...")
            break

        # ДОБАВЛЕНО: Обнаружение рабочей области
        workspace = detect_workspace(frame)
        
        # Обработка кадра с новой функцией
        processed_frame = process_frame_with_localization(frame, REAL_OBJECT_WIDTH_MM, workspace)

        # Отображение обработанного кадра
        cv2.imshow('Object Localization | Center (0;0)', processed_frame)

        # Выход по нажатию клавиши 'q'
        if cv2.waitKey(1) == ord('q'):
            break

    # Освобождение ресурсов
    cap.release()
    cv2.destroyAllWindows()

def process_frame_with_localization(frame, real_object_width_mm=None, workspace=None):
    """
    Обрабатывает кадр:
    - Находит самый большой контур (объект)
    - Вписывает в minAreaRect
    - Строит диагонали
    - Привязывает центр к (0;0)
    - Если рабочая область найдена, вычисляет координаты объекта относительно неё
    """
    # Преобразование в оттенки серого
    img_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Повышение контрастности
    img_grey = cv2.equalizeHist(img_grey)

    # Применение размытия для снижения шума
    blurred = cv2.GaussianBlur(img_grey, (5, 5), 0)

    # Применение пороговой обработки для выделения объектов
    ret, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Морфологические операции для удаления мелких шумов
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Заполнение "дыр" внутри объектов
    thresh = cv2.dilate(thresh, kernel, iterations=3)

    # Поиск контуров
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output = frame.copy()
    
    if workspace is not None:
        # Рисуем рабочую область
        center_x, center_y = workspace['center']
        width, height = workspace['width'], workspace['height']
        angle = workspace['angle']
        
        # Получаем 4 угла повёрнутого прямоугольника
        rect = ((center_x, center_y), (width, height), angle)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        
        # Рисуем контур рабочей области
        cv2.drawContours(output, [box], 0, (255, 255, 0), 2)
        
        # Рисуем центр рабочей области
        cv2.circle(output, (int(center_x), int(center_y)), 5, (255, 255, 0), -1)
        cv2.putText(output, "Workspace", (int(center_x) + 10, int(center_y) - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Выводим информацию о рабочей области
        cv2.putText(output, f"Workspace: {width:.0f}x{height:.0f} px, {angle:.1f} deg", 
                   (10, output.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    # Фильтруем контуры по площади (минимум 500 пикселей)
    min_contour_area = 500
    valid_contours = [c for c in contours if cv2.contourArea(c) > min_contour_area]
    
    if valid_contours:
        main_contour = max(valid_contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(main_contour)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        obj_center_x, obj_center_y = rect[0]
        width_px, height_px = rect[1]
        angle = rect[2]
        scale_mm_per_px = None
        scale_text = ""
        if real_object_width_mm is not None and width_px > 0:
            scale_mm_per_px = real_object_width_mm / width_px
            scale_text = f"Scale: {scale_mm_per_px:.3f} mm/px"
        else:
            scale_mm_per_px = 1.0
            scale_text = "Scale: 1 px/unit"
        cv2.drawContours(output, [box], 0, (255, 0, 0), 2)
        cv2.line(output, tuple(box[0]), tuple(box[2]), (0, 0, 255), 2)  
        cv2.line(output, tuple(box[1]), tuple(box[3]), (0, 0, 255), 2)
        cv2.circle(output, (int(obj_center_x), int(obj_center_y)), 7, (0, 255, 255), -1)
        cv2.putText(output, "(0;0) Obj", (int(obj_center_x) + 15, int(obj_center_y) - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        axis_len = 100
        # Ось X (вправо) - Красная
        end_x = (int(obj_center_x + axis_len), int(obj_center_y))
        cv2.arrowedLine(output, (int(obj_center_x), int(obj_center_y)), end_x, 
                       (0, 0, 255), 2, tipLength=0.2)
        cv2.putText(output, "X", end_x, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        # Ось Y (вверх) - Синяя
        end_y = (int(obj_center_x), int(obj_center_y - axis_len))
        cv2.arrowedLine(output, (int(obj_center_x), int(obj_center_y)), end_y,
                       (255, 0, 0), 2, tipLength=0.2)
        cv2.putText(output, "Y", end_y, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        info_y = 30
        cv2.putText(output, f"Object Center: (0;0)", (10, info_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        info_y += 25
        cv2.putText(output, f"Object Angle: {angle:.1f} deg", (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        info_y += 25
        cv2.putText(output, f"Object Size: {width_px:.0f} x {height_px:.0f} px", (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        info_y += 25
        cv2.putText(output, scale_text, (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        if workspace is not None:
            ws_center_x, ws_center_y = workspace['center']
            ws_width, ws_height = workspace['width'], workspace['height']
            ws_angle = workspace['angle']
            
            # Вычисляем смещение объекта относительно центра рабочей области
            delta_x_mm = (obj_center_x - ws_center_x) * scale_mm_per_px
            delta_y_mm = (ws_center_y - obj_center_y) * scale_mm_per_px  # Инвертируем Y
            
            info_y += 30
            cv2.putText(output, "Position relative to Workspace:", (10, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            info_y += 20
            cv2.putText(output, f"  dX: {delta_x_mm:+.2f} mm, dY: {delta_y_mm:+.2f} mm", 
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # Выводим угол между объектом и рабочей областью
            relative_angle = angle - ws_angle
            info_y += 20
            cv2.putText(output, f"  Relative angle: {relative_angle:+.1f} deg", 
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # Рисуем линию от центра рабочей области к центру объекта
            cv2.line(output, (int(ws_center_x), int(ws_center_y)), 
                     (int(obj_center_x), int(obj_center_y)), (0, 255, 255), 1, cv2.LINE_AA)
        
        # Выводим площадь контура
        area = cv2.contourArea(main_contour)
        cv2.putText(output, f"Object Area: {area:.0f} px^2", (10, output.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    else:
        cv2.putText(output, "Object not found", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return output

if __name__ == "__main__":
    main()
