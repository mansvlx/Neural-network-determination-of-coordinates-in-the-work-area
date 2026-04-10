import cv2
import numpy as np

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

        # Обработка кадра с новой функцией
        processed_frame = process_frame_with_localization(frame, REAL_OBJECT_WIDTH_MM)

        # Отображение обработанного кадра
        cv2.imshow('Object Localization | Center (0;0)', processed_frame)

        # Выход по нажатию клавиши 'q'
        if cv2.waitKey(1) == ord('q'):
            break

    # Освобождение ресурсов
    cap.release()
    cv2.destroyAllWindows()

def process_frame_with_localization(frame, real_object_width_mm=None):
    """
    Обрабатывает кадр:
    - Находит самый большой контур
    - Вписывает в minAreaRect
    - Строит диагонали
    - Привязывает центр к (0;0)
    - Вычисляет масштаб
    """
    # Преобразование в оттенки серого
    img_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Повышение контрастности
    img_grey = cv2.equalizeHist(img_grey)

    # Применение размытия для снижения шума
    blurred = cv2.GaussianBlur(img_grey, (5, 5), 0)

    # Применение пороговой обработки для выделения объектов
    # Убрал THRESH_BINARY_INV, оставил только OTSU для лучшей адаптации
    ret, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Морфологические операции для удаления мелких шумов
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Заполнение "дыр" внутри объектов
    thresh = cv2.dilate(thresh, kernel, iterations=3)

    # Поиск контуров
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output = frame.copy()
    
    # Фильтруем контуры по площади (минимум 500 пикселей)
    min_contour_area = 500
    valid_contours = [c for c in contours if cv2.contourArea(c) > min_contour_area]
    
    if valid_contours:
        # Находим самый большой контур (главный объект)
        main_contour = max(valid_contours, key=cv2.contourArea)
        
        # 1. Вписываем в минимальный ограничивающий прямоугольник (учитывает поворот)
        rect = cv2.minAreaRect(main_contour)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        
        # 2. Центр объекта - пересечение диагоналей (это и есть (0;0))
        center_x, center_y = rect[0]
        width_px, height_px = rect[1]
        angle = rect[2]
        
        # 3. Расчет масштаба
        scale_mm_per_px = None
        scale_text = ""
        if real_object_width_mm is not None and width_px > 0:
            scale_mm_per_px = real_object_width_mm / width_px
            scale_text = f"Scale: {scale_mm_per_px:.3f} mm/px"
        else:
            scale_mm_per_px = 1.0
            scale_text = "Scale: 1 px/unit"
        
        # 4. Рисуем контур объекта
        cv2.drawContours(output, [main_contour], -1, (0, 255, 0), 2)
        
        # 5. Рисуем ограничивающий прямоугольник
        cv2.drawContours(output, [box], 0, (255, 0, 0), 2)
        
        # 6. Строим диагонали прямоугольника
        # box содержит 4 точки. Диагонали соединяют противоположные углы
        cv2.line(output, tuple(box[0]), tuple(box[2]), (0, 0, 255), 2)  # Диагональ 1
        cv2.line(output, tuple(box[1]), tuple(box[3]), (0, 0, 255), 2)  # Диагональ 2
        
        # 7. Рисуем центр объекта (0;0)
        cv2.circle(output, (int(center_x), int(center_y)), 7, (0, 255, 255), -1)
        cv2.putText(output, "(0;0)", (int(center_x) + 15, int(center_y) - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # 8. Рисуем оси координат от центра (привязка к объекту)
        axis_len = 100
        
        # Ось X (вправо) - Красная
        end_x = (int(center_x + axis_len), int(center_y))
        cv2.arrowedLine(output, (int(center_x), int(center_y)), end_x, 
                       (0, 0, 255), 2, tipLength=0.2)
        cv2.putText(output, "X", end_x, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Ось Y (вверх) - Синяя
        end_y = (int(center_x), int(center_y - axis_len))
        cv2.arrowedLine(output, (int(center_x), int(center_y)), end_y,
                       (255, 0, 0), 2, tipLength=0.2)
        cv2.putText(output, "Y", end_y, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # 9. Выводим информацию о координатах углов
        info_y = 30
        cv2.putText(output, f"Center: (0;0)", (10, info_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        info_y += 25
        cv2.putText(output, f"Angle: {angle:.1f} deg", (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        info_y += 25
        cv2.putText(output, f"Size: {width_px:.0f} x {height_px:.0f} px", (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        info_y += 25
        cv2.putText(output, scale_text, (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 10. Выводим координаты углов в мм (если известен масштаб)
        if real_object_width_mm is not None:
            info_y += 30
            cv2.putText(output, "Corners (mm from center):", (10, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            for i, pt in enumerate(box):
                # Пересчет в локальные координаты с центром в (0;0)
                local_x_px = pt[0] - center_x
                local_y_px = center_y - pt[1]  # Инвертируем Y, чтобы он шел вверх
                
                local_x_mm = local_x_px * scale_mm_per_px
                local_y_mm = local_y_px * scale_mm_per_px
                
                # Подписываем углы на изображении
                cv2.putText(output, f"C{i}", tuple(pt), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                
                # Выводим координаты
                info_y += 20
                cv2.putText(output, f"  C{i}: ({local_x_mm:+.1f}, {local_y_mm:+.1f}) mm", 
                           (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        
        # Выводим площадь контура
        area = cv2.contourArea(main_contour)
        cv2.putText(output, f"Area: {area:.0f} px^2", (10, frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    else:
        # Если объект не найден
        cv2.putText(output, "Object not found", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    return output

if __name__ == "__main__":
    main()