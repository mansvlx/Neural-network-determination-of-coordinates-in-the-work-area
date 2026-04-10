import os
import cv2
import numpy as np


def detect_contours_with_localization(image_path, real_object_width_mm=None):
    """
    Находит контур объекта, вписывает в прямоугольник, строит диагонали,
    привязывает центр к (0;0) и вычисляет масштаб.
    
    Параметры:
    - image_path: путь к изображению
    - real_object_width_mm: реальная ширина объекта в мм (если None - масштаб в пикселях)
    """
    # Загрузка изображения
    image = cv2.imread(image_path)
    if image is None:
        print(f"Не удалось загрузить изображение: {image_path}")
        return None

    # Создаем копию для визуализации
    output_image = image.copy()
    
    # Преобразование в оттенки серого
    img_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Опционально: повышение контрастности (раскомментируйте если нужно)
    # img_grey = cv2.equalizeHist(img_grey)

    # Применение размытия для снижения шума
    blurred = cv2.GaussianBlur(img_grey, (5, 5), 0)

    # Применение пороговой обработки для выделения объектов
    ret, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Морфологические операции для удаления мелких шумов
    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Заполнение "дыр" внутри объектов
    thresh = cv2.dilate(thresh, kernel, iterations=3)

    # Поиск контуров
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Фильтрация контуров по площади (игнорируем слишком маленькие)
    min_contour_area = 500
    valid_contours = [c for c in contours if cv2.contourArea(c) > min_contour_area]
    
    if not valid_contours:
        print(f"Объекты не найдены на изображении: {image_path}")
        # Все равно сохраняем оригинал для проверки
        output_dir = 'photos_out'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}_no_object.png")
        cv2.imwrite(output_path, output_image)
        return None
    
    # Находим самый большой контур (предполагаем, что это наш объект)
    main_contour = max(valid_contours, key=cv2.contourArea)
    
    # ========== ОСНОВНАЯ ЛОГИКА ПО ЗАДАНИЮ ==========
    
    # 1. Вписываем в минимальный ограничивающий прямоугольник (учитывает поворот)
    rect = cv2.minAreaRect(main_contour)
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    
    # 2. Центр объекта - точка пересечения диагоналей (0;0)
    center_x, center_y = rect[0]
    width_px, height_px = rect[1]
    angle = rect[2]
    
    # 3. Расчет масштаба
    scale_mm_per_px = None
    if real_object_width_mm is not None and width_px > 0:
        scale_mm_per_px = real_object_width_mm / width_px
        print(f"[{os.path.basename(image_path)}] Масштаб: 1 пиксель = {scale_mm_per_px:.4f} мм")
    else:
        scale_mm_per_px = 1.0
        print(f"[{os.path.basename(image_path)}] Масштаб: не задан (координаты в пикселях)")
    
    # 4. Рисуем контур объекта (зеленый)
    cv2.drawContours(output_image, [main_contour], -1, (0, 255, 0), 2)
    
    # 5. Рисуем ограничивающий прямоугольник (синий)
    cv2.drawContours(output_image, [box], 0, (255, 0, 0), 2)
    
    # 6. Строим диагонали прямоугольника (красные)
    cv2.line(output_image, tuple(box[0]), tuple(box[2]), (0, 0, 255), 2)  # Диагональ 1
    cv2.line(output_image, tuple(box[1]), tuple(box[3]), (0, 0, 255), 2)  # Диагональ 2
    
    # 7. Отмечаем центр объекта (0;0) - желтая точка
    cv2.circle(output_image, (int(center_x), int(center_y)), 7, (0, 255, 255), -1)
    cv2.putText(output_image, "(0;0)", (int(center_x) + 15, int(center_y) - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # 8. Рисуем оси координат от центра (привязка к объекту)
    axis_len = 150  # Длина осей в пикселях
    
    # Ось X (вправо) - Красная стрелка
    end_x = (int(center_x + axis_len), int(center_y))
    cv2.arrowedLine(output_image, (int(center_x), int(center_y)), end_x,
                    (0, 0, 255), 2, tipLength=0.15)
    cv2.putText(output_image, "X", (end_x[0] + 5, end_x[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Ось Y (вверх) - Синяя стрелка
    end_y = (int(center_x), int(center_y - axis_len))
    cv2.arrowedLine(output_image, (int(center_x), int(center_y)), end_y,
                    (255, 0, 0), 2, tipLength=0.15)
    cv2.putText(output_image, "Y", (end_y[0] + 5, end_y[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    # 9. Подписываем углы прямоугольника
    for i, pt in enumerate(box):
        cv2.putText(output_image, f"C{i}", tuple(pt),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Вычисляем координаты относительно центра (0;0)
        local_x_px = pt[0] - center_x
        local_y_px = center_y - pt[1]  # Инвертируем Y для математической системы координат
        
        if real_object_width_mm is not None:
            local_x_mm = local_x_px * scale_mm_per_px
            local_y_mm = local_y_px * scale_mm_per_px
            coord_text = f"({local_x_mm:+.1f}, {local_y_mm:+.1f}) mm"
        else:
            coord_text = f"({local_x_px:+d}, {local_y_px:+d}) px"
        
        # Подписываем координаты рядом с углом
        cv2.putText(output_image, coord_text, (pt[0] + 10, pt[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    
    # 10. Выводим информационную панель
    info_panel_y = 30
    cv2.putText(output_image, f"Object: {os.path.basename(image_path)}", 
                (10, info_panel_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    info_panel_y += 30
    
    cv2.putText(output_image, f"Center: (0; 0)", 
                (10, info_panel_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    info_panel_y += 25
    
    cv2.putText(output_image, f"Angle: {angle:.1f} deg", 
                (10, info_panel_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    info_panel_y += 25
    
    cv2.putText(output_image, f"Size: {width_px:.0f} x {height_px:.0f} px", 
                (10, info_panel_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    info_panel_y += 25
    
    if real_object_width_mm is not None:
        real_width = width_px * scale_mm_per_px
        real_height = height_px * scale_mm_per_px
        cv2.putText(output_image, f"Real size: {real_width:.1f} x {real_height:.1f} mm", 
                    (10, info_panel_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        info_panel_y += 25
        cv2.putText(output_image, f"Scale: {scale_mm_per_px:.4f} mm/px", 
                    (10, info_panel_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Площадь контура
    area = cv2.contourArea(main_contour)
    cv2.putText(output_image, f"Area: {area:.0f} px^2", 
                (10, output_image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Сохраняем результат
    output_dir = 'photos_out'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}_localized.png")
    cv2.imwrite(output_path, output_image)
    print(f"[{os.path.basename(image_path)}] Результат сохранен: {output_path}")
    
    # Дополнительно сохраняем текстовый файл с координатами
    txt_path = os.path.join(output_dir, f"{base_name}_coordinates.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"Файл: {os.path.basename(image_path)}\n")
        f.write(f"Центр объекта: (0; 0)\n")
        f.write(f"Угол поворота: {angle:.2f} градусов\n")
        f.write(f"Размеры в пикселях: {width_px:.2f} x {height_px:.2f}\n")
        if real_object_width_mm is not None:
            f.write(f"Масштаб: {scale_mm_per_px:.4f} мм/пиксель\n")
            f.write(f"Реальные размеры: {width_px * scale_mm_per_px:.2f} x {height_px * scale_mm_per_px:.2f} мм\n")
        f.write("\nКоординаты углов относительно центра (0;0):\n")
        
        for i, pt in enumerate(box):
            local_x_px = pt[0] - center_x
            local_y_px = center_y - pt[1]
            
            if real_object_width_mm is not None:
                local_x_mm = local_x_px * scale_mm_per_px
                local_y_mm = local_y_px * scale_mm_per_px
                f.write(f"  Угол {i}: X = {local_x_mm:+.2f} мм, Y = {local_y_mm:+.2f} мм\n")
            else:
                f.write(f"  Угол {i}: X = {local_x_px:+d} px, Y = {local_y_px:+d} px\n")
    
    print(f"[{os.path.basename(image_path)}] Координаты сохранены: {txt_path}")
    
    return {
        'center': (center_x, center_y),
        'angle': angle,
        'width_px': width_px,
        'height_px': height_px,
        'scale': scale_mm_per_px,
        'corners': box,
        'contour': main_contour
    }


if __name__ == "__main__":
    # ========== НАСТРОЙКИ ==========
    # Укажите реальную ширину вашего объекта в миллиметрах
    # Например, если объект шириной 50 мм:
    REAL_OBJECT_WIDTH_MM = 23.0
    
    # Если реальный размер неизвестен, поставьте None:
    # REAL_OBJECT_WIDTH_MM = None
    # =================================
    
    # Создаем папку для входных изображений, если её нет
    input_dir = "photos_in"
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
        print(f"Создана папка '{input_dir}'. Поместите туда изображения для обработки.")
        exit()
    
    # Получаем список всех изображений
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    images = [f for f in os.listdir(input_dir) if f.lower().endswith(image_extensions)]
    
    if not images:
        print(f"В папке '{input_dir}' нет изображений для обработки.")
        print(f"Поддерживаемые форматы: {', '.join(image_extensions)}")
        exit()
    
    print(f"Найдено изображений: {len(images)}")
    print(f"Реальная ширина объекта: {REAL_OBJECT_WIDTH_MM if REAL_OBJECT_WIDTH_MM else 'не задана (расчет в пикселях)'}")
    print("-" * 50)
    
    # Обрабатываем каждое изображение
    for filename in images:
        image_path = os.path.join(input_dir, filename)
        print(f"\nОбработка: {filename}")
        result = detect_contours_with_localization(image_path, REAL_OBJECT_WIDTH_MM)
        
        if result:
            print(f"  Центр: ({result['center'][0]:.1f}, {result['center'][1]:.1f}) px")
            print(f"  Размер: {result['width_px']:.1f} x {result['height_px']:.1f} px")
            print(f"  Угол поворота: {result['angle']:.1f}°")
    
    print("\n" + "=" * 50)
    print("Обработка завершена!")
    print(f"Результаты сохранены в папку 'photos_out'")