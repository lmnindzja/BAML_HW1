# plot/src/plot.py

import time
import pandas as pd
import matplotlib.pyplot as plt
import os

# Путь к CSV-файлу с метриками
LOG_FILE = '/app/logs/metric_log.csv'  # Абсолютный путь внутри контейнера
# Путь к файлу для сохранения гистограммы
OUTPUT_PLOT = '/app/logs/error_distribution.png'  # Абсолютный путь внутри контейнера
# Интервал обновления графика в секундах
SLEEP_TIME = 5  # можно настроить по необходимости

def plot_error_distribution():
    while True:
        try:
            if os.path.exists(LOG_FILE):
                # Читаем CSV-файл
                df = pd.read_csv(LOG_FILE)
                errors = df['absolute_error']
                
                if not errors.empty:
                    # Создаём гистограмму
                    plt.figure(figsize=(10, 6))
                    plt.hist(errors, bins=20, edgecolor='black', color='skyblue')  # Настройте bins и color по необходимости
                    plt.title('Распределение Абсолютных Ошибок')  # Измените заголовок при необходимости
                    plt.xlabel('Абсолютная Ошибка')  # Измените подпись оси X при необходимости
                    plt.ylabel('Частота')  # Измените подпись оси Y при необходимости
                    
                    # Добавляем сетку для лучшей читаемости
                    plt.grid(True, linestyle='--', alpha=0.7)
                    
                    # Сохраняем график
                    plt.savefig(OUTPUT_PLOT)
                    plt.close()
                    print(f"Гистограмма обновлена и сохранена в {OUTPUT_PLOT}")
                else:
                    print("Нет данных для построения гистограммы.")
            else:
                print(f"Файл {LOG_FILE} не найден.")
        except Exception as e:
            print(f"Ошибка при построении графика: {e}")
        
        # Ждём перед следующим обновлением
        time.sleep(SLEEP_TIME)

if __name__ == "__main__":
    print("Сервис plot запущен и ожидает обновлений метрик...")
    plot_error_distribution()
