# plot/Dockerfile

# Используем официальный образ Python в качестве базового
FROM python:3.9-slim

# Устанавливаем рабочую директорию внутри контейнера
WORKDIR /app

# Копируем файлы зависимостей
COPY requirements.txt .

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /app/logs

RUN touch /app/logs/metric_log.csv

# Копируем скрипт plot.py
COPY ./src /app

# Определяем команду запуска контейнера
CMD ["python", "plot.py"]
