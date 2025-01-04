import pika
import numpy as np
import json
from sklearn.datasets import load_diabetes
import time
from datetime import datetime

np.random.seed(42)

# Загружаем датасет о диабете
X, y = load_diabetes(return_X_y=True)

# Подключение к серверу на локальном хосте:
connection = pika.BlockingConnection(
    pika.ConnectionParameters(
        host='rabbitmq',
        credentials=pika.PlainCredentials('user', 'password')  # Убедитесь, что здесь правильные учетные данные
    )
)
channel = connection.channel()

# Создаём очередь y_true
channel.queue_declare(queue='y_true')
# Создаём очередь features
channel.queue_declare(queue='features')

try:
    while True:
        # Формируем случайный индекс строки
        random_row = np.random.randint(0, X.shape[0])

        # Создаём уникальный идентификатор сообщения
        message_id = datetime.timestamp(datetime.now())

        # Формируем сообщение для y_true с уникальным ID
        message_y_true = {
            'id': message_id,
            'body': y[random_row]
        }

        # Формируем сообщение для features с тем же уникальным ID
        message_features = {
            'id': message_id,
            'body': list(X[random_row])
        }

        # Публикуем сообщение в очередь y_true
        channel.basic_publish(
            exchange='',
            routing_key='y_true',
            body=json.dumps(message_y_true)
        )
        print(f'Сообщение с правильным ответом отправлено в очередь y_true: {message_y_true}')

        # Публикуем сообщение в очередь features
        channel.basic_publish(
            exchange='',
            routing_key='features',
            body=json.dumps(message_features)
        )
        print(f'Сообщение с вектором признаков отправлено в очередь features: {message_features}')

        # Задержка перед следующей итерацией (например, 2 секунды)
        time.sleep(2)

except KeyboardInterrupt:
    print("Остановка отправки сообщений пользователем.")

finally:
    # Закрываем подключение
    connection.close()
    print("Соединение закрыто.")
