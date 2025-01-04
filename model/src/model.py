import pika
import pickle
import numpy as np
import json

# Читаем файл с сериализованной моделью
with open('myfile.pkl', 'rb') as pkl_file:
    regressor = pickle.load(pkl_file)

# Создаём подключение к серверу на локальном хосте:
connection = pika.BlockingConnection(
    pika.ConnectionParameters(
        host='rabbitmq',
        credentials=pika.PlainCredentials('user', 'password')  # Убедитесь, что здесь правильные учетные данные
    )
)
channel = connection.channel()

# Объявляем очереди features и y_pred
channel.queue_declare(queue='features')
channel.queue_declare(queue='y_pred')  # Объявление очереди y_pred

# Создаём функцию callback для обработки данных из очереди features
def callback(ch, method, properties, body):
    print(f'Получен вектор признаков {body}')
    
    try:
        # Десериализуем сообщение с помощью json.loads
        message = json.loads(body)
        
        # Извлекаем id и тело сообщения
        message_id = message.get('id')
        features = message.get('body')
        
        if message_id is None or features is None:
            raise ValueError("Сообщение не содержит 'id' или 'body'.")
        
        # Преобразуем список признаков в numpy-массив нужной размерности
        shaped_features = np.array(features).reshape(1, -1)
        
        # Выполняем предсказание с использованием модели regressor
        prediction = regressor.predict(shaped_features)[0]
        
        # Округляем предсказание до целого числа
        prediction_rounded = int(round(prediction))
        
        # Формируем сообщение с предсказанием и тем же id
        prediction_message = {
            'id': message_id,
            'body': prediction_rounded
        }
        
        # Сериализуем предсказание с помощью json.dumps
        prediction_json = json.dumps(prediction_message)
        
        # Отправляем предсказание в очередь y_pred
        channel.basic_publish(
            exchange='',
            routing_key='y_pred',
            body=prediction_json
        )
        print(f'Предсказание {prediction_rounded} отправлено в очередь y_pred с id {message_id}')
        
    except json.JSONDecodeError:
        print("Ошибка декодирования JSON.")
    except Exception as e:
        print(f"Ошибка при обработке сообщения: {e}")

# Извлекаем сообщение из очереди features
# on_message_callback показывает, какую функцию вызвать при получении сообщения
channel.basic_consume(
    queue='features',
    on_message_callback=callback,
    auto_ack=True  # Осталось auto_ack=True, если требуется ручное подтверждение, измените на False и добавьте ch.basic_ack(...)
)
print('...Ожидание сообщений, для выхода нажмите CTRL+C')

# Запускаем режим ожидания прихода сообщений
channel.start_consuming()
