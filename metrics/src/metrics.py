import pika
import json
import os
import csv

# Путь к директории логов и файлу
LOG_DIR = 'logs'
LOG_FILE = os.path.join(LOG_DIR, 'metric_log.csv')

# Убедимся, что директория logs существует
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
    print(f"Создана директория {LOG_DIR}")

# Инициализация CSV-файла с заголовком, если он не существует
if not os.path.isfile(LOG_FILE):
    with open(LOG_FILE, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['id', 'y_true', 'y_pred', 'absolute_error'])
    print(f"Создан файл {LOG_FILE} с заголовком.")

# Создаём подключение к серверу на локальном хосте
connection = pika.BlockingConnection(
    pika.ConnectionParameters(
        host='rabbitmq',
        credentials=pika.PlainCredentials('user', 'password')  # Убедитесь, что здесь правильные учетные данные
    )
)
channel = connection.channel()

# Объявляем очереди y_true и y_pred
channel.queue_declare(queue='y_true')
channel.queue_declare(queue='y_pred')

# Хранилище для сопоставления id и значений
data_store = {}

# Функция для записи метрик в CSV
def log_metrics(message_id, y_true, y_pred, absolute_error):
    with open(LOG_FILE, mode='a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([message_id, y_true, y_pred, absolute_error])
    print(f"Записано в CSV: id={message_id}, y_true={y_true}, y_pred={y_pred}, absolute_error={absolute_error}")

# Функция для обработки сообщений
def callback(ch, method, properties, body):
    try:
        # Десериализуем сообщение
        message = json.loads(body)
        message_id = message.get('id')
        value = message.get('body')
        
        if message_id is None or value is None:
            raise ValueError("Сообщение не содержит 'id' или 'body'.")
        
        # Определяем, из какой очереди пришло сообщение
        queue = method.routing_key
        
        if queue == 'y_true':
            data_store.setdefault(message_id, {})['y_true'] = value
            print(f"Получено y_true: {value} с id {message_id}")
        elif queue == 'y_pred':
            data_store.setdefault(message_id, {})['y_pred'] = value
            print(f"Получено y_pred: {value} с id {message_id}")
        
        # Проверяем, есть ли оба значения для данного id
        if 'y_true' in data_store[message_id] and 'y_pred' in data_store[message_id]:
            y_true = data_store[message_id]['y_true']
            y_pred = data_store[message_id]['y_pred']
            
            # Вычисляем абсолютную ошибку
            absolute_error = abs(y_true - y_pred)
            print(f"Для id {message_id}: y_true = {y_true}, y_pred = {y_pred}, ошибка = {absolute_error}")
            
            # Записываем метрики в CSV
            log_metrics(message_id, y_true, y_pred, absolute_error)
            
            # После обработки удаляем запись из data_store
            del data_store[message_id]
            
    except json.JSONDecodeError:
        print("Ошибка декодирования JSON.")
    except Exception as e:
        print(f"Ошибка при обработке сообщения: {e}")

# Подписываемся на обе очереди
channel.basic_consume(
    queue='y_true',
    on_message_callback=callback,
    auto_ack=True
)

channel.basic_consume(
    queue='y_pred',
    on_message_callback=callback,
    auto_ack=True
)

print('...Ожидание сообщений, для выхода нажмите CTRL+C')
channel.start_consuming()
