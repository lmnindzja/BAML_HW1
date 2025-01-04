# metrics/Dockerfile

FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y netcat-openbsd && apt-get clean

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./src /app
COPY wait-for-rabbitmq.sh ./wait-for-rabbitmq.sh

RUN chmod +x wait-for-rabbitmq.sh

CMD ["./wait-for-rabbitmq.sh", "rabbitmq", "python", "metrics.py"]
