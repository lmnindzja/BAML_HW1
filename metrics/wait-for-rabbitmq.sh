#!/bin/bash
set -e

host="$1"
shift
cmd="$@"

echo "Ожидание готовности RabbitMQ на хосте $host:5672..."

until nc -z "$host" 5672; do
  >&2 echo "RabbitMQ недоступен - ожидаем..."
  sleep 1
done

>&2 echo "RabbitMQ доступен - запускаем команду: $cmd"
exec $cmd
