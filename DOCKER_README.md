# Docker Setup for CT Classification System

## Обзор

Этот проект поддерживает несколько способов запуска через Docker:

1. **Раздельные сервисы** - бэкенд и фронтенд в отдельных контейнерах
2. **Fullstack** - всё в одном контейнере с nginx и supervisor

## Быстрый старт

### Раздельные сервисы (рекомендуется для разработки)

```bash
# Запуск бэкенда и фронтенда отдельно
make docker-up

# Бэкенд будет доступен по адресу: http://localhost:8000
# Фронтенд будет доступен по адресу: http://localhost:3000
```

### Fullstack (рекомендуется для продакшена)

```bash
# Запуск всего в одном контейнере
make docker-up-fullstack

# Приложение будет доступно по адресу: http://localhost
# API доступно через: http://localhost/api/
```

## Полезные команды

```bash
# Остановить все сервисы
make docker-down

# Просмотр логов
make docker-logs

# Пересборка с нуля
make docker-rebuild

# Очистка Docker ресурсов
make docker-clean
```

## Структура Docker

### Multi-stage Dockerfile

Dockerfile содержит 5 стадий:

1. **backend-builder** - сборка зависимостей бэкенда с Poetry
2. **frontend-builder** - сборка React приложения
3. **backend** - runtime контейнер для бэкенда
4. **frontend** - nginx контейнер для фронтенда
5. **fullstack** - объединённый контейнер с nginx + supervisor

### Volumes

Следующие директории монтируются для персистентности данных:

- `./data:/app/data` - данные приложения
- `./logs:/app/logs` - логи
- `./cache:/app/cache` - кэш
- `./checkpoints:/app/checkpoints` - модели ML
- `./configs:/app/configs` - конфигурации

## Конфигурация

### nginx.conf
Конфигурация nginx для standalone фронтенда с поддержкой React Router.

### nginx-fullstack.conf
Конфигурация nginx для fullstack режима с проксированием API запросов на бэкенд.

### supervisord.conf
Конфигурация supervisor для управления nginx и uvicorn в fullstack режиме.

## Порты

- **3000** - Frontend (раздельный режим)
- **8000** - Backend API
- **80** - Fullstack приложение
- **8001** - Backend API в fullstack режиме (альтернативный доступ)

## Healthcheck

Бэкенд контейнер включает healthcheck, который проверяет доступность `/health` endpoint каждые 30 секунд.

## Troubleshooting

### Контейнер не запускается
```bash
# Проверить логи
docker-compose logs backend
docker-compose logs frontend
```

### Проблемы с зависимостями
```bash
# Пересобрать без кэша
make docker-rebuild
```

### Очистка всех ресурсов
```bash
# Полная очистка
make docker-clean
docker system prune -a
```
