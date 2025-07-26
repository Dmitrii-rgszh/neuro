# Dockerfile для Telegram Sentiment Bot (опционально)
FROM python:3.10-slim

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Рабочая директория
WORKDIR /app

# Копирование зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Загрузка NLTK данных
RUN python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

# Копирование проекта
COPY . .

# Создание структуры директорий
RUN python setup_project.py

# Команда по умолчанию
CMD ["python", "src/bot.py"]