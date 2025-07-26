"""
Конфигурация для Telegram Sentiment Bot
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Загрузка переменных окружения
load_dotenv()

# Пути проекта
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Создание директорий, если не существуют
for directory in [DATA_DIR, MODELS_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Telegram Bot
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# Модель
MODEL_CONFIG = {
    # Параметры текста
    "max_sequence_length": 128,
    "max_features": 20000,
    "embedding_dim": 300,
    
    # Архитектура модели
    "lstm_units": 128,
    "dropout_rate": 0.5,
    "recurrent_dropout": 0.2,
    
    # Обучение
    "batch_size": 64,
    "epochs": 100,  # Много эпох для высокой точности
    "validation_split": 0.2,
    "learning_rate": 0.001,
    
    # Классы
    "num_classes": 3,
    "class_names": ["Негативный", "Нейтральный", "Позитивный"],
    "class_labels": {
        "negative": 0,
        "neutral": 1,
        "positive": 2
    }
}

# Предобработка текста
TEXT_PREPROCESSING = {
    "lowercase": True,
    "remove_punctuation": True,
    "remove_numbers": True,
    "remove_urls": True,
    "remove_emails": True,
    "remove_emoji": False,  # Эмодзи могут быть важны для sentiment
    "lemmatize": True,
    "remove_stopwords": True
}

# Пути к моделям
MODEL_PATH = MODELS_DIR / "sentiment_model"
TOKENIZER_PATH = MODELS_DIR / "tokenizer.pkl"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"

# Логирование
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "datefmt": "%Y-%m-%d %H:%M:%S"
}