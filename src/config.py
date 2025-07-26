"""
Улучшенная конфигурация для высокоточного анализа настроений
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
LOGS_DIR = BASE_DIR / "logs"

# Создание директорий, если не существуют
for directory in [DATA_DIR, MODELS_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Telegram Bot
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# Источники данных
DATA_SOURCES = {
    "rusentiment": {
        "urls": [
            "http://text-machine.cs.uml.edu/projects/rusentiment/rusentiment_random_posts.csv",
            "https://raw.githubusercontent.com/text-machine-lab/rusentiment/master/Dataset/rusentiment_random_posts.csv"
        ],
        "enabled": True,
        "weight": 1.0  # Вес при объединении датасетов
    },
    "linis_crowd": {
        "url": "https://github.com/nicolay-r/RuSentRel/raw/master/data/linis_crowd.csv",
        "enabled": True,
        "weight": 1.0
    },
    "synthetic": {
        "enabled": True,
        "weight": 0.7,  # Меньший вес для синтетических данных
        "samples": 50000
    },
    "kaggle_russian": {
        "enabled": False,  # Требует API ключ
        "weight": 1.0
    }
}

# Улучшенная конфигурация модели для высокой точности
MODEL_CONFIG = {
    # Параметры текста - увеличены для лучшего понимания
    "max_sequence_length": 256,  # Увеличено для длинных текстов
    "max_features": 50000,       # Увеличен словарь
    "embedding_dim": 300,        # Размер эмбеддингов как в Word2Vec
    
    # Архитектура модели - более сложная для высокой точности
    "lstm_units": 256,           # Увеличены LSTM units
    "dense_units": [512, 256, 128], # Многослойная архитектура
    "dropout_rate": 0.4,         # Умеренный dropout
    "recurrent_dropout": 0.3,
    "spatial_dropout": 0.2,
    
    # Механизм внимания
    "attention_heads": 8,        # Multi-head attention
    "attention_key_dim": 64,
    
    # Обучение - настроено для высокой точности
    "batch_size": 32,            # Уменьшен для стабильности
    "epochs": 150,               # Много эпох для глубокого обучения
    "validation_split": 0.15,
    "learning_rate": 0.0005,     # Уменьшена для стабильности
    "learning_rate_schedule": True,  # Динамическое изменение LR
    
    # Регуляризация
    "l1_reg": 0.01,
    "l2_reg": 0.01,
    "class_weight": True,        # Автоматическая балансировка классов
    
    # Early stopping и callbacks
    "early_stopping_patience": 15,
    "reduce_lr_patience": 7,
    "reduce_lr_factor": 0.5,
    "min_lr": 1e-7,
    
    # Классы
    "num_classes": 3,
    "class_names": ["Негативный", "Нейтральный", "Позитивный"],
    "class_labels": {
        "negative": 0,
        "neutral": 1,
        "positive": 2
    },
    
    # Аугментация данных
    "data_augmentation": {
        "enabled": True,
        "synonym_replacement": 0.1,   # Замена синонимами
        "random_insertion": 0.1,      # Случайные вставки
        "random_swap": 0.1,           # Перестановка слов
        "random_deletion": 0.05       # Удаление слов
    },
    
    # Ансамбль моделей
    "ensemble": {
        "enabled": True,
        "models": ["lstm", "cnn_lstm", "transformer"],
        "voting": "soft"  # Мягкое голосование
    }
}

# Улучшенная предобработка текста
TEXT_PREPROCESSING = {
    "lowercase": True,
    "remove_punctuation": False,  # Пунктуация может быть важна
    "remove_numbers": False,      # Числа тоже могут нести смысл
    "remove_urls": True,
    "remove_emails": True,
    "remove_emoji": False,        # Эмодзи важны для sentiment
    "lemmatize": True,
    "remove_stopwords": False,    # Некоторые стоп-слова важны для тона
    "normalize_elongated": True,  # "оооочень" -> "очень"
    "fix_encoding": True,         # Исправление кодировки
    "spell_check": False,         # Опционально, требует pyspellchecker
    
    # Обработка специальных случаев
    "handle_negation": True,      # Учет отрицаний
    "expand_contractions": True,  # "не'll" -> "не will"
    "normalize_whitespace": True
}

# Конфигурация токенизатора
TOKENIZER_CONFIG = {
    "num_words": MODEL_CONFIG["max_features"],
    "oov_token": "<UNK>",
    "filters": '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',  # Сохраняем некоторые символы
    "lower": TEXT_PREPROCESSING["lowercase"],
    "split": ' ',
    "char_level": False
}

# Пути к моделям и артефактам
MODEL_PATHS = {
    "main_model": MODELS_DIR / "sentiment_model_v2.h5",
    "lstm_model": MODELS_DIR / "lstm_sentiment.h5",
    "cnn_lstm_model": MODELS_DIR / "cnn_lstm_sentiment.h5",
    "transformer_model": MODELS_DIR / "transformer_sentiment.h5",
    "ensemble_model": MODELS_DIR / "ensemble_sentiment.h5",
    "tokenizer": MODELS_DIR / "tokenizer_v2.pkl",
    "label_encoder": MODELS_DIR / "label_encoder_v2.pkl",
    "scaler": MODELS_DIR / "feature_scaler.pkl"
}

# Конфигурация валидации модели
VALIDATION_CONFIG = {
    "cross_validation_folds": 5,
    "test_size": 0.2,
    "validation_size": 0.15,
    "stratify": True,
    "random_state": 42,
    
    # Метрики для оценки
    "metrics": [
        "accuracy", "precision", "recall", "f1_score",
        "confusion_matrix", "classification_report",
        "roc_auc"  # Для каждого класса отдельно
    ],
    
    # Целевые показатели качества
    "target_accuracy": 0.90,     # Минимальная точность 90%
    "target_f1_score": 0.88,     # Минимальный F1-score
    "min_class_precision": 0.85  # Минимальная точность для каждого класса
}

# Конфигурация для тестирования в реальном времени
REALTIME_CONFIG = {
    "batch_prediction": True,
    "max_batch_size": 100,
    "prediction_timeout": 5.0,   # секунд
    "confidence_threshold": 0.7,  # Минимальная уверенность
    "fallback_prediction": "neutral",  # При низкой уверенности
    
    # Кэширование предсказаний
    "cache_predictions": True,
    "cache_size": 10000,
    "cache_ttl": 3600  # время жизни кэша в секундах
}

# Конфигурация логирования
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "datefmt": "%Y-%m-%d %H:%M:%S",
    "handlers": {
        "file": {
            "filename": LOGS_DIR / "sentiment_bot.log",
            "max_bytes": 10 * 1024 * 1024,  # 10MB
            "backup_count": 5
        },
        "console": {
            "level": "INFO"
        }
    }
}

# Конфигурация мониторинга
MONITORING_CONFIG = {
    "enable_metrics": True,
    "metrics_collection": {
        "prediction_latency": True,
        "prediction_accuracy": True,
        "user_feedback": True,
        "error_rate": True
    },
    
    # Алерты
    "alerts": {
        "accuracy_drop_threshold": 0.05,  # Падение точности на 5%
        "error_rate_threshold": 0.1,      # Доля ошибок выше 10%
        "latency_threshold": 1.0           # Задержка выше 1 секунды
    }
}

# Дополнительные библиотеки для улучшения точности
OPTIONAL_LIBRARIES = {
    "word2vec": {
        "enabled": False,  # Требует gensim
        "model_path": "ruwikiruscorpora_upos_skipgram_300_2_2018.vec.gz",
        "url": "http://vectors.nlpl.eu/repository/20/180.zip"
    },
    "fasttext": {
        "enabled": False,  # Требует fasttext
        "model_path": "cc.ru.300.bin",
        "url": "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ru.300.bin.gz"
    },
    "transformers": {
        "enabled": False,  # Требует transformers
        "model_name": "DeepPavlov/rubert-base-cased-sentiment",
        "use_for_ensemble": True
    }
}

# Конфигурация A/B тестирования
AB_TESTING_CONFIG = {
    "enabled": False,
    "models": {
        "model_a": MODEL_PATHS["main_model"],
        "model_b": MODEL_PATHS["ensemble_model"]
    },
    "traffic_split": 0.5,  # 50/50 между моделями
    "metrics_to_track": ["accuracy", "user_satisfaction", "response_time"]
}

# Экспорт основных путей для обратной совместимости
MODEL_PATH = MODEL_PATHS["main_model"]
TOKENIZER_PATH = MODEL_PATHS["tokenizer"]
LABEL_ENCODER_PATH = MODEL_PATHS["label_encoder"]