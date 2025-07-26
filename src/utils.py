"""
Вспомогательные функции для проекта
"""
import re
import json
import os
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import pandas as pd
from pathlib import Path

from config import MODELS_DIR


def clean_text(text: str) -> str:
    """
    Очистка текста от специальных символов
    
    Args:
        text: Исходный текст
        
    Returns:
        Очищенный текст
    """
    # Удаление HTML тегов
    text = re.sub(r'<[^>]+>', '', text)
    
    # Удаление множественных пробелов
    text = re.sub(r'\s+', ' ', text)
    
    # Удаление специальных символов (сохраняем эмодзи)
    text = re.sub(r'[^\w\s\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF.,!?-]', '', text)
    
    return text.strip()


def extract_emojis(text: str) -> List[str]:
    """
    Извлечение эмодзи из текста
    
    Args:
        text: Текст для анализа
        
    Returns:
        Список найденных эмодзи
    """
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE
    )
    
    return emoji_pattern.findall(text)


def save_training_report(history: Dict, metrics: Dict, save_path: Path = None):
    """
    Сохранение отчета об обучении модели
    
    Args:
        history: История обучения
        metrics: Метрики модели
        save_path: Путь для сохранения
    """
    if save_path is None:
        save_path = MODELS_DIR / f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    report = {
        "training_date": datetime.now().isoformat(),
        "final_metrics": metrics,
        "training_history": {
            "loss": history.history.get('loss', []),
            "accuracy": history.history.get('accuracy', []),
            "val_loss": history.history.get('val_loss', []),
            "val_accuracy": history.history.get('val_accuracy', [])
        },
        "epochs_trained": len(history.history.get('loss', [])),
        "best_val_accuracy": max(history.history.get('val_accuracy', [0])),
        "best_val_loss": min(history.history.get('val_loss', [float('inf')]))
    }
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"Отчет сохранен: {save_path}")


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                         class_names: List[str], save_path: Path = None):
    """
    Построение матрицы ошибок
    
    Args:
        y_true: Истинные метки
        y_pred: Предсказанные метки
        class_names: Названия классов
        save_path: Путь для сохранения
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Матрица ошибок')
    plt.ylabel('Истинные метки')
    plt.xlabel('Предсказанные метки')
    
    if save_path:
        plt.savefig(save_path)
    plt.show()


def analyze_dataset_balance(df: pd.DataFrame, label_column: str = 'sentiment'):
    """
    Анализ баланса классов в датасете
    
    Args:
        df: DataFrame с данными
        label_column: Название столбца с метками
    """
    # Подсчет классов
    class_counts = df[label_column].value_counts()
    
    # Визуализация
    plt.figure(figsize=(10, 6))
    
    # График распределения
    ax1 = plt.subplot(1, 2, 1)
    class_counts.plot(kind='bar', color=['red', 'gray', 'green'])
    plt.title('Распределение классов')
    plt.xlabel('Класс')
    plt.ylabel('Количество')
    plt.xticks(rotation=45)
    
    # Круговая диаграмма
    ax2 = plt.subplot(1, 2, 2)
    class_counts.plot(kind='pie', autopct='%1.1f%%', 
                     colors=['red', 'gray', 'green'])
    plt.title('Процентное соотношение')
    plt.ylabel('')
    
    plt.tight_layout()
    plt.show()
    
    # Статистика
    print("Статистика по классам:")
    print(f"Всего примеров: {len(df)}")
    print("\nРаспределение:")
    for class_name, count in class_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {class_name}: {count} ({percentage:.1f}%)")
    
    # Проверка баланса
    max_count = class_counts.max()
    min_count = class_counts.min()
    imbalance_ratio = max_count / min_count
    
    print(f"\nСоотношение дисбаланса: {imbalance_ratio:.2f}")
    if imbalance_ratio > 2:
        print("⚠️  Датасет несбалансирован! Рекомендуется балансировка.")


def export_model_to_onnx(model, save_path: Path = None):
    """
    Экспорт модели в формат ONNX для использования в других приложениях
    
    Args:
        model: Модель Keras
        save_path: Путь для сохранения
    """
    try:
        import tf2onnx
        
        if save_path is None:
            save_path = MODELS_DIR / "sentiment_model.onnx"
        
        # Конвертация
        onnx_model, _ = tf2onnx.convert.from_keras(model)
        
        # Сохранение
        with open(save_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        
        print(f"Модель экспортирована в ONNX: {save_path}")
        
    except ImportError:
        print("Для экспорта в ONNX установите tf2onnx: pip install tf2onnx")


def create_sample_dataset(num_samples: int = 1000) -> pd.DataFrame:
    """
    Создание примерного датасета для тестирования
    
    Args:
        num_samples: Количество примеров
        
    Returns:
        DataFrame с примерами
    """
    positive_phrases = [
        "Отличный", "Прекрасный", "Замечательный", "Супер", "Великолепно",
        "Рад", "Счастлив", "Доволен", "Восхитительно", "Круто"
    ]
    
    negative_phrases = [
        "Ужасный", "Плохой", "Отвратительный", "Разочарован", "Грустно",
        "Печально", "Злой", "Раздражен", "Недоволен", "Кошмар"
    ]
    
    neutral_phrases = [
        "Нормально", "Обычно", "Средне", "Так себе", "Пойдет",
        "Ничего особенного", "Как всегда", "Стандартно", "Типично", "Обыденно"
    ]
    
    data = []
    
    for _ in range(num_samples // 3):
        # Позитивные
        text = f"{np.random.choice(positive_phrases)} день! Все {np.random.choice(positive_phrases).lower()}!"
        data.append({"text": text, "sentiment": "positive"})
        
        # Негативные
        text = f"{np.random.choice(negative_phrases)} опыт. Очень {np.random.choice(negative_phrases).lower()}."
        data.append({"text": text, "sentiment": "negative"})
        
        # Нейтральные
        text = f"{np.random.choice(neutral_phrases)}. Все {np.random.choice(neutral_phrases).lower()}."
        data.append({"text": text, "sentiment": "neutral"})
    
    return pd.DataFrame(data)


def calculate_model_size(model_path: Path) -> Dict[str, Any]:
    """
    Расчет размера модели и её характеристик
    
    Args:
        model_path: Путь к модели
        
    Returns:
        Словарь с информацией о модели
    """
    from tensorflow import keras
    
    model = keras.models.load_model(model_path)
    
    # Размер файла
    file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    
    # Количество параметров
    total_params = model.count_params()
    trainable_params = sum([np.prod(var.shape) for var in model.trainable_variables])
    non_trainable_params = total_params - trainable_params
    
    # Информация о слоях
    layer_info = []
    for layer in model.layers:
        layer_info.append({
            "name": layer.name,
            "type": layer.__class__.__name__,
            "params": layer.count_params(),
            "output_shape": str(layer.output_shape)
        })
    
    return {
        "file_size_mb": round(file_size_mb, 2),
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable_params,
        "num_layers": len(model.layers),
        "layer_info": layer_info
    }


def benchmark_prediction_speed(model, tokenizer, test_texts: List[str], num_runs: int = 100):
    """
    Тестирование скорости предсказаний
    
    Args:
        model: Модель для тестирования
        tokenizer: Токенизатор
        test_texts: Тексты для тестирования
        num_runs: Количество запусков
        
    Returns:
        Статистика по времени выполнения
    """
    import time
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    
    times = []
    
    for _ in range(num_runs):
        start_time = time.time()
        
        # Предобработка
        sequences = tokenizer.texts_to_sequences(test_texts)
        X = pad_sequences(sequences, maxlen=128)
        
        # Предсказание
        predictions = model.predict(X, verbose=0)
        
        end_time = time.time()
        times.append(end_time - start_time)
    
    times = np.array(times)
    
    return {
        "mean_time_ms": np.mean(times) * 1000,
        "std_time_ms": np.std(times) * 1000,
        "min_time_ms": np.min(times) * 1000,
        "max_time_ms": np.max(times) * 1000,
        "texts_per_second": len(test_texts) / np.mean(times)
    }


# Пример использования
if __name__ == "__main__":
    # Тест функций
    test_text = "Привет! 😊 Сегодня отличный день!"
    print(f"Исходный текст: {test_text}")
    print(f"Очищенный текст: {clean_text(test_text)}")
    print(f"Найденные эмодзи: {extract_emojis(test_text)}")
    
    # Создание примерного датасета
    sample_df = create_sample_dataset(100)
    print(f"\nСоздан примерный датасет с {len(sample_df)} примерами")
    
    # Анализ баланса
    analyze_dataset_balance(sample_df)