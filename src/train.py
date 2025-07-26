"""
Скрипт для обучения модели анализа настроений
"""
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import seaborn as sns

from config import (
    PROCESSED_DATA_DIR, MODEL_CONFIG, TOKENIZER_PATH, 
    LABEL_ENCODER_PATH, MODEL_PATH
)
from model import SentimentModel
from data_loader import DataLoader


class ModelTrainer:
    """Класс для обучения модели"""
    
    def __init__(self):
        self.tokenizer = Tokenizer(num_words=MODEL_CONFIG["max_features"])
        self.model = None
        self.label_encoder = None
        
    def load_data(self):
        """Загрузка подготовленных данных"""
        train_path = PROCESSED_DATA_DIR / "train_data.csv"
        test_path = PROCESSED_DATA_DIR / "test_data.csv"
        
        if not train_path.exists() or not test_path.exists():
            print("Данные не найдены. Запуск подготовки данных...")
            loader = DataLoader()
            train_df, test_df = loader.prepare_rusentiment_data()
            loader.save_processed_data(train_df, test_df)
        
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        # Загрузка label encoder
        self.label_encoder = joblib.load(PROCESSED_DATA_DIR / "label_encoder.pkl")
        
        return train_df, test_df
    
    def prepare_sequences(self, texts):
        """Преобразование текстов в последовательности"""
        sequences = self.tokenizer.texts_to_sequences(texts)
        return pad_sequences(
            sequences, 
            maxlen=MODEL_CONFIG["max_sequence_length"],
            padding='post',
            truncating='post'
        )
    
    def train_model(self):
        """Основная функция обучения"""
        # Загрузка данных
        print("Загрузка данных...")
        train_df, test_df = self.load_data()
        
        # Подготовка текстов
        print("Подготовка последовательностей...")
        self.tokenizer.fit_on_texts(train_df['processed_text'])
        
        X_train = self.prepare_sequences(train_df['processed_text'])
        X_test = self.prepare_sequences(test_df['processed_text'])
        
        y_train = train_df['label_encoded'].values
        y_test = test_df['label_encoded'].values
        
        # Разделение на train/validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, 
            test_size=0.15, 
            random_state=42,
            stratify=y_train
        )
        
        print(f"\nРазмеры данных:")
        print(f"Обучающая выборка: {X_train.shape}")
        print(f"Валидационная выборка: {X_val.shape}")
        print(f"Тестовая выборка: {X_test.shape}")
        print(f"Размер словаря: {len(self.tokenizer.word_index)}")
        
        # Создание и обучение модели
        print("\nСоздание модели...")
        vocab_size = min(len(self.tokenizer.word_index) + 1, MODEL_CONFIG["max_features"])
        
        sentiment_model = SentimentModel(vocab_size=vocab_size)
        sentiment_model.build_model()
        
        print(sentiment_model.model.summary())
        
        # Обучение
        print("\nНачало обучения...")
        history = sentiment_model.train(X_train, y_train, X_val, y_val)
        
        # Оценка на тестовой выборке
        print("\nОценка на тестовой выборке:")
        metrics = sentiment_model.evaluate(X_test, y_test)
        
        # Сохранение модели и токенизатора
        self.save_artifacts(sentiment_model)
        
        # Визуализация обучения
        self.plot_training_history(history)
        
        return sentiment_model, history, metrics
    
    def save_artifacts(self, sentiment_model):
        """Сохранение всех артефактов"""
        # Сохранение модели
        sentiment_model.save_model()
        
        # Сохранение токенизатора
        joblib.dump(self.tokenizer, TOKENIZER_PATH)
        
        # Сохранение label encoder
        joblib.dump(self.label_encoder, LABEL_ENCODER_PATH)
        
        print(f"\nАртефакты сохранены:")
        print(f"- Модель: {MODEL_PATH}")
        print(f"- Токенизатор: {TOKENIZER_PATH}")
        print(f"- Label Encoder: {LABEL_ENCODER_PATH}")
    
    def plot_training_history(self, history):
        """Визуализация процесса обучения"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # График точности
        ax1.plot(history.history['accuracy'], label='Train Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # График потерь
        ax2.plot(history.history['loss'], label='Train Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(MODEL_PATH / 'training_history.png')
        plt.show()
    
    def test_predictions(self, sentiment_model, num_samples=10):
        """Тестирование предсказаний на примерах"""
        test_texts = [
            "Это просто великолепно! Я в восторге!",
            "Ужасный сервис, никому не рекомендую",
            "Нормально, ничего особенного",
            "Отвратительно! Худший опыт в моей жизни",
            "Супер! Очень доволен покупкой",
            "Обычный продукт, как и все остальные",
            "Не понравилось, ожидал большего",
            "Восхитительно! Превзошло все ожидания",
            "Средненько, есть и лучше",
            "Кошмар! Деньги на ветер"
        ]
        
        # Подготовка текстов
        sequences = self.tokenizer.texts_to_sequences(test_texts)
        X_test = pad_sequences(
            sequences, 
            maxlen=MODEL_CONFIG["max_sequence_length"]
        )
        
        # Предсказания
        predictions = sentiment_model.model.predict(X_test)
        predicted_classes = np.argmax(predictions, axis=1)
        
        print("\nТестовые предсказания:")
        print("-" * 80)
        
        for text, pred_class, probs in zip(test_texts, predicted_classes, predictions):
            sentiment = self.label_encoder.inverse_transform([pred_class])[0]
            confidence = probs[pred_class] * 100
            
            print(f"Текст: {text}")
            print(f"Настроение: {sentiment} (уверенность: {confidence:.1f}%)")
            print(f"Вероятности: Негатив={probs[0]:.3f}, "
                  f"Нейтрал={probs[1]:.3f}, Позитив={probs[2]:.3f}")
            print("-" * 80)


def main():
    """Основная функция"""
    trainer = ModelTrainer()
    
    # Обучение модели
    model, history, metrics = trainer.train_model()
    
    # Тестирование предсказаний
    trainer.test_predictions(model)
    
    print("\n✅ Обучение завершено успешно!")
    print(f"Финальная точность на тесте: {metrics['accuracy']:.4f}")


if __name__ == "__main__":
    main()