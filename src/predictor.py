"""
Класс для выполнения предсказаний настроений
"""
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from typing import List, Dict, Union

from config import MODEL_PATH, TOKENIZER_PATH, LABEL_ENCODER_PATH, MODEL_CONFIG
from data_loader import DataLoader


class SentimentPredictor:
    """Класс для предсказания настроений текстов"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.label_encoder = None
        self.data_loader = DataLoader()
        self._load_artifacts()
    
    def _load_artifacts(self):
        """Загрузка модели и артефактов"""
        try:
            # Загрузка модели
            model_files = list(MODEL_PATH.glob("*.h5"))
            if model_files:
                # Берем последнюю модель
                latest_model = sorted(model_files)[-1]
                self.model = load_model(latest_model)
                print(f"Модель загружена: {latest_model}")
            else:
                raise FileNotFoundError("Модель не найдена")
            
            # Загрузка токенизатора
            self.tokenizer = joblib.load(TOKENIZER_PATH)
            print(f"Токенизатор загружен")
            
            # Загрузка label encoder
            self.label_encoder = joblib.load(LABEL_ENCODER_PATH)
            print(f"Label encoder загружен")
            
        except Exception as e:
            print(f"Ошибка при загрузке артефактов: {e}")
            print("Необходимо сначала обучить модель (запустите train.py)")
            raise
    
    def predict(self, text: Union[str, List[str]]) -> Union[Dict, List[Dict]]:
        """
        Предсказание настроения для текста или списка текстов
        
        Args:
            text: Строка или список строк для анализа
            
        Returns:
            Словарь или список словарей с предсказаниями
        """
        # Преобразование в список для единообразной обработки
        if isinstance(text, str):
            texts = [text]
            single_text = True
        else:
            texts = text
            single_text = False
        
        # Предобработка текстов
        processed_texts = [self.data_loader.preprocess_text(t) for t in texts]
        
        # Преобразование в последовательности
        sequences = self.tokenizer.texts_to_sequences(processed_texts)
        X = pad_sequences(
            sequences, 
            maxlen=MODEL_CONFIG["max_sequence_length"],
            padding='post',
            truncating='post'
        )
        
        # Предсказание
        predictions = self.model.predict(X, verbose=0)
        
        # Формирование результатов
        results = []
        for i, (orig_text, probs) in enumerate(zip(texts, predictions)):
            # Определение класса
            predicted_class = np.argmax(probs)
            sentiment = self.label_encoder.inverse_transform([predicted_class])[0]
            
            # Уверенность предсказания
            confidence = float(probs[predicted_class])
            
            # Детальные вероятности
            probabilities = {}
            for idx, label in enumerate(self.label_encoder.classes_):
                probabilities[label] = float(probs[idx])
            
            # Эмодзи для визуализации
            emoji_map = {
                'negative': '😢',
                'neutral': '😐',
                'positive': '😊'
            }
            
            result = {
                'text': orig_text,
                'sentiment': sentiment,
                'sentiment_ru': self._translate_sentiment(sentiment),
                'emoji': emoji_map.get(sentiment, ''),
                'confidence': confidence,
                'probabilities': probabilities,
                'processed_text': processed_texts[i]
            }
            
            results.append(result)
        
        return results[0] if single_text else results
    
    def _translate_sentiment(self, sentiment: str) -> str:
        """Перевод настроения на русский"""
        translations = {
            'negative': 'Негативное',
            'neutral': 'Нейтральное',
            'positive': 'Позитивное'
        }
        return translations.get(sentiment, sentiment)
    
    def predict_batch(self, texts: List[str], batch_size: int = 32) -> List[Dict]:
        """
        Предсказание для большого количества текстов с батчингом
        
        Args:
            texts: Список текстов
            batch_size: Размер батча
            
        Returns:
            Список результатов предсказаний
        """
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_results = self.predict(batch)
            results.extend(batch_results)
        
        return results
    
    def analyze_sentiment_distribution(self, texts: List[str]) -> Dict:
        """
        Анализ распределения настроений в списке текстов
        
        Args:
            texts: Список текстов для анализа
            
        Returns:
            Статистика по настроениям
        """
        predictions = self.predict_batch(texts)
        
        # Подсчет статистики
        sentiment_counts = {'negative': 0, 'neutral': 0, 'positive': 0}
        total_confidence = {'negative': 0, 'neutral': 0, 'positive': 0}
        
        for pred in predictions:
            sentiment = pred['sentiment']
            sentiment_counts[sentiment] += 1
            total_confidence[sentiment] += pred['confidence']
        
        # Расчет процентов и средней уверенности
        total = len(texts)
        stats = {}
        
        for sentiment in sentiment_counts:
            count = sentiment_counts[sentiment]
            percentage = (count / total * 100) if total > 0 else 0
            avg_confidence = (total_confidence[sentiment] / count * 100) if count > 0 else 0
            
            stats[sentiment] = {
                'count': count,
                'percentage': percentage,
                'average_confidence': avg_confidence
            }
        
        return {
            'total_texts': total,
            'sentiment_stats': stats,
            'dominant_sentiment': max(sentiment_counts, key=sentiment_counts.get)
        }
    
    def get_confidence_level(self, confidence: float) -> str:
        """
        Определение уровня уверенности предсказания
        
        Args:
            confidence: Значение уверенности (0-1)
            
        Returns:
            Текстовое описание уровня уверенности
        """
        if confidence >= 0.9:
            return "Очень высокая"
        elif confidence >= 0.7:
            return "Высокая"
        elif confidence >= 0.5:
            return "Средняя"
        else:
            return "Низкая"


# Пример использования
if __name__ == "__main__":
    predictor = SentimentPredictor()
    
    # Тестовые примеры
    test_texts = [
        "Отличный продукт, всем рекомендую!",
        "Ужасное качество, полное разочарование",
        "Нормально, но есть недостатки",
        "Просто восхитительно! Лучше не бывает!",
        "Не впечатлило, ожидал большего"
    ]
    
    print("\n=== Тестирование предсказаний ===\n")
    
    # Одиночное предсказание
    result = predictor.predict(test_texts[0])
    print(f"Текст: {result['text']}")
    print(f"Настроение: {result['sentiment_ru']} {result['emoji']}")
    print(f"Уверенность: {result['confidence']:.2%}")
    print(f"Уровень уверенности: {predictor.get_confidence_level(result['confidence'])}")
    print()
    
    # Батч предсказание
    print("\n=== Анализ нескольких текстов ===\n")
    results = predictor.predict(test_texts)
    
    for result in results:
        print(f"{result['emoji']} {result['text']}")
        print(f"   → {result['sentiment_ru']} (уверенность: {result['confidence']:.2%})")
    
    # Статистика
    print("\n=== Статистика настроений ===\n")
    stats = predictor.analyze_sentiment_distribution(test_texts)
    
    print(f"Всего текстов: {stats['total_texts']}")
    print(f"Доминирующее настроение: {predictor._translate_sentiment(stats['dominant_sentiment'])}")
    print("\nРаспределение:")
    
    for sentiment, data in stats['sentiment_stats'].items():
        print(f"  {predictor._translate_sentiment(sentiment)}: "
              f"{data['count']} ({data['percentage']:.1f}%), "
              f"средняя уверенность: {data['average_confidence']:.1f}%")