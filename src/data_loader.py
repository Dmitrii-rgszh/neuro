"""
Загрузка и подготовка данных для обучения модели
"""
import os
import re
import json
import pandas as pd
import numpy as np
from typing import Tuple, List
import requests
import zipfile
from tqdm import tqdm

import nltk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, TEXT_PREPROCESSING, LABEL_ENCODER_PATH

# Загрузка необходимых ресурсов NLTK
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
from nltk.corpus import stopwords

# Инициализация
try:
    russian_stopwords = set(stopwords.words('russian'))
except:
    print("Загрузка стоп-слов...")
    nltk.download('stopwords')
    russian_stopwords = set(stopwords.words('russian'))


class DataLoader:
    """Класс для загрузки и подготовки данных"""
    
    def __init__(self):
        self.label_encoder = LabelEncoder()
        
    def download_rusentiment(self):
        """Загрузка датасета RuSentiment"""
        # Альтернативные URL для датасета
        urls = [
            "http://text-machine.cs.uml.edu/projects/rusentiment/rusentiment_random_posts.csv",
            "https://raw.githubusercontent.com/text-machine-lab/rusentiment/master/Dataset/rusentiment_random_posts.csv"
        ]
        
        filepath = RAW_DATA_DIR / "rusentiment.csv"
        
        if not filepath.exists():
            print("Загрузка RuSentiment датасета...")
            
            for url in urls:
                try:
                    response = requests.get(url, timeout=30)
                    if response.status_code == 200:
                        with open(filepath, 'wb') as f:
                            f.write(response.content)
                        print("Датасет загружен!")
                        break
                except Exception as e:
                    print(f"Ошибка загрузки с {url}: {e}")
                    continue
            else:
                print("Не удалось загрузить датасет с указанных URL")
                return None
        
        return filepath
    
    def load_custom_dataset(self, filepath: str) -> pd.DataFrame:
        """Загрузка пользовательского датасета"""
        if filepath.endswith('.csv'):
            return pd.read_csv(filepath)
        elif filepath.endswith('.json'):
            return pd.read_json(filepath)
        else:
            raise ValueError("Поддерживаются только CSV и JSON файлы")
    
    def preprocess_text(self, text: str) -> str:
        """Предобработка текста"""
        if pd.isna(text):
            return ""
        
        text = str(text)
        
        # Приведение к нижнему регистру
        if TEXT_PREPROCESSING["lowercase"]:
            text = text.lower()
        
        # Удаление URL
        if TEXT_PREPROCESSING["remove_urls"]:
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Удаление email
        if TEXT_PREPROCESSING["remove_emails"]:
            text = re.sub(r'\S+@\S+', '', text)
        
        # Удаление чисел
        if TEXT_PREPROCESSING["remove_numbers"]:
            text = re.sub(r'\d+', '', text)
        
        # Удаление пунктуации (кроме эмодзи)
        if TEXT_PREPROCESSING["remove_punctuation"]:
            text = re.sub(r'[^\w\s\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]', ' ', text)
        
        # Токенизация
        try:
            tokens = nltk.word_tokenize(text)
        except:
            # Если не удается токенизировать, используем простое разделение
            tokens = text.split()
        
        # Простая нормализация вместо лемматизации
        if TEXT_PREPROCESSING["lemmatize"] and tokens:
            # Удаляем окончания для базовой нормализации
            normalized_tokens = []
            for token in tokens:
                # Простое удаление типичных русских окончаний
                if len(token) > 3:
                    # Удаляем окончания: -ов, -ев, -ий, -ый, -ая, -ое, -ые и т.д.
                    if token.endswith(('ов', 'ев', 'ий', 'ый', 'ая', 'ое', 'ые', 'ие', 
                                     'ого', 'его', 'ому', 'ему', 'ом', 'ем', 'ой', 'ей',
                                     'ую', 'юю', 'ая', 'яя', 'ое', 'ее', 'ые', 'ие')):
                        token = token[:-2]
                    elif token.endswith(('а', 'я', 'о', 'е', 'у', 'ю', 'ы', 'и')):
                        token = token[:-1]
                normalized_tokens.append(token)
            tokens = normalized_tokens
        
        # Удаление стоп-слов
        if TEXT_PREPROCESSING["remove_stopwords"]:
            tokens = [token for token in tokens if token not in russian_stopwords and len(token) > 2]
        
        return ' '.join(tokens)
    
    def map_sentiment_to_3_classes(self, sentiment: str) -> str:
        """Преобразование 5 классов sentiment в 3"""
        mapping = {
            'negative': 'negative',
            'positive': 'positive',
            'neutral': 'neutral',
            'speech': 'neutral',
            'skip': 'neutral'
        }
        return mapping.get(sentiment.lower(), 'neutral')
    
    def create_synthetic_dataset(self, num_samples: int = 10000) -> pd.DataFrame:
        """Создание синтетического датасета для обучения"""
        print("Создание синтетического датасета...")
        
        # Примеры позитивных фраз
        positive_templates = [
            "Это просто {adj}! Я очень {feeling}!",
            "{adj} продукт! Всем рекомендую!",
            "Супер! Очень {feeling} покупкой!",
            "Великолепно! {adj} качество!",
            "Отличный сервис! {feeling}!",
            "{adj}! Превзошло все ожидания!",
            "Замечательный опыт! Буду {action} снова!",
            "Лучшее, что я {action}! {adj}!",
            "Очень {feeling}! Спасибо за {adj} работу!",
            "Рекомендую всем! {adj} выбор!"
        ]
        positive_adj = ["великолепно", "замечательно", "отлично", "прекрасно", "восхитительно", 
                       "потрясающе", "превосходно", "изумительно", "чудесно", "роскошно"]
        positive_feeling = ["доволен", "рад", "счастлив", "восхищен", "впечатлен", 
                           "удовлетворен", "воодушевлен", "благодарен"]
        positive_action = ["покупать", "заказывать", "использовать", "брать", "пробовать"]
        
        # Примеры негативных фраз
        negative_templates = [
            "Это {adj}! Очень {feeling}!",
            "{adj} качество! Не рекомендую!",
            "Ужасно! {feeling} покупкой!",
            "{adj} сервис! Больше не буду {action}!",
            "Кошмар! {adj} опыт!",
            "Отвратительно! {feeling}!",
            "Худшее, что я {action}! {adj}!",
            "Полное разочарование! {adj} товар!",
            "Не советую никому! {feeling}!",
            "Деньги на ветер! {adj}!"
        ]
        negative_adj = ["ужасно", "отвратительно", "плохо", "кошмарно", "неприемлемо", 
                       "безобразно", "отвратительно", "никуда не годится", "провально"]
        negative_feeling = ["разочарован", "расстроен", "недоволен", "возмущен", 
                           "раздражен", "огорчен", "обманут", "зол"]
        negative_action = ["покупать", "заказывать", "связываться", "брать", "обращаться"]
        
        # Примеры нейтральных фраз
        neutral_templates = [
            "Обычный продукт, ничего особенного",
            "Нормально, как и ожидалось",
            "Средне, есть плюсы и минусы",
            "Пойдет, но можно найти лучше",
            "Стандартное качество за свои деньги",
            "Ничего выдающегося, обычный товар",
            "Соответствует описанию, без сюрпризов",
            "Неплохо, но есть к чему стремиться",
            "Обычное обслуживание, без излишеств",
            "Приемлемо для своей категории"
        ]
        
        data = []
        
        # Генерация позитивных примеров
        for _ in range(num_samples // 3):
            template = np.random.choice(positive_templates)
            text = template
            if '{adj}' in template:
                text = text.replace('{adj}', np.random.choice(positive_adj))
            if '{feeling}' in template:
                text = text.replace('{feeling}', np.random.choice(positive_feeling))
            if '{action}' in template:
                text = text.replace('{action}', np.random.choice(positive_action))
            data.append({"text": text, "label": "positive"})
        
        # Генерация негативных примеров
        for _ in range(num_samples // 3):
            template = np.random.choice(negative_templates)
            text = template
            if '{adj}' in template:
                text = text.replace('{adj}', np.random.choice(negative_adj))
            if '{feeling}' in template:
                text = text.replace('{feeling}', np.random.choice(negative_feeling))
            if '{action}' in template:
                text = text.replace('{action}', np.random.choice(negative_action))
            data.append({"text": text, "label": "negative"})
        
        # Генерация нейтральных примеров
        for _ in range(num_samples - 2 * (num_samples // 3)):
            text = np.random.choice(neutral_templates)
            data.append({"text": text, "label": "neutral"})
        
        # Перемешивание данных
        df = pd.DataFrame(data)
        df = df.sample(frac=1).reset_index(drop=True)
        
        print(f"Создан синтетический датасет с {len(df)} примерами")
        return df
    
    def prepare_rusentiment_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Подготовка данных RuSentiment"""
        try:
            filepath = self.download_rusentiment()
            
            if filepath is None:
                raise ValueError("Не удалось загрузить датасет")
            
            # Загрузка данных с разными кодировками
            encodings = ['utf-8', 'cp1251', 'latin1']
            df = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(filepath, encoding=encoding)
                    print(f"Датасет загружен с кодировкой {encoding}")
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                raise ValueError("Не удалось прочитать датасет")
            
            # Проверка структуры датасета
            print(f"Колонки в датасете: {df.columns.tolist()}")
            print(f"Размер датасета: {len(df)} строк")
            
            # Определение правильных колонок
            text_column = None
            label_column = None
            
            # Поиск колонки с текстом
            for col in ['text', 'Text', 'comment', 'review', 'message']:
                if col in df.columns:
                    text_column = col
                    break
            
            # Поиск колонки с метками
            for col in ['label', 'Label', 'sentiment', 'Sentiment', 'category']:
                if col in df.columns:
                    label_column = col
                    break
            
            if text_column is None or label_column is None:
                # Если стандартные имена не найдены, используем первые две колонки
                if len(df.columns) >= 2:
                    text_column = df.columns[0]
                    label_column = df.columns[1]
                    print(f"Используем колонки: текст='{text_column}', метка='{label_column}'")
                else:
                    raise ValueError("Не удалось определить структуру датасета")
            
            # Переименование для единообразия
            df = df.rename(columns={text_column: 'text', label_column: 'label'})
            
        except Exception as e:
            print(f"Ошибка при загрузке RuSentiment: {e}")
            print("Используем синтетический датасет для обучения...")
            df = self.create_synthetic_dataset(num_samples=15000)
        
        # Преобразование меток в строковый формат
        df['label'] = df['label'].astype(str).str.lower()
        
        # Преобразование в 3 класса
        df['sentiment_3'] = df['label'].apply(self.map_sentiment_to_3_classes)
        
        # Предобработка текста
        print("Предобработка текстов...")
        df['processed_text'] = df['text'].apply(self.preprocess_text)
        
        # Удаление пустых строк
        df = df[df['processed_text'].str.len() > 0]
        
        # Кодирование меток
        df['label_encoded'] = self.label_encoder.fit_transform(df['sentiment_3'])
        
        # Разделение на обучающую и тестовую выборки
        train_df, test_df = train_test_split(
            df[['processed_text', 'label_encoded', 'sentiment_3']], 
            test_size=0.2, 
            random_state=42,
            stratify=df['label_encoded']
        )
        
        return train_df, test_df
    
    def load_additional_datasets(self):
        """Загрузка дополнительных датасетов для улучшения модели"""
        datasets = []
        
        # Здесь можно добавить загрузку других датасетов
        # Например, Russian Twitter Corpus, отзывы с маркетплейсов и т.д.
        
        return datasets
    
    def save_processed_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        """Сохранение обработанных данных"""
        train_df.to_csv(PROCESSED_DATA_DIR / "train_data.csv", index=False)
        test_df.to_csv(PROCESSED_DATA_DIR / "test_data.csv", index=False)
        
        # Сохранение label encoder
        import joblib
        joblib.dump(self.label_encoder, PROCESSED_DATA_DIR / "label_encoder.pkl")
        
        # Также сохраним в основной директории моделей
        joblib.dump(self.label_encoder, LABEL_ENCODER_PATH)
        
        print(f"Данные сохранены в {PROCESSED_DATA_DIR}")
        print(f"Обучающая выборка: {len(train_df)} примеров")
        print(f"Тестовая выборка: {len(test_df)} примеров")
        print(f"Распределение классов в обучающей выборке:")
        print(train_df['sentiment_3'].value_counts())


if __name__ == "__main__":
    # Инициализация tqdm для pandas
    tqdm.pandas()
    
    # Загрузка и подготовка данных
    loader = DataLoader()
    train_df, test_df = loader.prepare_rusentiment_data()
    
    # Сохранение
    loader.save_processed_data(train_df, test_df)