"""
Расширенная загрузка данных с множественными источниками
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
import gdown

import nltk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, TEXT_PREPROCESSING, LABEL_ENCODER_PATH

# Загрузка необходимых ресурсов NLTK
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
from nltk.corpus import stopwords

try:
    russian_stopwords = set(stopwords.words('russian'))
except:
    print("Загрузка стоп-слов...")
    nltk.download('stopwords')
    russian_stopwords = set(stopwords.words('russian'))


class EnhancedDataLoader:
    """Расширенный класс для загрузки данных из множественных источников"""
    
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.datasets = []
        
    def download_rusentiment(self):
        """Загрузка датасета RuSentiment"""
        urls = [
            "http://text-machine.cs.uml.edu/projects/rusentiment/rusentiment_random_posts.csv",
            "https://raw.githubusercontent.com/text-machine-lab/rusentiment/master/Dataset/rusentiment_random_posts.csv",
            "https://github.com/text-machine-lab/rusentiment/raw/master/Dataset/rusentiment_random_posts.csv"
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
                        print("✅ RuSentiment загружен!")
                        break
                except Exception as e:
                    print(f"❌ Ошибка загрузки с {url}: {e}")
                    continue
            else:
                print("⚠️ Не удалось загрузить RuSentiment")
                return None
        
        return filepath

    def download_russian_twitter_corpus(self):
        """Загрузка Russian Twitter Corpus"""
        # Заглушка - требует регистрации
        print("📱 Russian Twitter Corpus требует регистрации на сайте")
        print("   Доступен по адресу: http://study.mokoron.com/")
        return None

    def download_linis_crowd(self):
        """Загрузка LINIS Crowd dataset"""
        url = "https://github.com/nicolay-r/RuSentRel/raw/master/data/linis_crowd.csv"
        filepath = RAW_DATA_DIR / "linis_crowd.csv"
        
        if not filepath.exists():
            try:
                print("📊 Загрузка LINIS Crowd...")
                response = requests.get(url, timeout=30)
                if response.status_code == 200:
                    with open(filepath, 'wb') as f:
                        f.write(response.content)
                    print("✅ LINIS Crowd загружен!")
                    return filepath
            except Exception as e:
                print(f"❌ Ошибка загрузки LINIS Crowd: {e}")
        
        return filepath if filepath.exists() else None

    def download_kaggle_russian_news(self):
        """Загрузка русских новостей с Kaggle (требует API ключ)"""
        try:
            import kaggle
            filepath = RAW_DATA_DIR / "russian_news.csv"
            
            if not filepath.exists():
                print("📰 Загрузка русских новостей с Kaggle...")
                kaggle.api.dataset_download_files(
                    'blackmoon/russian-language-toxic-comments',
                    path=str(RAW_DATA_DIR),
                    unzip=True
                )
                print("✅ Kaggle dataset загружен!")
            
            return filepath if filepath.exists() else None
            
        except Exception as e:
            print(f"⚠️ Kaggle API недоступен: {e}")
            print("   Установите: pip install kaggle")
            print("   Настройте API ключ: https://github.com/Kaggle/kaggle-api")
            return None

    def load_rutweetcorp(self):
        """Загрузка RuTweetCorp (если доступен)"""
        filepath = RAW_DATA_DIR / "rutweetcorp.csv"
        
        if filepath.exists():
            try:
                df = pd.read_csv(filepath, encoding='utf-8')
                print(f"✅ RuTweetCorp загружен: {len(df)} записей")
                return df
            except Exception as e:
                print(f"❌ Ошибка загрузки RuTweetCorp: {e}")
        
        return None

    def create_extended_synthetic_dataset(self, num_samples: int = 50000) -> pd.DataFrame:
        """Создание расширенного синтетического датасета"""
        print(f"🔄 Создание расширенного синтетического датасета ({num_samples} примеров)...")
        
        # Расширенные шаблоны
        positive_patterns = {
            "reviews": [
                "Отличный {product}! Всем рекомендую купить.",
                "Превосходное качество {product}. Очень доволен покупкой.",
                "Лучший {product} который я когда-либо {action}!",
                "Супер {product}! Стоит каждой копейки.",
                "Великолепный {product}. Буду заказывать еще.",
                "Потрясающий {product}! Превзошел все ожидания.",
                "Замечательный {product}. Спасибо продавцу!",
                "Идеальный {product} для {purpose}. Рекомендую!",
                "Восхитительный {product}! Качество на высоте.",
                "Прекрасный {product}. Очень быстрая доставка."
            ],
            "emotions": [
                "Сегодня {feeling}! Все получается отлично.",
                "Какой {adj} день! Настроение {mood}.",
                "Очень {feeling} результатом. Все супер!",
                "Прекрасное {feeling}! Жизнь удалась.",
                "Сегодня особенно {feeling}. Все идет как надо.",
                "Отличное {feeling}! Солнце светит ярко.",
                "Замечательное {feeling} от {activity}.",
                "Восхитительное {feeling}! Спасибо всем.",
                "Потрясающее {feeling} от проделанной работы.",
                "Великолепное {feeling}! Так держать!"
            ],
            "services": [
                "Отличный сервис в {place}. Персонал {adj}.",
                "Супер обслуживание! Официант был {adj}.",
                "Прекрасный {service}. Буду обращаться еще.",
                "Качественный {service} по доступной цене.",
                "Быстрый и {adj} {service}. Рекомендую!",
                "Профессиональный {service}. Все на высшем уровне.",
                "Отзывчивый персонал и {adj} {service}.",
                "Удобный {service} с {adj} интерфейсом.",
                "Надежный {service}. Пользуюсь уже давно.",
                "Современный {service} с {adj} поддержкой."
            ]
        }
        
        negative_patterns = {
            "reviews": [
                "Ужасный {product}! Не рекомендую никому.",
                "Отвратительное качество {product}. Деньги на ветер.",
                "Худший {product} который я {action}. Кошмар!",
                "Провальный {product}. Полное разочарование.",
                "Неприемлемый {product}. Верну обратно.",
                "Безобразный {product}! Как такое можно продавать?",
                "Никуда не годный {product}. Не покупайте!",
                "Бракованный {product}. Потерянные деньги.",
                "Некачественный {product}. Обман покупателей!",
                "Поломанный {product}. Ужасная работа магазина."
            ],
            "emotions": [
                "Сегодня {feeling}. Все идет не так.",
                "Ужасное {feeling}! День не задался.",
                "Очень {feeling} результатом. Все плохо.",
                "Депрессивное {feeling} от {activity}.",
                "Сегодня особенно {feeling}. Ничего не получается.",
                "Печальное {feeling} от происходящего.",
                "Расстроенное {feeling} из-за неудач.",
                "Болезненное {feeling} от потерь.",
                "Мрачное {feeling} на душе.",
                "Тяжелое {feeling} весь день."
            ],
            "services": [
                "Ужасный сервис в {place}. Персонал {adj}.",
                "Отвратительное обслуживание! Официант был {adj}.",
                "Провальный {service}. Больше не обращусь.",
                "Некачественный {service} за большие деньги.",
                "Медленный и {adj} {service}. Не рекомендую!",
                "Непрофессиональный {service}. Все на низком уровне.",
                "Грубый персонал и {adj} {service}.",
                "Неудобный {service} с {adj} интерфейсом.",
                "Ненадежный {service}. Постоянные сбои.",
                "Устаревший {service} с {adj} поддержкой."
            ]
        }
        
        neutral_patterns = [
            "Обычный {product}. Ничего особенного, но пойдет.",
            "Стандартное качество {product}. Соответствует цене.",
            "Средний {product}. Есть плюсы и минусы.",
            "Приемлемый {product} для своих задач.",
            "Нормальный {product}. Без излишеств, но работает.",
            "Типичный {product} в своей категории.",
            "Неплохой {product}, но можно найти лучше.",
            "Удовлетворительный {product} за свои деньги.",
            "Обыкновенный {product}. Справляется с задачами.",
            "Рядовой {product}. Ожидания оправдались."
        ]
        
        # Словари для подстановок
        products = [
            "товар", "продукт", "телефон", "ноутбук", "планшет", "наушники",
            "книга", "фильм", "игра", "приложение", "сайт", "сервис",
            "ресторан", "кафе", "отель", "магазин", "курс", "программа"
        ]
        
        positive_adj = [
            "отличный", "превосходный", "замечательный", "великолепный",
            "потрясающий", "восхитительный", "прекрасный", "идеальный",
            "профессиональный", "качественный", "быстрый", "удобный"
        ]
        
        negative_adj = [
            "ужасный", "отвратительный", "провальный", "неприемлемый",
            "безобразный", "некачественный", "медленный", "неудобный",
            "грубый", "непрофессиональный", "ненадежный", "устаревший"
        ]
        
        positive_feelings = [
            "радуюсь", "доволен", "счастлив", "восхищен", "вдохновлен",
            "воодушевлен", "благодарен", "удовлетворен", "взволнован"
        ]
        
        negative_feelings = [
            "расстроен", "разочарован", "огорчен", "раздражен", "недоволен",
            "возмущен", "обижен", "злюсь", "грущу", "печалюсь"
        ]
        
        services = [
            "сервис", "обслуживание", "поддержка", "доставка", "ремонт",
            "консультация", "установка", "настройка", "обучение"
        ]
        
        places = [
            "ресторане", "кафе", "магазине", "салоне", "банке", "поликлинике",
            "офисе", "мастерской", "центре", "компании"
        ]
        
        activities = [
            "работы", "учебы", "покупки", "путешествия", "отдыха",
            "тренировки", "встречи", "проекта", "мероприятия"
        ]
        
        purposes = [
            "работы", "дома", "учебы", "отдыха", "путешествий",
            "спорта", "хобби", "бизнеса", "развлечений"
        ]
        
        actions = ["покупал", "заказывал", "использовал", "пробовал", "тестировал"]
        moods = ["отличное", "прекрасное", "замечательное", "великолепное"]
        
        data = []
        samples_per_class = num_samples // 3
        
        # Генерация позитивных примеров
        for _ in range(samples_per_class):
            category = np.random.choice(list(positive_patterns.keys()))
            template = np.random.choice(positive_patterns[category])
            
            text = template.format(
                product=np.random.choice(products),
                adj=np.random.choice(positive_adj),
                feeling=np.random.choice(positive_feelings),
                mood=np.random.choice(moods),
                service=np.random.choice(services),
                place=np.random.choice(places),
                activity=np.random.choice(activities),
                purpose=np.random.choice(purposes),
                action=np.random.choice(actions)
            )
            
            data.append({"text": text, "label": "positive"})
        
        # Генерация негативных примеров
        for _ in range(samples_per_class):
            category = np.random.choice(list(negative_patterns.keys()))
            template = np.random.choice(negative_patterns[category])
            
            text = template.format(
                product=np.random.choice(products),
                adj=np.random.choice(negative_adj),
                feeling=np.random.choice(negative_feelings),
                service=np.random.choice(services),
                place=np.random.choice(places),
                activity=np.random.choice(activities)
            )
            
            data.append({"text": text, "label": "negative"})
        
        # Генерация нейтральных примеров
        for _ in range(num_samples - 2 * samples_per_class):
            template = np.random.choice(neutral_patterns)
            text = template.format(product=np.random.choice(products))
            data.append({"text": text, "label": "neutral"})
        
        df = pd.DataFrame(data)
        return df.sample(frac=1).reset_index(drop=True)

    def load_all_available_datasets(self) -> pd.DataFrame:
        """Загрузка всех доступных датасетов"""
        combined_data = []
        
        # 1. Попытка загрузить RuSentiment
        rusentiment_path = self.download_rusentiment()
        if rusentiment_path:
            try:
                encodings = ['utf-8', 'cp1251', 'latin1']
                df = None
                
                for encoding in encodings:
                    try:
                        df = pd.read_csv(rusentiment_path, encoding=encoding)
                        print(f"✅ RuSentiment загружен ({len(df)} записей)")
                        break
                    except UnicodeDecodeError:
                        continue
                
                if df is not None and len(df) > 0:
                    # Обработка RuSentiment
                    df = self._process_rusentiment(df)
                    combined_data.append(df)
                    
            except Exception as e:
                print(f"❌ Ошибка обработки RuSentiment: {e}")
        
        # 2. Попытка загрузить LINIS Crowd
        linis_path = self.download_linis_crowd()
        if linis_path:
            try:
                df = pd.read_csv(linis_path, encoding='utf-8')
                df = self._process_linis_crowd(df)
                if len(df) > 0:
                    print(f"✅ LINIS Crowd добавлен ({len(df)} записей)")
                    combined_data.append(df)
            except Exception as e:
                print(f"❌ Ошибка обработки LINIS Crowd: {e}")
        
        # 3. Попытка загрузить Kaggle данные
        kaggle_path = self.download_kaggle_russian_news()
        if kaggle_path:
            try:
                df = pd.read_csv(kaggle_path, encoding='utf-8')
                df = self._process_kaggle_data(df)
                if len(df) > 0:
                    print(f"✅ Kaggle данные добавлены ({len(df)} записей)")
                    combined_data.append(df)
            except Exception as e:
                print(f"❌ Ошибка обработки Kaggle данных: {e}")
        
        # 4. Создание синтетических данных
        synthetic_size = 50000 if not combined_data else 20000
        synthetic_df = self.create_extended_synthetic_dataset(synthetic_size)
        print(f"✅ Синтетические данные созданы ({len(synthetic_df)} записей)")
        combined_data.append(synthetic_df)
        
        # Объединение всех датасетов
        if combined_data:
            final_df = pd.concat(combined_data, ignore_index=True)
            final_df = final_df.drop_duplicates(subset=['text']).reset_index(drop=True)
            print(f"\n🎯 Итоговый датасет: {len(final_df)} уникальных записей")
            
            # Балансировка классов
            final_df = self._balance_dataset(final_df)
            
            return final_df
        
        return self.create_extended_synthetic_dataset()

    def _process_rusentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """Обработка датасета RuSentiment"""
        # Поиск нужных колонок
        text_cols = [col for col in df.columns if any(name in col.lower() for name in ['text', 'comment', 'post'])]
        label_cols = [col for col in df.columns if any(name in col.lower() for name in ['label', 'sentiment', 'class'])]
        
        if text_cols and label_cols:
            df = df[[text_cols[0], label_cols[0]]].copy()
            df.columns = ['text', 'label']
            
            # Преобразование меток
            df['label'] = df['label'].astype(str).str.lower()
            df['label'] = df['label'].map({
                'positive': 'positive', 'negative': 'negative', 'neutral': 'neutral',
                'pos': 'positive', 'neg': 'negative', 'neu': 'neutral',
                '1': 'positive', '0': 'neutral', '-1': 'negative',
                'speech': 'neutral', 'skip': 'neutral'
            }).fillna('neutral')
            
            return df[df['text'].notna() & (df['text'] != '')]
        
        return pd.DataFrame(columns=['text', 'label'])

    def _process_linis_crowd(self, df: pd.DataFrame) -> pd.DataFrame:
        """Обработка датасета LINIS Crowd"""
        if 'text' in df.columns and 'label' in df.columns:
            df = df[['text', 'label']].copy()
            df['label'] = df['label'].astype(str).str.lower()
            return df[df['text'].notna() & (df['text'] != '')]
        return pd.DataFrame(columns=['text', 'label'])

    def _process_kaggle_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Обработка Kaggle данных"""
        # Адаптировать под конкретную структуру Kaggle датасета
        if 'comment' in df.columns and 'toxic' in df.columns:
            df = df[['comment', 'toxic']].copy()
            df.columns = ['text', 'label']
            df['label'] = df['label'].map({1: 'negative', 0: 'neutral'})
            return df[df['text'].notna() & (df['text'] != '')]
        return pd.DataFrame(columns=['text', 'label'])

    def _balance_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Балансировка датасета"""
        label_counts = df['label'].value_counts()
        min_count = label_counts.min()
        
        balanced_dfs = []
        for label in ['positive', 'negative', 'neutral']:
            label_df = df[df['label'] == label]
            if len(label_df) > min_count:
                label_df = label_df.sample(n=min_count, random_state=42)
            balanced_dfs.append(label_df)
        
        balanced_df = pd.concat(balanced_dfs, ignore_index=True)
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"📊 Датасет сбалансирован:")
        print(balanced_df['label'].value_counts())
        
        return balanced_df

    def preprocess_text(self, text: str) -> str:
        """Улучшенная предобработка текста"""
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
        
        # Удаление чисел (опционально)
        if TEXT_PREPROCESSING["remove_numbers"]:
            text = re.sub(r'\d+', '', text)
        
        # Удаление лишних пробелов
        text = re.sub(r'\s+', ' ', text)
        
        # Удаление пунктуации (кроме эмодзи)
        if TEXT_PREPROCESSING["remove_punctuation"]:
            text = re.sub(r'[^\w\s\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]', ' ', text)
        
        # Токенизация
        try:
            tokens = nltk.word_tokenize(text, language='russian')
        except:
            tokens = text.split()
        
        # Удаление стоп-слов
        if TEXT_PREPROCESSING["remove_stopwords"]:
            tokens = [token for token in tokens if token not in russian_stopwords and len(token) > 2]
        
        return ' '.join(tokens)

    def prepare_final_dataset(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Подготовка финального датасета"""
        print("🚀 Начинаем подготовку расширенного датасета...")
        
        # Загрузка всех доступных данных
        df = self.load_all_available_datasets()
        
        # Предобработка текстов
        print("🔄 Предобработка текстов...")
        df['processed_text'] = df['text'].apply(self.preprocess_text)
        
        # Удаление пустых строк
        df = df[df['processed_text'].str.len() > 0]
        
        # Кодирование меток
        df['label_encoded'] = self.label_encoder.fit_transform(df['label'])
        
        # Разделение на train/test
        train_df, test_df = train_test_split(
            df[['text', 'processed_text', 'label_encoded', 'label']], 
            test_size=0.2, 
            random_state=42,
            stratify=df['label_encoded']
        )
        
        print(f"\n✅ Финальная статистика:")
        print(f"   Обучающая выборка: {len(train_df)} примеров")
        print(f"   Тестовая выборка: {len(test_df)} примеров")
        print(f"   Распределение в обучающей выборке:")
        print(f"   {train_df['label'].value_counts()}")
        
        return train_df, test_df

    def save_processed_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        """Сохранение обработанных данных"""
        train_df.to_csv(PROCESSED_DATA_DIR / "train_data.csv", index=False)
        test_df.to_csv(PROCESSED_DATA_DIR / "test_data.csv", index=False)
        
        # Сохранение label encoder
        import joblib
        joblib.dump(self.label_encoder, PROCESSED_DATA_DIR / "label_encoder.pkl")
        joblib.dump(self.label_encoder, LABEL_ENCODER_PATH)
        
        print(f"\n💾 Данные сохранены в {PROCESSED_DATA_DIR}")


if __name__ == "__main__":
    # Инициализация tqdm для pandas
    tqdm.pandas()
    
    # Создание расширенного загрузчика
    loader = EnhancedDataLoader()
    
    # Подготовка данных
    train_df, test_df = loader.prepare_final_dataset()
    
    # Сохранение
    loader.save_processed_data(train_df, test_df)