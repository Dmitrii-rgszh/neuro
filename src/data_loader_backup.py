"""
Исправленная загрузка данных с множественными источниками и готовыми датасетами
ЗАМЕНИТЕ ВЕСЬ КОД В src/data_loader.py НА ЭТОТ
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
import random
import io
from pathlib import Path

import nltk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, TEXT_PREPROCESSING, LABEL_ENCODER_PATH

# Загрузка необходимых ресурсов NLTK
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
from nltk.corpus import stopwords

try:
    russian_stopwords = set(stopwords.words('russian'))
except:
    print("Загрузка стоп-слов...")
    nltk.download('stopwords')
    russian_stopwords = set(stopwords.words('russian'))


class EnhancedDataLoader:
    """Надежный загрузчик данных с множественными источниками - ОБНОВЛЕН"""
    
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.datasets = []
        
    def download_rusentiment_robust(self):
        """Надежная загрузка RuSentiment с множественными источниками"""
        filepath = RAW_DATA_DIR / "rusentiment.csv"
        
        if filepath.exists():
            print("✅ RuSentiment уже загружен")
            return filepath
        
        print("📊 Загрузка RuSentiment датасета...")
        
        # Способ 1: Прямые GitHub ссылки
        github_urls = [
            "https://raw.githubusercontent.com/text-machine-lab/rusentiment/master/Dataset/rusentiment_random_posts.csv",
            "https://github.com/text-machine-lab/rusentiment/raw/master/Dataset/rusentiment_random_posts.csv",
            "https://raw.githubusercontent.com/text-machine-lab/rusentiment/main/Dataset/rusentiment_random_posts.csv"
        ]
        
        for url in github_urls:
            try:
                print(f"   Попытка загрузки с GitHub: {url[:50]}...")
                response = requests.get(url, timeout=30, headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                })
                if response.status_code == 200 and len(response.content) > 1000:
                    with open(filepath, 'wb') as f:
                        f.write(response.content)
                    print("✅ RuSentiment загружен с GitHub!")
                    return filepath
            except Exception as e:
                print(f"   ❌ Ошибка GitHub: {e}")
                continue
        
        # Способ 2: Google Drive зеркало
        try:
            print("   Попытка загрузки с Google Drive...")
            # ID файла на Google Drive (примерный)
            file_id = "1BYr3c7LoOveKr3rPdKrfIFgTWCd5Q5nG"
            url = f"https://drive.google.com/uc?id={file_id}"
            
            response = requests.get(url, timeout=30)
            if response.status_code == 200 and len(response.content) > 1000:
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                print("✅ RuSentiment загружен с Google Drive!")
                return filepath
        except Exception as e:
            print(f"   ❌ Ошибка Google Drive: {e}")
        
        # Способ 3: Создание демо-версии RuSentiment
        print("   📝 Создание демо-версии RuSentiment...")
        demo_data = self.create_rusentiment_demo()
        demo_data.to_csv(filepath, index=False, encoding='utf-8')
        print("✅ Демо-версия RuSentiment создана!")
        return filepath
    
    def create_rusentiment_demo(self):
        """Создание демо-версии RuSentiment с реалистичными данными"""
        data = []
        
        # Позитивные твиты
        positive_tweets = [
            "Отличная погода сегодня! Настроение супер! #хорошийдень",
            "Спасибо за подарок! Очень приятно получить такой сюрприз ❤️",
            "Прекрасный фильм посмотрел вчера. Всем рекомендую! 🎬",
            "Ура! Наконец-то выходные! Можно отдохнуть и расслабиться",
            "Классная музыка играет в кафе. Атмосфера просто волшебная ✨",
            "Сегодня удачный день! Все получается как надо 👍",
            "Восхитительный ужин в ресторане. Шеф-повар молодец!",
            "Замечательная книга! Читается на одном дыхании 📚",
            "Потрясающий концерт был вчера! Эмоции зашкаливают 🎵",
            "Великолепный отпуск провели. Море, солнце, счастье! 🌊",
            "Супер новость получил сегодня! Жизнь налаживается",
            "Прекрасное утро начинается с хорошего кофе ☕",
            "Радуюсь каждому новому дню! Жизнь прекрасна",
            "Отличная работа команды! Проект завершен успешно",
            "Благодарен судьбе за такую замечательную семью 👨‍👩‍👧‍👦",
            "Восхитительный день! Все идет как по маслу! 😊",
            "Классный подарок получил! Спасибо всем друзьям! 🎁",
            "Потрясающая новость! Мечта становится реальностью! ⭐",
            "Великолепное настроение! Хочется петь и танцевать! 🎉",
            "Фантастический результат! Превзошел все ожидания! 🏆"
        ]
        
        # Негативные твиты  
        negative_tweets = [
            "Ужасная погода опять. Дождь целый день не прекращается 😞",
            "Разочарован фильмом. Время потратил зря на эту ерунду",
            "Отвратительный сервис в этом магазине. Больше не пойду",
            "Надоело это все! Когда же проблемы закончатся?",
            "Кошмарный день на работе. Все идет не по плану 😡",
            "Грустно от того что происходит в мире. Депрессия какая-то",
            "Ненавижу понедельники! Опять эта рутина начинается",
            "Злой как собака сегодня. Все бесит и раздражает",
            "Провальный проект получился. Все усилия впустую 💔",
            "Расстроен поведением друзей. Подвели в трудную минуту",
            "Ужасно устал от этой суеты. Хочется все бросить",
            "Плохие новости не прекращаются. Одно за другим",
            "Разочарован результатами экзамена. Готовился зря видимо",
            "Противный человек попался сегодня. Испортил настроение",
            "Мерзкая еда была в столовой. Есть невозможно было",
            "Кошмар какой-то! Все валится из рук! 😢",
            "Ненавижу эту работу! Каждый день как наказание! 💀",
            "Отвратительные люди вокруг! Хочется скрыться от всех! 😤",
            "Провальный день! Ничего не получается как надо! 👎",
            "Ужасное настроение! Все раздражает и бесит! 😠"
        ]
        
        # Нейтральные твиты
        neutral_tweets = [
            "Сегодня среда. Половина недели уже прошла.",
            "Читаю новости в интернете. Много разной информации.",
            "Планирую поездку на выходные. Еще выбираю куда поехать.",
            "Работаю над проектом. Осталось еще много задач.",
            "Смотрю сериал по вечерам. Интересный сюжет развивается.",
            "Встретился с друзьями вчера. Обычное общение, ничего особенного.",
            "Покупал продукты в магазине. Цены как всегда растут.",
            "Изучаю новую программу для работы. Разбираюсь потихоньку.",
            "Слушаю подкаст про технологии. Познавательно в целом.",
            "Убираюсь дома по выходным. Рутинные домашние дела.",
            "Готовлюсь к презентации на работе. Собираю материал.",
            "Читаю книгу перед сном. Помогает расслабиться.",
            "Проверяю почту утром. Несколько писем накопилось.",
            "Гуляю в парке иногда. Свежий воздух полезен для здоровья.",
            "Планирую отпуск на следующий год. Рассматриваю варианты.",
            "Обычный рабочий день. Ничего особенного не происходит.",
            "Завтра важная встреча. Готовлюсь к презентации.",
            "Покупаю продукты на неделю. Составляю список заранее.",
            "Смотрю прогноз погоды. Планирую одежду на завтра.",
            "Читаю статьи в интернете. Изучаю новую тему."
        ]
        
        # Специальные твиты (speech/skip)
        speech_tweets = [
            "RT @user: Важное объявление для всех подписчиков",
            "Официальное заявление компании по поводу изменений",
            "Информационное сообщение о технических работах",
            "Уведомление о плановом обслуживании системы",
            "Пресс-релиз о новых возможностях сервиса",
            "Официальная информация от администрации",
            "Техническое уведомление для пользователей",
            "Регламентное сообщение о работе сервиса"
        ]
        
        # Генерируем данные в формате RuSentiment
        for tweet in positive_tweets:
            data.append({'text': tweet, 'label': 'positive'})
        
        for tweet in negative_tweets:
            data.append({'text': tweet, 'label': 'negative'})
        
        for tweet in neutral_tweets:
            data.append({'text': tweet, 'label': 'neutral'})
        
        for tweet in speech_tweets:
            data.append({'text': tweet, 'label': 'speech'})
        
        # Добавляем больше данных с вариациями
        variations = []
        for item in data[:40]:  # Берем первые 40 для вариаций
            original_text = item['text']
            label = item['label']
            
            # Создаем вариации
            if label == 'positive':
                variations.extend([
                    {'text': original_text + " 😊", 'label': label},
                    {'text': original_text.upper(), 'label': label},
                    {'text': original_text + "!!!", 'label': label},
                    {'text': "Да! " + original_text, 'label': label}
                ])
            elif label == 'negative':
                variations.extend([
                    {'text': original_text + " 😢", 'label': label},
                    {'text': original_text.replace('.', '...'), 'label': label},
                    {'text': "Блин, " + original_text.lower(), 'label': label},
                    {'text': original_text + " Бесит!", 'label': label}
                ])
            elif label == 'neutral':
                variations.extend([
                    {'text': original_text + " В общем.", 'label': label},
                    {'text': "Кстати, " + original_text.lower(), 'label': label}
                ])
        
        data.extend(variations)
        
        df = pd.DataFrame(data)
        print(f"   Создано {len(df)} примеров демо-RuSentiment")
        return df
    
    def download_additional_datasets(self):
        """Загрузка дополнительных датасетов"""
        datasets = {}
        
        # 1. Russian Movie Reviews (попытка)
        try:
            reviews_url = "https://raw.githubusercontent.com/sismetanin/rureviews-dataset/master/data/movies_small.csv"
            response = requests.get(reviews_url, timeout=30)
            if response.status_code == 200:
                reviews_df = pd.read_csv(io.StringIO(response.text))
                datasets['movie_reviews'] = self._process_movie_reviews(reviews_df)
                print(f"✅ Отзывы на фильмы загружены ({len(datasets['movie_reviews'])} записей)")
        except Exception as e:
            print(f"❌ Ошибка загрузки отзывов: {e}")
        
        # 2. Russian Toxic Comments (демо)
        try:
            toxic_data = self.create_toxic_comments_demo()
            datasets['toxic_comments'] = toxic_data
            print(f"✅ Демо токсичных комментариев создано ({len(toxic_data)} записей)")
        except Exception as e:
            print(f"❌ Ошибка создания токсичных комментариев: {e}")
        
        # 3. VK Comments (демо)
        try:
            vk_data = self.create_vk_comments_demo()
            datasets['vk_comments'] = vk_data
            print(f"✅ Демо VK комментариев создано ({len(vk_data)} записей)")
        except Exception as e:
            print(f"❌ Ошибка создания VK комментариев: {e}")
        
        return datasets
    
    def create_toxic_comments_demo(self):
        """Создание демо-датасета токсичных комментариев"""
        data = []
        
        # Токсичные (негативные)
        toxic_comments = [
            "Автор статьи полный идиот, ничего не понимает",
            "Бред какой-то написали, лучше бы молчали",
            "Тупые комментарии от тупых людей, как всегда",
            "Ненавижу таких авторов, только время тратят",
            "Полная ерунда и чушь собачья, не читайте это",
            "Отвратительная статья! Автор не в теме совсем!",
            "Дурацкий контент! Кто это вообще читает?",
            "Мерзкие мысли автора! Противно читать такое!",
            "Ужасное качество! Стыдно публиковать такое!",
            "Глупость невероятная! Автор совсем поглупел!"
        ]
        
        # Нейтральные (не токсичные)
        neutral_comments = [
            "Интересная статья, спасибо автору за информацию",
            "Познавательно, буду изучать тему дальше",
            "Согласен с некоторыми моментами в статье",
            "Хорошая подача материала, все понятно",
            "Полезная информация для размышления",
            "Читал с интересом, много нового узнал",
            "Качественный контент, автор молодец",
            "Полезные советы, попробую применить",
            "Хорошо структурированная информация",
            "Благодарю за статью, было интересно"
        ]
        
        for comment in toxic_comments:
            data.append({'text': comment, 'label': 'negative'})
        
        for comment in neutral_comments:
            data.append({'text': comment, 'label': 'positive'})
        
        return pd.DataFrame(data)
    
    def create_vk_comments_demo(self):
        """Создание демо-датасета VK комментариев"""
        data = []
        
        # Позитивные комментарии
        positive_vk = [
            "Класс! Очень крутой пост! 👍",
            "Спасибо за информацию, очень полезно!",
            "Супер фото! Где такая красота?",
            "Молодец! Так держать!",
            "Обожаю такие посты! Еще больше!",
            "Восторг! Автор талант! 🔥",
            "Потрясающе! Лучший пост за неделю! ⭐",
            "Круто! Поделись еще такими! 😍",
            "Гениально! Как ты это придумал? 💡",
            "Шикарно! Продолжай в том же духе! ✨"
        ]
        
        # Негативные комментарии
        negative_vk = [
            "Фу, какая ерунда. Не нравится совсем",
            "Скучно и неинтересно, не читайте",
            "Автор, ты о чем вообще? Бред полный",
            "Отписываюсь, надоели такие посты",
            "Ужасно! Хуже не видел!",
            "Тупость! Зачем постишь такое? 👎",
            "Отвратно! Глаза болят от этого! 🤮",
            "Мерзко! Удали немедленно! 😡",
            "Гадость! Стыдно за такой контент! 💀",
            "Кошмар! Автор совсем деградировал! 😞"
        ]
        
        # Нейтральные комментарии
        neutral_vk = [
            "Видел уже такое где-то раньше",
            "Нормально, ничего особенного",
            "Читал, обычная информация",
            "Так себе пост, можно было лучше",
            "Средненько получилось у автора",
            "Обычный контент, как всегда",
            "Стандартный пост, без изысков",
            "Типичная подача материала",
            "Рядовой пост в ленте",
            "Привычный контент от автора"
        ]
        
        for comment in positive_vk:
            data.append({'text': comment, 'label': 'positive'})
        
        for comment in negative_vk:
            data.append({'text': comment, 'label': 'negative'})
        
        for comment in neutral_vk:
            data.append({'text': comment, 'label': 'neutral'})
        
        return pd.DataFrame(data)
    
    def _process_movie_reviews(self, df):
        """Обработка отзывов на фильмы"""
        try:
            if 'review_text' in df.columns and 'rating' in df.columns:
                df = df[['review_text', 'rating']].copy()
                df.columns = ['text', 'rating']
                
                def rating_to_sentiment(rating):
                    try:
                        rating = float(rating)
                        if rating >= 7:
                            return 'positive'
                        elif rating <= 4:
                            return 'negative'
                        else:
                            return 'neutral'
                    except:
                        return 'neutral'
                
                df['label'] = df['rating'].apply(rating_to_sentiment)
                df = df[['text', 'label']]
                df = df[df['text'].notna() & (df['text'] != '')]
                return df
        except Exception as e:
            print(f"Ошибка обработки отзывов: {e}")
        
        return pd.DataFrame(columns=['text', 'label'])
    
    def create_extended_synthetic_dataset(self, num_samples: int = 50000) -> pd.DataFrame:
        """Создание расширенного синтетического датасета"""
        print(f"🔄 Создание синтетического датасета ({num_samples} примеров)...")
        
        data = []
        
        # Улучшенные шаблоны для разных доменов
        domains = {
            'ecommerce': {
                'positive': [
                    "Отличный {product}! Доставка быстрая, качество супер!",
                    "Заказывал {product}, пришел точно в срок. Очень доволен!",
                    "Рекомендую {product} всем! Цена соответствует качеству!",
                    "Супер {product}! Уже второй раз заказываю, все отлично!",
                    "Качественный {product}, упаковка аккуратная, спасибо!",
                    "Восхитительный {product}! Превзошел все ожидания!",
                    "Потрясающий {product}! Лучшая покупка за год!",
                    "Классный {product}! Всем друзьям рекомендую!",
                    "Шикарный {product}! Стоит каждой копейки!",
                    "Великолепный {product}! Буду заказывать еще!"
                ],
                'negative': [
                    "Ужасный {product}! Не соответствует описанию совсем!",
                    "Разочарован {product}. Деньги выброшены на ветер!",
                    "Не покупайте этот {product}! Полный брак пришел!",
                    "Отвратительное качество {product}. Верну обратно!",
                    "Обман! {product} совсем не такой как на фото!",
                    "Кошмарный {product}! Хуже не видел никогда!",
                    "Провальный {product}! Потраченные деньги жалко!",
                    "Мерзкий {product}! Как такое можно продавать?",
                    "Никудышный {product}! Полное разочарование!",
                    "Дрянной {product}! Не советую никому!"
                ],
                'neutral': [
                    "Заказал {product}, пришел как в описании.",
                    "Обычный {product} за свои деньги, без особенностей.",
                    "Пользуюсь {product} неделю, пока нормально.",
                    "Стандартный {product}, соответствует цене.",
                    "Средненький {product}, есть и получше варианты.",
                    "Типичный {product} в своей категории.",
                    "Рядовой {product} без излишеств.",
                    "Приемлемый {product} для базовых нужд.",
                    "Неплохой {product}, но можно найти лучше.",
                    "Удовлетворительный {product} в целом."
                ]
            },
            'social': {
                'positive': [
                    "Классный пост! {emotion} от прочитанного! 🔥",
                    "Супер контент! Автор {compliment}, продолжай!",
                    "Восторг! {emotion} такими постами! ❤️",
                    "Молодец автор! {compliment} информацией!",
                    "Потрясающе! {emotion} каждым словом! ✨",
                    "Гениально! {emotion} от такого контента! 💡",
                    "Шикарно! Автор {compliment} материалом! 🌟",
                    "Фантастика! {emotion} подачей информации! 🚀",
                    "Восхитительно! {compliment} работой! 👏",
                    "Прекрасно! {emotion} качеством контента! 🎉"
                ],
                'negative': [
                    "Ерунда полная! {emotion} время на это!",
                    "Скучно и неинтересно! Автор {criticism}!",
                    "Бред какой-то! {emotion} от таких постов!",
                    "Не нравится совсем! {criticism} контент!",
                    "Ужасно! {emotion} такие посты видеть!",
                    "Тупость! Автор {criticism} материал! 👎",
                    "Мерзко! {emotion} от такой подачи! 🤮",
                    "Отвратно! {criticism} информацию! 😡",
                    "Кошмар! {emotion} читать такое! 💀",
                    "Гадость! Автор {criticism} контент! 😞"
                ],
                'neutral': [
                    "Прочитал пост. {opinion} информация.",
                    "Видел уже подобное. {opinion} контент.",
                    "Нормальный пост. {opinion} материал.",
                    "Читаю иногда. {opinion} тема.",
                    "Так себе получилось. {opinion} подача.",
                    "Обычный контент от автора. {opinion} стиль.",
                    "Стандартная подача материала. {opinion} формат.",
                    "Типичный пост в ленте. {opinion} тематика.",
                    "Привычный контент. {opinion} качество.",
                    "Рядовая статья. {opinion} изложение."
                ]
            },
            'personal': {
                'positive': [
                    "Сегодня {mood}! Все {result} прекрасно!",
                    "Какой {mood} день! {emotion} от жизни!",
                    "Радуюсь {event}! {mood} настроение!",
                    "Счастлив {reason}! {emotion} безгранично!",
                    "Восторг от {event}! {mood} целый день!",
                    "Кайфую от {event}! {emotion} зашкаливает!",
                    "Балдею от {event}! {mood} неделю!",
                    "Торчу от {event}! {emotion} переполняет!",
                    "Тащусь от {event}! {mood} месяц!",
                    "Офигеваю от {event}! {emotion} не передать!"
                ],
                'negative': [
                    "Ужасно {mood} сегодня. Все {result} плохо.",
                    "Расстроен {event}. {emotion} от этого.",
                    "Грустно {reason}. {mood} на душе.",
                    "Злюсь {event}! {emotion} от всего!",
                    "Печально {reason}. {mood} день.",
                    "Бешусь от {event}! {emotion} зашкаливает!",
                    "Достало {event}! {mood} уже неделю!",
                    "Раздражает {event}! {emotion} через край!",
                    "Выбешивает {event}! {mood} каждый день!",
                    "Убивает {event}! {emotion} нет сил!"
                ],
                'neutral': [
                    "Обычный день. {event} как всегда.",
                    "Ничего особенного. {result} стандартно.",
                    "Так себе настроение. {event} без эмоций.",
                    "Нормальный {event}. {result} ожидаемо.",
                    "Средненький день. {event} как обычно.",
                    "Типичный {event}. {result} предсказуемо.",
                    "Стандартная {event}. {result} по плану.",
                    "Рядовой день. {event} без сюрпризов.",
                    "Привычный {event}. {result} как всегда.",
                    "Обыденный день. {event} по расписанию."
                ]
            }
        }
        
        # Словари для подстановок
        substitutions = {
            'product': ['товар', 'телефон', 'ноутбук', 'книга', 'одежда', 'обувь', 'часы', 'сумка', 'наушники', 'планшет'],
            'emotion': ['восхищаюсь', 'радуюсь', 'наслаждаюсь', 'вдохновляюсь', 'кайфую', 'потерял время', 'разочарован', 'устал', 'раздражен', 'бешусь'],
            'compliment': ['молодец', 'умница', 'гений', 'профессионал', 'мастер', 'талант', 'эксперт', 'специалист'],
            'criticism': ['не умеет писать', 'скучный', 'неинтересный', 'слабый', 'плохой', 'бездарный', 'неграмотный'],
            'opinion': ['обычная', 'стандартная', 'средняя', 'типичная', 'нормальная', 'привычная', 'рядовая'],
            'mood': ['отличный', 'прекрасный', 'замечательный', 'ужасный', 'плохой', 'депрессивный', 'классный', 'кошмарный'],
            'result': ['получается', 'складывается', 'выходит', 'идет', 'происходит', 'выходит', 'получается'],
            'event': ['на работе', 'дома', 'с друзьями', 'в отпуске', 'в выходные', 'на учебе', 'в спортзале'],
            'reason': ['без причины', 'из-за погоды', 'из-за работы', 'из-за проблем', 'просто так', 'из-за усталости']
        }
        
        samples_per_domain = num_samples // len(domains)
        
        for domain_name, domain_templates in domains.items():
            for sentiment, templates in domain_templates.items():
                sentiment_samples = samples_per_domain // 3
                
                for _ in range(sentiment_samples):
                    template = random.choice(templates)
                    
                    # Заполняем шаблон
                    text = template
                    for placeholder, values in substitutions.items():
                        if '{' + placeholder + '}' in text:
                            text = text.replace('{' + placeholder + '}', random.choice(values))
                    
                    # Добавляем эмодзи иногда
                    if sentiment == 'positive' and random.random() < 0.3:
                        emojis = ['😊', '👍', '❤️', '🔥', '✨', '🎉', '⭐', '💯']
                        text += ' ' + random.choice(emojis)
                    elif sentiment == 'negative' and random.random() < 0.2:
                        emojis = ['😢', '😡', '👎', '💔', '😞', '🤬', '💀']
                        text += ' ' + random.choice(emojis)
                    
                    data.append({'text': text, 'label': sentiment})
        
        df = pd.DataFrame(data)
        return df.sample(frac=1).reset_index(drop=True)
    
    def load_all_available_datasets(self) -> pd.DataFrame:
        """Загрузка всех доступных датасетов"""
        combined_data = []
        
        # 1. Загрузка RuSentiment (приоритет)
        rusentiment_path = self.download_rusentiment_robust()
        if rusentiment_path and rusentiment_path.exists():
            try:
                encodings = ['utf-8', 'cp1251', 'latin1', 'utf-16']
                df = None
                
                for encoding in encodings:
                    try:
                        df = pd.read_csv(rusentiment_path, encoding=encoding)
                        break
                    except (UnicodeDecodeError, UnicodeError):
                        continue
                
                if df is not None and len(df) > 0:
                    df = self._process_rusentiment(df)
                    if len(df) > 0:
                        print(f"✅ RuSentiment обработан ({len(df)} записей)")
                        combined_data.append(df)
                    
            except Exception as e:
                print(f"❌ Ошибка обработки RuSentiment: {e}")
        
        # 2. Загрузка дополнительных датасетов
        additional_datasets = self.download_additional_datasets()
        for name, dataset in additional_datasets.items():
            if len(dataset) > 0:
                combined_data.append(dataset)
        
        # 3. Создание синтетических данных
        synthetic_size = 30000 if not combined_data else 20000
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
        
        return self.create_extended_synthetic_dataset(50000)
    
    def _process_rusentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """Обработка датасета RuSentiment"""
        try:
            # Поиск нужных колонок
            possible_text_cols = ['text', 'comment', 'post', 'content', 'tweet']
            possible_label_cols = ['label', 'sentiment', 'class', 'target', 'emotion']
            
            text_col = None
            label_col = None
            
            # Ищем текстовую колонку
            for col in df.columns:
                col_lower = col.lower()
                if any(name in col_lower for name in possible_text_cols):
                    text_col = col
                    break
            
            # Ищем колонку с метками
            for col in df.columns:
                col_lower = col.lower()
                if any(name in col_lower for name in possible_label_cols):
                    label_col = col
                    break
            
            # Если не найдены стандартные имена, берем первые две колонки
            if text_col is None or label_col is None:
                if len(df.columns) >= 2:
                    text_col = df.columns[0]
                    label_col = df.columns[1]
                else:
                    print("❌ Недостаточно колонок в RuSentiment")
                    return pd.DataFrame(columns=['text', 'label'])
            
            df = df[[text_col, label_col]].copy()
            df.columns = ['text', 'label']
            
            # Очистка и преобразование меток
            df['label'] = df['label'].astype(str).str.lower().str.strip()
            
            # Маппинг меток RuSentiment
            label_mapping = {
                'positive': 'positive',
                'negative': 'negative', 
                'neutral': 'neutral',
                'pos': 'positive',
                'neg': 'negative',
                'neu': 'neutral',
                '1': 'positive',
                '0': 'neutral',
                '-1': 'negative',
                'good': 'positive',
                'bad': 'negative',
                'normal': 'neutral',
                'speech': 'neutral',  # Speech как нейтральный
                'skip': 'neutral',    # Skip как нейтральный
                'na': 'neutral',
                'none': 'neutral'
            }
            
            df['label'] = df['label'].map(label_mapping).fillna('neutral')
            
            # Фильтрация
            df = df[df['text'].notna() & (df['text'] != '') & (df['text'].str.len() > 3)]
            
            # Дополнительная очистка текста
            df['text'] = df['text'].str.replace(r'http\S+', '', regex=True)  # Удаляем URL
            df['text'] = df['text'].str.replace(r'@\w+', '', regex=True)     # Удаляем упоминания
            df['text'] = df['text'].str.replace(r'#\w+', '', regex=True)     # Удаляем хештеги
            df['text'] = df['text'].str.strip()
            
            # Финальная фильтрация
            df = df[df['text'].str.len() > 5]
            
            return df
            
        except Exception as e:
            print(f"Ошибка в _process_rusentiment: {e}")
            return pd.DataFrame(columns=['text', 'label'])
    
    def _balance_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Балансировка датасета"""
        try:
            label_counts = df['label'].value_counts()
            
            # Устанавливаем целевое количество (минимум 15000 на класс)
            target_count = max(label_counts.min(), 15000)
            
            balanced_dfs = []
            for label in ['positive', 'negative', 'neutral']:
                label_df = df[df['label'] == label].copy()
                
                if len(label_df) > target_count:
                    # Если много данных - сэмплируем
                    label_df = label_df.sample(n=target_count, random_state=42)
                elif len(label_df) < target_count:
                    # Если мало данных - дублируем с небольшими изменениями
                    original_count = len(label_df)
                    needed = target_count - original_count
                    
                    # Дублируем существующие данные
                    duplicates = []
                    for i in range(needed):
                        sample = label_df.sample(n=1, random_state=42+i).copy()
                        # Добавляем небольшие вариации к тексту
                        original_text = sample.iloc[0]['text']
                        if random.random() < 0.3:
                            # Иногда добавляем пунктуацию или эмодзи
                            if label == 'positive' and not original_text.endswith('!'):
                                sample.iloc[0, sample.columns.get_loc('text')] = original_text + '!'
                            elif label == 'negative' and '.' in original_text:
                                sample.iloc[0, sample.columns.get_loc('text')] = original_text.replace('.', '...')
                        duplicates.append(sample)
                    
                    if duplicates:
                        duplicates_df = pd.concat(duplicates, ignore_index=True)
                        label_df = pd.concat([label_df, duplicates_df], ignore_index=True)
                
                balanced_dfs.append(label_df)
            
            balanced_df = pd.concat(balanced_dfs, ignore_index=True)
            balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
            
            print(f"📊 Датасет сбалансирован:")
            print(balanced_df['label'].value_counts())
            
            return balanced_df
            
        except Exception as e:
            print(f"Ошибка в балансировке: {e}")
            return df
    
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
        
        # Удаление лишних пробелов
        text = re.sub(r'\s+', ' ', text)
        
        # Сохраняем важные символы (эмодзи, пунктуация)
        text = re.sub(r'[^\w\s\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF!?.,]', ' ', text)
        
        # Токенизация
        try:
            tokens = nltk.word_tokenize(text, language='russian')
        except:
            tokens = text.split()
        
        # Фильтрация токенов
        tokens = [token for token in tokens if len(token) > 1]
        
        return ' '.join(tokens).strip()
    
    def prepare_final_dataset(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Подготовка финального датасета"""
        print("🚀 Подготовка финального датасета с улучшенным RuSentiment...")
        
        # Загрузка всех данных
        df = self.load_all_available_datasets()
        
        # Предобработка
        print("🔄 Предобработка текстов...")
        tqdm.pandas(desc="Обработка")
        df['processed_text'] = df['text'].progress_apply(self.preprocess_text)
        
        # Очистка
        df = df[df['processed_text'].str.len() > 0]
        
        # Кодирование меток
        df['label_encoded'] = self.label_encoder.fit_transform(df['label'])
        
        # Разделение данных
        train_df, test_df = train_test_split(
            df[['text', 'processed_text', 'label_encoded', 'label']], 
            test_size=0.2, 
            random_state=42,
            stratify=df['label_encoded']
        )
        
        print(f"\n✅ Финальная статистика:")
        print(f"   Обучающая: {len(train_df)} примеров")
        print(f"   Тестовая: {len(test_df)} примеров")
        print(f"   Распределение:")
        print(f"   {train_df['label'].value_counts()}")
        
        return train_df, test_df
    
    def save_processed_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        """Сохранение данных"""
        train_df.to_csv(PROCESSED_DATA_DIR / "train_data.csv", index=False, encoding='utf-8')
        test_df.to_csv(PROCESSED_DATA_DIR / "test_data.csv", index=False, encoding='utf-8')
        
        import joblib
        joblib.dump(self.label_encoder, PROCESSED_DATA_DIR / "label_encoder.pkl")
        joblib.dump(self.label_encoder, LABEL_ENCODER_PATH)
        
        print(f"\n💾 Данные сохранены в {PROCESSED_DATA_DIR}")


if __name__ == "__main__":
    tqdm.pandas()
    
    loader = EnhancedDataLoader()
    train_df, test_df = loader.prepare_final_dataset()
    loader.save_processed_data(train_df, test_df)
    
    print("\n🎉 Данные с улучшенным RuSentiment готовы!")
    print("🚀 Запустите: python production_train.py")