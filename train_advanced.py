"""
Продвинутое обучение модели с качественными данными
"""
import sys
sys.path.append('src')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, Conv1D, GlobalMaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import joblib
import re
from pathlib import Path
import random

# Конфигурация
MAX_WORDS = 50000  # Больше слов для лучшего понимания
MAX_LENGTH = 200   # Длиннее тексты
EMBEDDING_DIM = 300  # Больше размерность
LSTM_UNITS = 128
EPOCHS = 50  # Больше эпох
BATCH_SIZE = 32  # Меньше батч для лучшего обучения

# Пути
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

def create_quality_dataset(num_samples=50000):
    """Создание качественного разнообразного датасета"""
    print("Создание качественного датасета...")
    
    data = []
    
    # ПОЗИТИВНЫЕ ПРИМЕРЫ (разнообразные контексты)
    positive_contexts = {
        'покупки': [
            "Заказал {item}, пришло быстро, качество {quality}! {emotion}",
            "Купил {item} и остался очень доволен. {quality} товар, {service} сервис!",
            "{item} превзошел все ожидания! {quality} качество, {emotion}",
            "Наконец-то нашел идеальный {item}! {service} обслуживание, всем рекомендую!",
            "Покупкой {item} очень доволен, {quality} исполнение, {emotion}"
        ],
        'эмоции': [
            "Сегодня такой {day} день! Настроение {mood}, все {result}!",
            "Я так {emotion}! Наконец-то {event}, это {feeling}!",
            "Какое {feeling} утро! Проснулся с {mood} настроением!",
            "Это просто {amazing}! Я в полном {emotion}!",
            "{event} прошло {result}! Чувствую себя {mood}!"
        ],
        'достижения': [
            "Ура! Я {achievement}! Это было {difficulty}, но я {result}!",
            "Наконец-то {achievement}! Столько {effort}, но оно того {worth}!",
            "Горжусь собой - {achievement}! {emotion} безгранична!",
            "Сделал это! {achievement} - моя новая {milestone}!",
            "Невероятно! Я смог {achievement}! {emotion}!"
        ],
        'отношения': [
            "Встретил {person} человека, мы так {connection}!",
            "С {person} так {feeling} проводить время! {emotion}!",
            "Люблю свою {family}, они делают меня {mood}!",
            "Друзья - это {treasure}! С ними всегда {feeling}!",
            "Благодарен {person} за {action}, это так {emotion}!"
        ]
    }
    
    positive_items = ["телефон", "ноутбук", "наушники", "часы", "кроссовки", "рюкзак", "книгу", "подарок"]
    positive_quality = ["отличное", "превосходное", "замечательное", "шикарное", "высококлассное", "премиальное"]
    positive_service = ["быстрый", "вежливый", "профессиональный", "внимательный", "отзывчивый"]
    positive_emotion = ["Супер", "Восторг", "Счастлив", "Рад", "Доволен", "В восхищении", "Радость"]
    positive_day = ["прекрасный", "чудесный", "солнечный", "удачный", "счастливый", "волшебный"]
    positive_mood = ["отличное", "прекрасное", "радостное", "приподнятое", "восторженное"]
    positive_result = ["получается", "удается", "складывается", "выходит", "получилось"]
    positive_feeling = ["счастье", "радость", "восторг", "удовольствие", "блаженство"]
    positive_amazing = ["потрясающе", "невероятно", "фантастика", "чудесно", "великолепно"]
    positive_achievement = ["сдал экзамен", "получил работу", "закончил проект", "выиграл конкурс", "достиг цели"]
    positive_difficulty = ["сложно", "трудно", "непросто", "тяжело", "нелегко"]
    positive_effort = ["усилий", "труда", "стараний", "попыток", "работы"]
    positive_worth = ["стоило", "стоит", "оправдано", "заслужено", "оправдало"]
    positive_milestone = ["победа", "достижение", "вершина", "рекорд", "успех"]
    positive_person = ["замечательного", "прекрасного", "удивительного", "чудесного", "классного"]
    positive_connection = ["совпали", "подружились", "сошлись", "понимаем друг друга", "на одной волне"]
    positive_family = ["семью", "родных", "близких", "друзей", "команду"]
    positive_treasure = ["сокровище", "богатство", "счастье", "подарок", "благо"]
    positive_action = ["поддержку", "помощь", "совет", "внимание", "заботу"]
    
    # НЕГАТИВНЫЕ ПРИМЕРЫ
    negative_contexts = {
        'покупки': [
            "Купил {item}, полное {disappointment}. {quality} качество, {emotion}",
            "Заказывал {item}, пришел {defect}. {service} сервис, {emotion}!",
            "{item} оказался {quality}, деньги {waste}. {emotion}",
            "Разочарован покупкой {item}. {defect}, {service} поддержка",
            "Не покупайте {item}! {quality} товар, {emotion}"
        ],
        'эмоции': [
            "Что за {day} день... Настроение {mood}, все {result}",
            "Я так {emotion}... Опять {event}, это {feeling}",
            "{feeling} утро, проснулся с {mood} настроением",
            "Все {result}, я в {emotion}. Когда это {end}?",
            "Снова {event}. Чувствую себя {mood}"
        ],
        'проблемы': [
            "{problem} опять {occurred}. Уже {tired} от этого",
            "Не могу больше! {problem} меня {effect}",
            "Когда же {problem} закончится? {emotion}",
            "Из-за {problem} все {result}. {feeling}",
            "{problem} разрушает мою {life}. {emotion}"
        ],
        'разочарования': [
            "Ожидал {expected}, получил {reality}. {emotion}",
            "Думал будет {expected}, а вышло {reality}. {disappointment}",
            "Надеялся на {expected}, но {reality}. {feeling}",
            "Вместо {expected} получил {reality}. {emotion}",
            "Обещали {expected}, дали {reality}. {disappointment}"
        ]
    }
    
    negative_item = ["телефон", "ноутбук", "товар", "заказ", "продукт", "девайс"]
    negative_disappointment = ["разочарование", "кошмар", "ужас", "провал", "фиаско"]
    negative_quality = ["ужасное", "отвратительное", "никудышное", "плохое", "низкое"]
    negative_emotion = ["Зол", "Расстроен", "Разочарован", "Обижен", "Раздражен", "Злюсь"]
    negative_defect = ["брак", "дефект", "сломанный товар", "некачественный", "испорченный"]
    negative_service = ["ужасный", "грубый", "некомпетентный", "медленный", "отвратительный"]
    negative_waste = ["на ветер", "выброшены", "потрачены зря", "пропали", "потеряны"]
    negative_day = ["ужасный", "кошмарный", "паршивый", "отвратительный", "проклятый"]
    negative_mood = ["плохое", "ужасное", "паршивое", "мерзкое", "отвратительное"]
    negative_result = ["плохо", "разваливается", "рушится", "идет не так", "провалилось"]
    negative_event = ["не получилось", "провалилось", "сорвалось", "пошло не так", "случилось"]
    negative_feeling = ["ужас", "кошмар", "отчаяние", "грусть", "печаль"]
    negative_end = ["кончится", "закончится", "прекратится", "остановится", "пройдет"]
    negative_problem = ["Проблема", "Неприятность", "Беда", "Кризис", "Катастрофа"]
    negative_occurred = ["случилась", "произошла", "повторилась", "вернулась", "началась"]
    negative_tired = ["устал", "измучен", "вымотан", "изнурен", "замучен"]
    negative_effect = ["достала", "измучила", "довела", "разрушает", "убивает"]
    negative_life = ["жизнь", "планы", "настроение", "здоровье", "работу"]
    negative_expected = ["лучшее", "качество", "хорошее", "нормальное", "достойное"]
    negative_reality = ["худшее", "мусор", "ерунду", "кошмар", "ужас"]
    
    # НЕЙТРАЛЬНЫЕ ПРИМЕРЫ
    neutral_contexts = {
        'описания': [
            "{item} имеет {feature}. Работает {performance}",
            "Использую {item} уже {time}. {observation}",
            "{item} стоит {price}. Функции {standard}",
            "В {item} есть {pros}, но также {cons}",
            "{item} подходит для {purpose}. {summary}"
        ],
        'факты': [
            "Сегодня {day}. Погода {weather}, планы {plans}",
            "На работе {routine}. Все {status}",
            "Сделал {task}. Теперь нужно {next}",
            "{event} прошло {normally}. Результат {expected}",
            "День как день. {routine}, ничего {special}"
        ],
        'наблюдения': [
            "Заметил, что {observation}. {conclusion}",
            "Люди {behavior}. Это {normal}",
            "В городе {happening}. Жизнь {continues}",
            "Читал про {topic}. {interesting}",
            "Видел {event}. {reaction}"
        ]
    }
    
    neutral_item = ["товар", "продукт", "устройство", "предмет", "вещь"]
    neutral_feature = ["стандартные функции", "обычные характеристики", "базовые опции", "средние параметры"]
    neutral_performance = ["нормально", "стандартно", "как ожидалось", "без сюрпризов", "обычно"]
    neutral_time = ["неделю", "месяц", "несколько дней", "пару недель", "некоторое время"]
    neutral_observation = ["Работает стабильно", "Без особенностей", "Все стандартно", "Ничего необычного"]
    neutral_price = ["среднюю цену", "обычные деньги", "стандартную стоимость", "рыночную цену"]
    neutral_standard = ["стандартные", "обычные", "базовые", "типичные", "средние"]
    neutral_pros = ["плюсы", "достоинства", "преимущества", "положительные стороны"]
    neutral_cons = ["минусы", "недостатки", "нюансы", "особенности"]
    neutral_purpose = ["повседневного использования", "обычных задач", "стандартных целей", "базовых нужд"]
    neutral_summary = ["В целом нормально", "Соответствует цене", "Для своих задач подходит", "Обычный вариант"]
    neutral_day = ["понедельник", "вторник", "среда", "четверг", "пятница"]
    neutral_weather = ["обычная", "переменная", "по сезону", "стандартная", "типичная"]
    neutral_plans = ["обычные", "стандартные", "рабочие", "повседневные", "типичные"]
    neutral_routine = ["обычная работа", "стандартные задачи", "рутина", "текучка", "обычные дела"]
    neutral_status = ["как обычно", "по плану", "в норме", "стандартно", "без изменений"]
    neutral_task = ["работу", "задание", "дело", "поручение", "задачу"]
    neutral_next = ["другое", "следующее", "остальное", "продолжить", "закончить"]
    neutral_event = ["Совещание", "Встреча", "Мероприятие", "Собрание", "События"]
    neutral_normally = ["по плану", "как обычно", "стандартно", "без эксцессов", "нормально"]
    neutral_expected = ["ожидаемый", "предсказуемый", "стандартный", "обычный", "типичный"]
    neutral_special = ["особенного", "необычного", "выдающегося", "примечательного", "интересного"]
    neutral_behavior = ["ведут себя обычно", "делают свое дело", "живут своей жизнью", "заняты делами"]
    neutral_normal = ["нормально", "обычно", "естественно", "типично", "стандартно"]
    neutral_happening = ["идет стройка", "ремонт дорог", "обычная жизнь", "повседневность", "будни"]
    neutral_continues = ["продолжается", "идет своим чередом", "течет", "движется", "не останавливается"]
    neutral_topic = ["новости", "технологии", "события", "тенденции", "изменения"]
    neutral_interesting = ["Любопытно", "Познавательно", "Информативно", "Полезно знать", "Интересный факт"]
    neutral_reaction = ["Обычное дело", "Ничего особенного", "Бывает", "Нормально", "Жизнь"]
    
    # Генерация позитивных примеров
    for _ in range(num_samples // 3):
        context_type = random.choice(list(positive_contexts.keys()))
        template = random.choice(positive_contexts[context_type])
        
        # Заполняем шаблон
        text = template
        replacements = {
            '{item}': random.choice(positive_items),
            '{quality}': random.choice(positive_quality),
            '{service}': random.choice(positive_service),
            '{emotion}': random.choice(positive_emotion),
            '{day}': random.choice(positive_day),
            '{mood}': random.choice(positive_mood),
            '{result}': random.choice(positive_result),
            '{event}': random.choice(positive_achievement),
            '{feeling}': random.choice(positive_feeling),
            '{amazing}': random.choice(positive_amazing),
            '{achievement}': random.choice(positive_achievement),
            '{difficulty}': random.choice(positive_difficulty),
            '{effort}': random.choice(positive_effort),
            '{worth}': random.choice(positive_worth),
            '{milestone}': random.choice(positive_milestone),
            '{person}': random.choice(positive_person),
            '{connection}': random.choice(positive_connection),
            '{family}': random.choice(positive_family),
            '{treasure}': random.choice(positive_treasure),
            '{action}': random.choice(positive_action)
        }
        
        for key, value in replacements.items():
            text = text.replace(key, value)
        
        # Добавляем вариативность
        if random.random() > 0.5:
            text += " " + random.choice(["😊", "😄", "🎉", "❤️", "👍", "✨", "🙌", ""])
        
        data.append({"text": text, "label": "positive"})
    
    # Генерация негативных примеров
    for _ in range(num_samples // 3):
        context_type = random.choice(list(negative_contexts.keys()))
        template = random.choice(negative_contexts[context_type])
        
        text = template
        replacements = {
            '{item}': random.choice(negative_item),
            '{disappointment}': random.choice(negative_disappointment),
            '{quality}': random.choice(negative_quality),
            '{emotion}': random.choice(negative_emotion),
            '{defect}': random.choice(negative_defect),
            '{service}': random.choice(negative_service),
            '{waste}': random.choice(negative_waste),
            '{day}': random.choice(negative_day),
            '{mood}': random.choice(negative_mood),
            '{result}': random.choice(negative_result),
            '{event}': random.choice(negative_event),
            '{feeling}': random.choice(negative_feeling),
            '{end}': random.choice(negative_end),
            '{problem}': random.choice(negative_problem),
            '{occurred}': random.choice(negative_occurred),
            '{tired}': random.choice(negative_tired),
            '{effect}': random.choice(negative_effect),
            '{life}': random.choice(negative_life),
            '{expected}': random.choice(negative_expected),
            '{reality}': random.choice(negative_reality)
        }
        
        for key, value in replacements.items():
            text = text.replace(key, value)
        
        if random.random() > 0.5:
            text += " " + random.choice(["😔", "😡", "😞", "💔", "👎", "😢", "🤬", ""])
        
        data.append({"text": text, "label": "negative"})
    
    # Генерация нейтральных примеров
    for _ in range(num_samples - 2 * (num_samples // 3)):
        context_type = random.choice(list(neutral_contexts.keys()))
        template = random.choice(neutral_contexts[context_type])
        
        text = template
        replacements = {
            '{item}': random.choice(neutral_item),
            '{feature}': random.choice(neutral_feature),
            '{performance}': random.choice(neutral_performance),
            '{time}': random.choice(neutral_time),
            '{observation}': random.choice(neutral_observation),
            '{price}': random.choice(neutral_price),
            '{standard}': random.choice(neutral_standard),
            '{pros}': random.choice(neutral_pros),
            '{cons}': random.choice(neutral_cons),
            '{purpose}': random.choice(neutral_purpose),
            '{summary}': random.choice(neutral_summary),
            '{day}': random.choice(neutral_day),
            '{weather}': random.choice(neutral_weather),
            '{plans}': random.choice(neutral_plans),
            '{routine}': random.choice(neutral_routine),
            '{status}': random.choice(neutral_status),
            '{task}': random.choice(neutral_task),
            '{next}': random.choice(neutral_next),
            '{event}': random.choice(neutral_event),
            '{normally}': random.choice(neutral_normally),
            '{expected}': random.choice(neutral_expected),
            '{special}': random.choice(neutral_special),
            '{behavior}': random.choice(neutral_behavior),
            '{normal}': random.choice(neutral_normal),
            '{happening}': random.choice(neutral_happening),
            '{continues}': random.choice(neutral_continues),
            '{topic}': random.choice(neutral_topic),
            '{interesting}': random.choice(neutral_interesting),
            '{reaction}': random.choice(neutral_reaction)
        }
        
        for key, value in replacements.items():
            text = text.replace(key, value)
        
        data.append({"text": text, "label": "neutral"})
    
    df = pd.DataFrame(data)
    return df.sample(frac=1).reset_index(drop=True)

def preprocess_text(text):
    """Продвинутая предобработка текста"""
    if pd.isna(text):
        return ""
    
    text = str(text)
    
    # Сохраняем эмодзи и знаки препинания (они важны для эмоций!)
    # Только базовая нормализация
    text = re.sub(r'http\S+|www\S+', '', text)  # Удаляем URL
    text = re.sub(r'\s+', ' ', text)  # Нормализуем пробелы
    
    # НЕ приводим к нижнему регистру - капс тоже передает эмоции
    # НЕ удаляем знаки препинания - они важны
    
    return text.strip()

def create_advanced_model(vocab_size, max_length):
    """Создание продвинутой модели"""
    model = Sequential([
        # Эмбеддинги
        Embedding(vocab_size, EMBEDDING_DIM, input_length=max_length),
        
        # Сверточный слой для локальных паттернов
        Conv1D(128, 5, activation='relu'),
        GlobalMaxPooling1D(),
        
        # Полносвязные слои
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.3),
        
        # Выходной слой
        Dense(3, activation='softmax')
    ])
    
    # Компиляция с продвинутыми настройками
    optimizer = Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def main():
    print("=" * 50)
    print("ПРОДВИНУТОЕ ОБУЧЕНИЕ МОДЕЛИ")
    print("=" * 50)
    
    # 1. Создание качественного датасета
    print("\n1. Создание качественного датасета...")
    df = create_quality_dataset(50000)  # Большой датасет
    print(f"   Создано {len(df)} разнообразных примеров")
    
    # 2. Предобработка
    print("\n2. Предобработка текстов...")
    df['processed_text'] = df['text'].apply(preprocess_text)
    
    # 3. Кодирование меток
    print("\n3. Кодирование меток...")
    label_encoder = LabelEncoder()
    df['label_encoded'] = label_encoder.fit_transform(df['label'])
    
    # Проверка баланса классов
    print("\n   Распределение классов:")
    for label in df['label'].unique():
        count = len(df[df['label'] == label])
        print(f"   {label}: {count} ({count/len(df)*100:.1f}%)")
    
    # 4. Разделение данных
    print("\n4. Разделение на обучающую и тестовую выборки...")
    X_train, X_test, y_train, y_test = train_test_split(
        df['processed_text'], 
        df['label_encoded'],
        test_size=0.2,
        random_state=42,
        stratify=df['label_encoded']
    )
    
    print(f"   Обучающая выборка: {len(X_train)}")
    print(f"   Тестовая выборка: {len(X_test)}")
    
    # 5. Токенизация
    print("\n5. Токенизация...")
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token='<OOV>')
    tokenizer.fit_on_texts(X_train)
    
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    
    X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LENGTH, padding='post', truncating='post')
    X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_LENGTH, padding='post', truncating='post')
    
    vocab_size = min(len(tokenizer.word_index) + 1, MAX_WORDS)
    print(f"   Размер словаря: {vocab_size}")
    print(f"   Уникальных слов: {len(tokenizer.word_index)}")
    
    # 6. Создание модели
    print("\n6. Создание продвинутой модели...")
    model = create_advanced_model(vocab_size, MAX_LENGTH)
    model.summary()
    
    # 7. Веса классов для балансировки
    class_weights = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(enumerate(class_weights))
    
    # 8. Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            str(MODELS_DIR / 'advanced_model.keras'),
            save_best_only=True,
            monitor='val_accuracy',
            mode='max',
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=0.00001,
            verbose=1
        )
    ]
    
    # 9. Обучение
    print(f"\n7. Обучение модели ({EPOCHS} эпох)...")
    history = model.fit(
        X_train_pad, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=0.15,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1
    )
    
    # 10. Оценка
    print("\n8. Оценка модели...")
    test_loss, test_accuracy = model.evaluate(X_test_pad, y_test, verbose=0)
    print(f"   Точность на тестовой выборке: {test_accuracy:.2%}")
    
    # 11. Детальная оценка
    from sklearn.metrics import classification_report, confusion_matrix
    y_pred = model.predict(X_test_pad)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    print("\n   Детальный отчет:")
    print(classification_report(y_test, y_pred_classes, 
                              target_names=['negative', 'neutral', 'positive']))
    
    # 12. Сохранение
    print("\n9. Сохранение модели и артефактов...")
    # Сохраняем как основную модель
    model.save(MODELS_DIR / 'sentiment_model.keras')
    model.save(MODELS_DIR / 'sentiment_model.h5')
    
    joblib.dump(tokenizer, MODELS_DIR / 'tokenizer.pkl')
    joblib.dump(label_encoder, MODELS_DIR / 'label_encoder.pkl')
    
    # Сохранение конфигурации
    config = {
        'max_words': MAX_WORDS,
        'max_length': MAX_LENGTH,
        'vocab_size': vocab_size,
        'test_accuracy': float(test_accuracy),
        'training_samples': len(X_train),
        'model_type': 'advanced_cnn'
    }
    joblib.dump(config, MODELS_DIR / 'model_config.pkl')
    
    # 13. Тестовые предсказания
    print("\n10. Примеры предсказаний:")
    test_texts = [
        "Отличный товар, всем рекомендую! Качество супер!",
        "Ужасное качество, полное разочарование. Не покупайте!",
        "Нормальный товар за свои деньги. Есть плюсы и минусы.",
        "Просто в восторге! Лучшая покупка в моей жизни! 😊",
        "Кошмар! Хуже не видел! Деньги на ветер! 😡",
        "Пользуюсь неделю. Пока все работает нормально.",
        "СУПЕР!!! ОБОЖАЮ!!! ❤️❤️❤️",
        "не советую никому(( очень расстроен",
        "Заказ пришел вовремя. Упаковка целая. Соответствует описанию."
    ]
    
    for text in test_texts:
        # Предобработка
        processed = preprocess_text(text)
        sequence = tokenizer.texts_to_sequences([processed])
        padded = pad_sequences(sequence, maxlen=MAX_LENGTH)
        
        # Предсказание
        prediction = model.predict(padded, verbose=0)[0]
        predicted_class = np.argmax(prediction)
        label = label_encoder.inverse_transform([predicted_class])[0]
        confidence = prediction[predicted_class]
        
        print(f"\n   Текст: '{text}'")
        print(f"   → {label} (уверенность: {confidence:.2%})")
        print(f"   Детали: neg={prediction[0]:.2%}, neu={prediction[1]:.2%}, pos={prediction[2]:.2%}")
    
    print("\n✅ Обучение завершено успешно!")
    print(f"\nМодель сохранена в: {MODELS_DIR}")
    print("\nТеперь можете запустить бота: python src/bot.py")
    print("Или упрощенного бота: python bot_simple.py")

if __name__ == "__main__":
    main()