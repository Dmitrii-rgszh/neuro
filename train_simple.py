"""
Упрощенное обучение модели без проблем с зависимостями
"""
import sys
sys.path.append('src')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import joblib
import re
from pathlib import Path

# Конфигурация
MAX_WORDS = 10000
MAX_LENGTH = 100
EMBEDDING_DIM = 100
LSTM_UNITS = 64
EPOCHS = 20
BATCH_SIZE = 64

# Пути
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

def preprocess_text(text):
    """Простая предобработка текста"""
    if pd.isna(text):
        return ""
    
    text = str(text).lower()
    # Удаление URL
    text = re.sub(r'http\S+|www\S+', '', text)
    # Удаление лишних пробелов
    text = re.sub(r'\s+', ' ', text)
    # Удаление спец символов, но оставляем русские буквы
    text = re.sub(r'[^а-яёa-z0-9\s]', ' ', text)
    
    return text.strip()

def create_dataset(num_samples=10000):
    """Создание синтетического датасета"""
    print("Создание датасета...")
    
    # Расширенные примеры для лучшего обучения
    positive_examples = [
        "отличный товар всем рекомендую",
        "очень доволен покупкой спасибо",
        "превосходное качество буду заказывать еще",
        "лучший сервис который я видел",
        "великолепно просто супер",
        "замечательный продукт стоит своих денег",
        "потрясающе очень рад",
        "прекрасная работа молодцы",
        "восхитительно превзошло ожидания",
        "идеально то что нужно"
    ]
    
    negative_examples = [
        "ужасное качество не рекомендую",
        "полное разочарование деньги на ветер",
        "худший товар который я покупал",
        "отвратительный сервис больше не буду",
        "кошмар а не продукт",
        "плохо очень плохо",
        "не стоит своих денег",
        "разочарован полностью",
        "никому не советую покупать",
        "провал полный провал"
    ]
    
    neutral_examples = [
        "нормальный товар ничего особенного",
        "обычное качество как везде",
        "средний продукт есть плюсы и минусы",
        "пойдет для своей цены",
        "стандартный товар без изысков",
        "неплохо но можно лучше",
        "соответствует описанию",
        "обычный сервис ничего выдающегося",
        "приемлемое качество",
        "нормально работает"
    ]
    
    data = []
    samples_per_class = num_samples // 3
    
    # Генерация с вариациями
    for _ in range(samples_per_class):
        # Позитивные
        base = np.random.choice(positive_examples)
        if np.random.random() > 0.5:
            base = base + " " + np.random.choice(["!", "!!", ""])
        data.append({"text": base, "label": "positive"})
        
        # Негативные
        base = np.random.choice(negative_examples)
        if np.random.random() > 0.5:
            base = base + " " + np.random.choice(["((", ":(", ""])
        data.append({"text": base, "label": "negative"})
        
        # Нейтральные
        base = np.random.choice(neutral_examples)
        data.append({"text": base, "label": "neutral"})
    
    df = pd.DataFrame(data)
    return df.sample(frac=1).reset_index(drop=True)

def create_model(vocab_size, max_length):
    """Создание модели"""
    model = Sequential([
        Embedding(vocab_size, EMBEDDING_DIM),
        Bidirectional(LSTM(LSTM_UNITS, return_sequences=True)),
        Dropout(0.5),
        LSTM(LSTM_UNITS // 2),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dropout(0.5),
        Dense(3, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def main():
    print("=" * 50)
    print("ОБУЧЕНИЕ МОДЕЛИ АНАЛИЗА НАСТРОЕНИЙ")
    print("=" * 50)
    
    # 1. Создание датасета
    print("\n1. Создание датасета...")
    df = create_dataset(12000)
    print(f"   Создано {len(df)} примеров")
    
    # 2. Предобработка
    print("\n2. Предобработка текстов...")
    df['processed_text'] = df['text'].apply(preprocess_text)
    
    # 3. Кодирование меток
    print("\n3. Кодирование меток...")
    label_encoder = LabelEncoder()
    df['label_encoded'] = label_encoder.fit_transform(df['label'])
    
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
    tokenizer = Tokenizer(num_words=MAX_WORDS)
    tokenizer.fit_on_texts(X_train)
    
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    
    X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LENGTH)
    X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_LENGTH)
    
    vocab_size = len(tokenizer.word_index) + 1
    print(f"   Размер словаря: {vocab_size}")
    
    # 6. Создание модели
    print("\n6. Создание модели...")
    model = create_model(min(vocab_size, MAX_WORDS), MAX_LENGTH)
    print(f"   Параметры модели: {model.count_params():,}")
    
    # 7. Обучение
    print(f"\n7. Обучение модели ({EPOCHS} эпох)...")
    
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        ),
        ModelCheckpoint(
            str(MODELS_DIR / 'best_model.keras'),
            save_best_only=True,
            monitor='val_accuracy',
            mode='max'
        )
    ]
    
    history = model.fit(
        X_train_pad, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=0.15,
        callbacks=callbacks,
        verbose=1
    )
    
    # 8. Оценка
    print("\n8. Оценка модели...")
    test_loss, test_accuracy = model.evaluate(X_test_pad, y_test, verbose=0)
    print(f"   Точность на тестовой выборке: {test_accuracy:.2%}")
    
    # 9. Сохранение
    print("\n9. Сохранение модели и артефактов...")
    model.save(MODELS_DIR / 'sentiment_model.keras')
    joblib.dump(tokenizer, MODELS_DIR / 'tokenizer.pkl')
    joblib.dump(label_encoder, MODELS_DIR / 'label_encoder.pkl')
    
    # Сохранение конфигурации
    config = {
        'max_words': MAX_WORDS,
        'max_length': MAX_LENGTH,
        'vocab_size': vocab_size,
        'test_accuracy': float(test_accuracy)
    }
    joblib.dump(config, MODELS_DIR / 'model_config.pkl')
    
    # 10. Тестовые предсказания
    print("\n10. Примеры предсказаний:")
    test_texts = [
        "Отличный товар, всем рекомендую!",
        "Ужасное качество, не покупайте",
        "Нормально, ничего особенного",
        "Супер! Очень доволен!",
        "Полное разочарование",
        "Средний продукт за свои деньги"
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
        
        print(f"   '{text}'")
        print(f"     → {label} ({confidence:.2%})")
    
    print("\n✅ Обучение завершено успешно!")
    print(f"\nМодель сохранена в: {MODELS_DIR}")
    print("\nТеперь можете запустить бота: python src/bot.py")

if __name__ == "__main__":
    main()