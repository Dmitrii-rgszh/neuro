"""
Быстрое обучение для тестирования
"""
import sys
sys.path.append('src')

from data_loader import DataLoader
from model import SentimentModel
from config import MODELS_DIR

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
import joblib

def create_simple_model(vocab_size=1000, max_length=50):
    """Создание очень простой модели"""
    model = Sequential([
        Embedding(vocab_size, 50),  # Убрали input_length
        LSTM(32),
        Dropout(0.5),
        Dense(3, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def quick_train():
    """Быстрое обучение на маленьком датасете"""
    print("=" * 50)
    print("БЫСТРОЕ ОБУЧЕНИЕ ДЛЯ ТЕСТИРОВАНИЯ")
    print("=" * 50)
    
    try:
        # 1. Создание простого датасета
        print("\n1. Создание данных...")
        loader = DataLoader()
        df = loader.create_synthetic_dataset(num_samples=300)
        
        # 2. Предобработка
        print("\n2. Предобработка текстов...")
        df['processed_text'] = df['text'].apply(loader.preprocess_text)
        
        # 3. Кодирование меток
        print("\n3. Кодирование меток...")
        df['label_encoded'] = loader.label_encoder.fit_transform(df['label'])
        
        # 4. Разделение на train/test
        X_train, X_test, y_train, y_test = train_test_split(
            df['processed_text'], df['label_encoded'], 
            test_size=0.2, random_state=42, stratify=df['label_encoded']
        )
        
        print(f"   Обучающая выборка: {len(X_train)}")
        print(f"   Тестовая выборка: {len(X_test)}")
        
        # 5. Токенизация
        print("\n4. Токенизация...")
        tokenizer = Tokenizer(num_words=1000)
        tokenizer.fit_on_texts(X_train)
        
        X_train_seq = tokenizer.texts_to_sequences(X_train)
        X_test_seq = tokenizer.texts_to_sequences(X_test)
        
        X_train_pad = pad_sequences(X_train_seq, maxlen=50)
        X_test_pad = pad_sequences(X_test_seq, maxlen=50)
        
        # 6. Создание простой модели
        print("\n5. Создание модели...")
        model = create_simple_model(vocab_size=1000, max_length=50)
        # Строим модель перед подсчетом параметров
        model.build(input_shape=(None, 50))
        print(f"   Параметры модели: {model.count_params():,}")
        
        # 7. Обучение (всего 5 эпох)
        print("\n6. Обучение (5 эпох)...")
        history = model.fit(
            X_train_pad, y_train,
            batch_size=32,
            epochs=5,
            validation_split=0.2,
            verbose=1
        )
        
        # 8. Оценка
        print("\n7. Оценка модели...")
        loss, accuracy = model.evaluate(X_test_pad, y_test, verbose=0)
        print(f"   Точность на тесте: {accuracy:.2%}")
        
        # 9. Сохранение
        print("\n8. Сохранение модели...")
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model.save(MODELS_DIR / "quick_model.h5")
        joblib.dump(tokenizer, MODELS_DIR / "quick_tokenizer.pkl")
        joblib.dump(loader.label_encoder, MODELS_DIR / "quick_label_encoder.pkl")
        
        # 10. Тест предсказаний
        print("\n9. Тест предсказаний:")
        test_texts = ["Отлично!", "Ужасно!", "Нормально"]
        test_seq = tokenizer.texts_to_sequences(test_texts)
        test_pad = pad_sequences(test_seq, maxlen=50)
        predictions = model.predict(test_pad, verbose=0)
        
        for text, pred in zip(test_texts, predictions):
            pred_class = np.argmax(pred)
            label = loader.label_encoder.inverse_transform([pred_class])[0]
            confidence = pred[pred_class]
            print(f"   '{text}' -> {label} ({confidence:.2%})")
        
        print("\n✅ Быстрое обучение завершено!")
        print("\nТеперь можете запустить полное обучение:")
        print("  python src/train.py")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = quick_train()
    if not success:
        print("\n💡 Подсказки:")
        print("1. Проверьте установку зависимостей: pip install -r requirements.txt")
        print("2. Убедитесь, что структура проекта создана: python setup_project.py")
        print("3. Попробуйте очистить данные: python clean_data.py")