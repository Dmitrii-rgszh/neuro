"""
Архитектура модели для анализа настроений
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
)
import numpy as np
from datetime import datetime

from config import MODEL_CONFIG, MODEL_PATH


class SentimentModel:
    """Модель для анализа настроений с использованием LSTM и внимания"""
    
    def __init__(self, vocab_size: int = None):
        self.vocab_size = vocab_size or MODEL_CONFIG["max_features"]
        self.model = None
        
    def build_model(self):
        """Построение архитектуры модели с механизмом внимания"""
        # Входной слой
        inputs = layers.Input(shape=(MODEL_CONFIG["max_sequence_length"],))
        
        # Embedding слой
        embedding = layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=MODEL_CONFIG["embedding_dim"],
            input_length=MODEL_CONFIG["max_sequence_length"],
            mask_zero=True
        )(inputs)
        
        # Dropout для регуляризации
        embedding = layers.SpatialDropout1D(0.3)(embedding)
        
        # Bidirectional LSTM для лучшего понимания контекста
        lstm_out = layers.Bidirectional(
            layers.LSTM(
                MODEL_CONFIG["lstm_units"],
                return_sequences=True,
                dropout=MODEL_CONFIG["dropout_rate"],
                recurrent_dropout=MODEL_CONFIG["recurrent_dropout"]
            )
        )(embedding)
        
        # Второй LSTM слой
        lstm_out = layers.Bidirectional(
            layers.LSTM(
                MODEL_CONFIG["lstm_units"] // 2,
                return_sequences=True,
                dropout=MODEL_CONFIG["dropout_rate"],
                recurrent_dropout=MODEL_CONFIG["recurrent_dropout"]
            )
        )(lstm_out)
        
        # Механизм внимания (Attention)
        attention = layers.MultiHeadAttention(
            num_heads=4,
            key_dim=MODEL_CONFIG["lstm_units"] // 2
        )(lstm_out, lstm_out)
        
        # GlobalMaxPooling для извлечения важных признаков
        max_pool = layers.GlobalMaxPooling1D()(attention)
        avg_pool = layers.GlobalAveragePooling1D()(attention)
        
        # Конкатенация признаков
        concatenated = layers.concatenate([max_pool, avg_pool])
        
        # Полносвязные слои
        dense = layers.Dense(256, activation='relu')(concatenated)
        dense = layers.BatchNormalization()(dense)
        dense = layers.Dropout(MODEL_CONFIG["dropout_rate"])(dense)
        
        dense = layers.Dense(128, activation='relu')(dense)
        dense = layers.BatchNormalization()(dense)
        dense = layers.Dropout(MODEL_CONFIG["dropout_rate"])(dense)
        
        # Выходной слой
        outputs = layers.Dense(
            MODEL_CONFIG["num_classes"], 
            activation='softmax'
        )(dense)
        
        # Создание модели
        self.model = keras.Model(inputs=inputs, outputs=outputs)
        
        # Компиляция модели
        self.compile_model()
        
        return self.model
    
    def build_simple_model(self):
        """Упрощенная архитектура для быстрого обучения"""
        # Входной слой
        inputs = layers.Input(shape=(MODEL_CONFIG["max_sequence_length"],))
        
        # Embedding слой
        embedding = layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=MODEL_CONFIG["embedding_dim"],
            input_length=MODEL_CONFIG["max_sequence_length"],
            mask_zero=True
        )(inputs)
        
        # Dropout для регуляризации
        embedding = layers.SpatialDropout1D(0.3)(embedding)
        
        # Один LSTM слой
        lstm_out = layers.LSTM(
            MODEL_CONFIG["lstm_units"],
            dropout=MODEL_CONFIG["dropout_rate"],
            recurrent_dropout=MODEL_CONFIG["recurrent_dropout"]
        )(embedding)
        
        # Полносвязные слои
        dense = layers.Dense(64, activation='relu')(lstm_out)
        dense = layers.Dropout(MODEL_CONFIG["dropout_rate"])(dense)
        
        # Выходной слой
        outputs = layers.Dense(
            MODEL_CONFIG["num_classes"], 
            activation='softmax'
        )(dense)
        
        # Создание модели
        self.model = keras.Model(inputs=inputs, outputs=outputs)
        
        # Компиляция модели
        self.compile_model()
        
        return self.model
        """Альтернативная архитектура CNN + LSTM"""
        inputs = layers.Input(shape=(MODEL_CONFIG["max_sequence_length"],))
        
        # Embedding
        embedding = layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=MODEL_CONFIG["embedding_dim"],
            input_length=MODEL_CONFIG["max_sequence_length"]
        )(inputs)
        
        # CNN слои для извлечения локальных признаков
        conv1 = layers.Conv1D(128, 3, activation='relu', padding='same')(embedding)
        conv1 = layers.MaxPooling1D(2)(conv1)
        
        conv2 = layers.Conv1D(64, 3, activation='relu', padding='same')(conv1)
        conv2 = layers.MaxPooling1D(2)(conv2)
        
        # LSTM для последовательных зависимостей
        lstm = layers.LSTM(
            MODEL_CONFIG["lstm_units"],
            dropout=MODEL_CONFIG["dropout_rate"],
            recurrent_dropout=MODEL_CONFIG["recurrent_dropout"]
        )(conv2)
        
        # Полносвязные слои
        dense = layers.Dense(128, activation='relu')(lstm)
        dense = layers.Dropout(MODEL_CONFIG["dropout_rate"])(dense)
        
        outputs = layers.Dense(
            MODEL_CONFIG["num_classes"], 
            activation='softmax'
        )(dense)
        
        self.model = keras.Model(inputs=inputs, outputs=outputs)
        self.compile_model()
        
        return self.model
    
    def compile_model(self):
        """Компиляция модели с оптимизатором и метриками"""
        # Оптимизатор с настраиваемой скоростью обучения
        optimizer = keras.optimizers.Adam(
            learning_rate=MODEL_CONFIG["learning_rate"]
        )
        
        # Компиляция
        self.model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.SparseCategoricalAccuracy(name='acc'),
                keras.metrics.SparseTopKCategoricalAccuracy(k=2, name='top2_acc')
            ]
        )
    
    def get_callbacks(self, patience: int = 5):
        """Получение callbacks для обучения"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        callbacks = [
            # Сохранение лучшей модели
            ModelCheckpoint(
                filepath=str(MODEL_PATH / f"best_model_{timestamp}.h5"),
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                mode='max',
                verbose=1
            ),
            
            # Ранняя остановка при отсутствии улучшений
            EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Уменьшение learning rate при застое
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=1
            ),
            
            # TensorBoard для визуализации
            TensorBoard(
                log_dir=f"logs/fit/{timestamp}",
                histogram_freq=1,
                write_graph=True,
                update_freq='epoch'
            )
        ]
        
        return callbacks
    
    def train(self, X_train, y_train, X_val, y_val):
        """Обучение модели"""
        if self.model is None:
            raise ValueError("Модель не построена. Вызовите build_model() сначала.")
        
        # Получение callbacks
        callbacks = self.get_callbacks()
        
        # Обучение
        history = self.model.fit(
            X_train, y_train,
            batch_size=MODEL_CONFIG["batch_size"],
            epochs=MODEL_CONFIG["epochs"],
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1,
            class_weight=self._compute_class_weights(y_train)
        )
        
        return history
    
    def _compute_class_weights(self, y_train):
        """Вычисление весов классов для балансировки"""
        from sklearn.utils.class_weight import compute_class_weight
        
        classes = np.unique(y_train)
        class_weights = compute_class_weight(
            'balanced',
            classes=classes,
            y=y_train
        )
        
        return dict(zip(classes, class_weights))
    
    def evaluate(self, X_test, y_test):
        """Оценка модели"""
        results = self.model.evaluate(X_test, y_test, verbose=1)
        
        metrics_dict = dict(zip(self.model.metrics_names, results))
        
        # Подробная оценка
        predictions = self.model.predict(X_test)
        predicted_classes = np.argmax(predictions, axis=1)
        
        from sklearn.metrics import classification_report, confusion_matrix
        
        print("\nClassification Report:")
        print(classification_report(
            y_test, 
            predicted_classes, 
            target_names=MODEL_CONFIG["class_names"]
        ))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, predicted_classes))
        
        return metrics_dict
    
    def save_model(self, filepath: str = None):
        """Сохранение модели"""
        if filepath is None:
            filepath = MODEL_PATH / "sentiment_model.h5"
        
        self.model.save(filepath)
        print(f"Модель сохранена: {filepath}")
    
    def load_model(self, filepath: str = None):
        """Загрузка модели"""
        if filepath is None:
            filepath = MODEL_PATH / "sentiment_model.h5"
        
        self.model = keras.models.load_model(filepath)
        print(f"Модель загружена: {filepath}")
        
        return self.model