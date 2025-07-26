"""
Улучшенная архитектура модели с ансамблем для высокоточного анализа настроений
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, 
    TensorBoard, LearningRateScheduler
)
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple
import joblib

from config import MODEL_CONFIG, MODEL_PATHS, VALIDATION_CONFIG


class AttentionLayer(layers.Layer):
    """Кастомный слой внимания"""
    
    def __init__(self, units, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.units = units
        self.W = layers.Dense(units)
        self.U = layers.Dense(units)
        self.V = layers.Dense(1)
        
    def call(self, inputs):
        # inputs shape: (batch_size, timesteps, features)
        score = tf.nn.tanh(self.W(inputs))
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        context_vector = attention_weights * inputs
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector
    
    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config


class SentimentModelEnsemble:
    """Ансамбль моделей для анализа настроений"""
    
    def __init__(self, vocab_size: int = None):
        self.vocab_size = vocab_size or MODEL_CONFIG["max_features"]
        self.models = {}
        self.ensemble_model = None
        
    def build_lstm_model(self, name="lstm") -> keras.Model:
        """Построение LSTM модели с вниманием"""
        inputs = layers.Input(shape=(MODEL_CONFIG["max_sequence_length"],), name="input")
        
        # Embedding слой
        embedding = layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=MODEL_CONFIG["embedding_dim"],
            input_length=MODEL_CONFIG["max_sequence_length"],
            mask_zero=True,
            name="embedding"
        )(inputs)
        
        # Spatial Dropout
        embedding = layers.SpatialDropout1D(MODEL_CONFIG["spatial_dropout"])(embedding)
        
        # Bidirectional LSTM слои
        lstm1 = layers.Bidirectional(
            layers.LSTM(
                MODEL_CONFIG["lstm_units"],
                return_sequences=True,
                dropout=MODEL_CONFIG["dropout_rate"],
                recurrent_dropout=MODEL_CONFIG["recurrent_dropout"],
                kernel_regularizer=regularizers.l2(MODEL_CONFIG["l2_reg"])
            ),
            name="bilstm_1"
        )(embedding)
        
        lstm2 = layers.Bidirectional(
            layers.LSTM(
                MODEL_CONFIG["lstm_units"] // 2,
                return_sequences=True,
                dropout=MODEL_CONFIG["dropout_rate"],
                recurrent_dropout=MODEL_CONFIG["recurrent_dropout"],
                kernel_regularizer=regularizers.l2(MODEL_CONFIG["l2_reg"])
            ),
            name="bilstm_2"
        )(lstm1)
        
        # Attention mechanism
        attention = AttentionLayer(MODEL_CONFIG["lstm_units"], name="attention")(lstm2)
        
        # Global pooling для дополнительных признаков
        max_pool = layers.GlobalMaxPooling1D(name="global_max_pool")(lstm2)
        avg_pool = layers.GlobalAveragePooling1D(name="global_avg_pool")(lstm2)
        
        # Конкатенация всех признаков
        concatenated = layers.concatenate([attention, max_pool, avg_pool], name="feature_concat")
        
        # Полносвязные слои
        dense = concatenated
        for i, units in enumerate(MODEL_CONFIG["dense_units"]):
            dense = layers.Dense(
                units, 
                activation='relu',
                kernel_regularizer=regularizers.l1_l2(
                    l1=MODEL_CONFIG["l1_reg"], 
                    l2=MODEL_CONFIG["l2_reg"]
                ),
                name=f"dense_{i+1}"
            )(dense)
            dense = layers.BatchNormalization(name=f"batch_norm_{i+1}")(dense)
            dense = layers.Dropout(MODEL_CONFIG["dropout_rate"], name=f"dropout_{i+1}")(dense)
        
        # Выходной слой
        outputs = layers.Dense(
            MODEL_CONFIG["num_classes"],
            activation='softmax',
            name="output"
        )(dense)
        
        model = keras.Model(inputs=inputs, outputs=outputs, name=name)
        return model
    
    def build_cnn_lstm_model(self, name="cnn_lstm") -> keras.Model:
        """Построение CNN+LSTM модели"""
        inputs = layers.Input(shape=(MODEL_CONFIG["max_sequence_length"],), name="input")
        
        # Embedding
        embedding = layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=MODEL_CONFIG["embedding_dim"],
            input_length=MODEL_CONFIG["max_sequence_length"],
            name="embedding"
        )(inputs)
        
        # CNN слои для извлечения локальных признаков
        conv_blocks = []
        filter_sizes = [2, 3, 4, 5]  # Разные размеры фильтров
        
        for filter_size in filter_sizes:
            conv = layers.Conv1D(
                filters=128,
                kernel_size=filter_size,
                activation='relu',
                padding='same',
                name=f"conv_{filter_size}"
            )(embedding)
            conv = layers.MaxPooling1D(2, name=f"maxpool_{filter_size}")(conv)
            conv_blocks.append(conv)
        
        # Объединение CNN признаков
        if len(conv_blocks) > 1:
            cnn_features = layers.concatenate(conv_blocks, name="cnn_concat")
        else:
            cnn_features = conv_blocks[0]
        
        # LSTM для последовательных зависимостей
        lstm = layers.Bidirectional(
            layers.LSTM(
                MODEL_CONFIG["lstm_units"] // 2,
                return_sequences=True,
                dropout=MODEL_CONFIG["dropout_rate"]
            ),
            name="bilstm"
        )(cnn_features)
        
        # Attention
        attention = AttentionLayer(MODEL_CONFIG["lstm_units"] // 2, name="attention")(lstm)
        
        # Global pooling
        max_pool = layers.GlobalMaxPooling1D(name="global_max_pool")(lstm)
        avg_pool = layers.GlobalAveragePooling1D(name="global_avg_pool")(lstm)
        
        # Объединение признаков
        features = layers.concatenate([attention, max_pool, avg_pool], name="feature_concat")
        
        # Полносвязные слои
        dense = features
        for i, units in enumerate([256, 128]):
            dense = layers.Dense(units, activation='relu', name=f"dense_{i+1}")(dense)
            dense = layers.BatchNormalization(name=f"batch_norm_{i+1}")(dense)
            dense = layers.Dropout(MODEL_CONFIG["dropout_rate"], name=f"dropout_{i+1}")(dense)
        
        # Выход
        outputs = layers.Dense(MODEL_CONFIG["num_classes"], activation='softmax', name="output")(dense)
        
        model = keras.Model(inputs=inputs, outputs=outputs, name=name)
        return model
    
    def build_transformer_model(self, name="transformer") -> keras.Model:
        """Построение Transformer-подобной модели"""
        inputs = layers.Input(shape=(MODEL_CONFIG["max_sequence_length"],), name="input")
        
        # Embedding
        embedding = layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=MODEL_CONFIG["embedding_dim"],
            input_length=MODEL_CONFIG["max_sequence_length"],
            name="embedding"
        )(inputs)
        
        # Positional encoding (упрощенная версия)
        positions = tf.range(start=0, limit=MODEL_CONFIG["max_sequence_length"], delta=1)
        positions = layers.Embedding(
            input_dim=MODEL_CONFIG["max_sequence_length"],
            output_dim=MODEL_CONFIG["embedding_dim"],
            name="positional_embedding"
        )(positions)
        
        # Добавление позиционного кодирования
        x = embedding + positions
        
        # Multi-head self-attention
        attention = layers.MultiHeadAttention(
            num_heads=MODEL_CONFIG["attention_heads"],
            key_dim=MODEL_CONFIG["attention_key_dim"],
            name="multihead_attention"
        )(x, x)
        
        # Add & Norm
        x = layers.Add(name="add_1")([x, attention])
        x = layers.LayerNormalization(name="layer_norm_1")(x)
        
        # Feed Forward Network
        ffn = layers.Dense(MODEL_CONFIG["embedding_dim"] * 2, activation='relu', name="ffn_1")(x)
        ffn = layers.Dense(MODEL_CONFIG["embedding_dim"], name="ffn_2")(ffn)
        
        # Add & Norm
        x = layers.Add(name="add_2")([x, ffn])
        x = layers.LayerNormalization(name="layer_norm_2")(x)
        
        # Global pooling
        pooled = layers.GlobalAveragePooling1D(name="global_avg_pool")(x)
        
        # Classification head
        dense = layers.Dense(256, activation='relu', name="dense_1")(pooled)
        dense = layers.Dropout(MODEL_CONFIG["dropout_rate"], name="dropout_1")(dense)
        dense = layers.Dense(128, activation='relu', name="dense_2")(dense)
        dense = layers.Dropout(MODEL_CONFIG["dropout_rate"], name="dropout_2")(dense)
        
        outputs = layers.Dense(MODEL_CONFIG["num_classes"], activation='softmax', name="output")(dense)
        
        model = keras.Model(inputs=inputs, outputs=outputs, name=name)
        return model
    
    def build_all_models(self) -> Dict[str, keras.Model]:
        """Построение всех моделей ансамбля"""
        print("Строим модели ансамбля...")
        
        # LSTM модель
        self.models["lstm"] = self.build_lstm_model("lstm")
        self.compile_model(self.models["lstm"])
        print(f"✅ LSTM модель: {self.models['lstm'].count_params():,} параметров")
        
        # CNN+LSTM модель
        self.models["cnn_lstm"] = self.build_cnn_lstm_model("cnn_lstm")
        self.compile_model(self.models["cnn_lstm"])
        print(f"✅ CNN+LSTM модель: {self.models['cnn_lstm'].count_params():,} параметров")
        
        # Transformer модель
        self.models["transformer"] = self.build_transformer_model("transformer")
        self.compile_model(self.models["transformer"])
        print(f"✅ Transformer модель: {self.models['transformer'].count_params():,} параметров")
        
        return self.models
    
    def build_ensemble_model(self) -> keras.Model:
        """Построение ансамбля из обученных моделей"""
        if not self.models:
            raise ValueError("Сначала постройте и обучите базовые модели")
        
        # Входы
        inputs = layers.Input(shape=(MODEL_CONFIG["max_sequence_length"],), name="ensemble_input")
        
        # Получение предсказаний от каждой модели
        model_outputs = []
        for name, model in self.models.items():
            # Заморозка весов базовых моделей
            model.trainable = False
            output = model(inputs)
            model_outputs.append(output)
        
        # Стекинг предсказаний
        if len(model_outputs) > 1:
            stacked = layers.concatenate(model_outputs, name="stack_predictions")
        else:
            stacked = model_outputs[0]
        
        # Meta-learner
        meta = layers.Dense(64, activation='relu', name="meta_dense_1")(stacked)
        meta = layers.Dropout(0.3, name="meta_dropout_1")(meta)
        meta = layers.Dense(32, activation='relu', name="meta_dense_2")(meta)
        meta = layers.Dropout(0.3, name="meta_dropout_2")(meta)
        
        # Финальный выход
        ensemble_output = layers.Dense(
            MODEL_CONFIG["num_classes"], 
            activation='softmax', 
            name="ensemble_output"
        )(meta)
        
        self.ensemble_model = keras.Model(inputs=inputs, outputs=ensemble_output, name="ensemble")
        self.compile_model(self.ensemble_model)
        
        print(f"✅ Ансамбль модель: {self.ensemble_model.count_params():,} параметров")
        return self.ensemble_model
    
    def compile_model(self, model: keras.Model):
        """Компиляция модели с оптимизатором"""
        # Scheduler для learning rate
        if MODEL_CONFIG["learning_rate_schedule"]:
            lr_schedule = keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=MODEL_CONFIG["learning_rate"],
                decay_steps=1000,
                decay_rate=0.9
            )
            optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
        else:
            optimizer = keras.optimizers.Adam(learning_rate=MODEL_CONFIG["learning_rate"])
        
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def get_callbacks(self, model_name: str) -> List:
        """Получение callbacks для обучения"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        callbacks = [
            # Сохранение лучшей модели
            ModelCheckpoint(
                filepath=str(MODEL_PATHS[f"{model_name}_model"]),
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                mode='max',
                verbose=1
            ),
            
            # Ранняя остановка
            EarlyStopping(
                monitor='val_loss',
                patience=MODEL_CONFIG["early_stopping_patience"],
                restore_best_weights=True,
                verbose=1
            ),
            
            # Уменьшение learning rate
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=MODEL_CONFIG["reduce_lr_factor"],
                patience=MODEL_CONFIG["reduce_lr_patience"],
                min_lr=MODEL_CONFIG["min_lr"],
                verbose=1
            ),
            
            # TensorBoard
            TensorBoard(
                log_dir=f"logs/{model_name}_{timestamp}",
                histogram_freq=1,
                write_graph=True,
                update_freq='epoch'
            )
        ]
        
        return callbacks
    
    def train_model(self, model: keras.Model, model_name: str, 
                   X_train, y_train, X_val, y_val):
        """Обучение одной модели"""
        print(f"\n🚀 Обучение модели {model_name}...")
        
        # Вычисление весов классов
        class_weights = None
        if MODEL_CONFIG["class_weight"]:
            class_weights = self._compute_class_weights(y_train)
        
        # Callbacks
        callbacks = self.get_callbacks(model_name)
        
        # Обучение
        history = model.fit(
            X_train, y_train,
            batch_size=MODEL_CONFIG["batch_size"],
            epochs=MODEL_CONFIG["epochs"],
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
        
        return history
    
    def train_ensemble(self, X_train, y_train, X_val, y_val):
        """Обучение всего ансамбля"""
        histories = {}
        
        # Обучение базовых моделей
        for name, model in self.models.items():
            histories[name] = self.train_model(model, name, X_train, y_train, X_val, y_val)
        
        # Построение и обучение ансамбля
        if MODEL_CONFIG["ensemble"]["enabled"]:
            self.build_ensemble_model()
            histories["ensemble"] = self.train_model(
                self.ensemble_model, "ensemble", X_train, y_train, X_val, y_val
            )
        
        return histories
    
    def _compute_class_weights(self, y_train):
        """Вычисление весов классов"""
        from sklearn.utils.class_weight import compute_class_weight
        
        classes = np.unique(y_train)
        class_weights = compute_class_weight(
            'balanced',
            classes=classes,
            y=y_train
        )
        
        return dict(zip(classes, class_weights))
    
    def evaluate_ensemble(self, X_test, y_test):
        """Комплексная оценка ансамбля"""
        results = {}
        
        for name, model in self.models.items():
            print(f"\n📊 Оценка модели {name}:")
            metrics = model.evaluate(X_test, y_test, verbose=0)
            results[name] = dict(zip(model.metrics_names, metrics))
            
            # Детальная оценка
            predictions = model.predict(X_test, verbose=0)
            predicted_classes = np.argmax(predictions, axis=1)
            
            from sklearn.metrics import classification_report, confusion_matrix
            
            print(f"Точность: {results[name]['accuracy']:.4f}")
            print("\nClassification Report:")
            print(classification_report(
                y_test, predicted_classes,
                target_names=MODEL_CONFIG["class_names"]
            ))
        
        # Оценка ансамбля
        if self.ensemble_model:
            print(f"\n🎯 Оценка ансамбля:")
            metrics = self.ensemble_model.evaluate(X_test, y_test, verbose=0)
            results["ensemble"] = dict(zip(self.ensemble_model.metrics_names, metrics))
            print(f"Точность ансамбля: {results['ensemble']['accuracy']:.4f}")
        
        return results
    
    def save_all_models(self):
        """Сохранение всех моделей"""
        for name, model in self.models.items():
            model_path = MODEL_PATHS[f"{name}_model"]
            model.save(str(model_path))
            print(f"💾 {name} модель сохранена: {model_path}")
        
        if self.ensemble_model:
            ensemble_path = MODEL_PATHS["ensemble_model"]
            self.ensemble_model.save(str(ensemble_path))
            print(f"💾 Ансамбль сохранен: {ensemble_path}")
    
    def load_models(self):
        """Загрузка всех сохраненных моделей"""
        for name in ["lstm", "cnn_lstm", "transformer"]:
            model_path = MODEL_PATHS[f"{name}_model"]
            if model_path.exists():
                self.models[name] = keras.models.load_model(
                    str(model_path),
                    custom_objects={"AttentionLayer": AttentionLayer}
                )
                print(f"📂 {name} модель загружена")
        
        # Загрузка ансамбля
        ensemble_path = MODEL_PATHS["ensemble_model"]
        if ensemble_path.exists():
            self.ensemble_model = keras.models.load_model(
                str(ensemble_path),
                custom_objects={"AttentionLayer": AttentionLayer}
            )
            print(f"📂 Ансамбль загружен")
    
    def predict_ensemble(self, X, use_ensemble=True):
        """Предсказание с использованием ансамбля"""
        if use_ensemble and self.ensemble_model:
            return self.ensemble_model.predict(X, verbose=0)
        
        # Мягкое голосование базовых моделей
        predictions = []
        for name, model in self.models.items():
            pred = model.predict(X, verbose=0)
            predictions.append(pred)
        
        if predictions:
            # Усреднение предсказаний
            ensemble_pred = np.mean(predictions, axis=0)
            return ensemble_pred
        
        return None


class AdvancedSentimentTrainer:
    """Продвинутый тренер для обучения ансамбля моделей"""
    
    def __init__(self):
        self.model_ensemble = None
        self.tokenizer = None
        self.label_encoder = None
        self.training_history = {}
        self.validation_results = {}
    
    def prepare_advanced_features(self, texts):
        """Подготовка дополнительных признаков"""
        features = []
        
        for text in texts:
            text_features = {
                'length': len(text.split()),
                'exclamation_count': text.count('!'),
                'question_count': text.count('?'),
                'caps_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0,
                'emoji_count': len([c for c in text if ord(c) > 127]),
            }
            features.append(list(text_features.values()))
        
        return np.array(features)
    
    def augment_data(self, texts, labels, augmentation_factor=0.2):
        """Аугментация данных для увеличения разнообразия"""
        if not MODEL_CONFIG["data_augmentation"]["enabled"]:
            return texts, labels
        
        augmented_texts = []
        augmented_labels = []
        
        import random
        import nltk
        from nltk.corpus import wordnet
        
        def get_synonyms(word):
            """Получение синонимов слова"""
            synonyms = set()
            try:
                for syn in wordnet.synsets(word, lang='rus'):
                    for lemma in syn.lemmas(lang='rus'):
                        synonyms.add(lemma.name().replace('_', ' '))
            except:
                pass
            return list(synonyms)
        
        def synonym_replacement(text, n=1):
            """Замена слов синонимами"""
            words = text.split()
            new_words = words.copy()
            random_word_list = list(set([word for word in words if len(word) > 3]))
            random.shuffle(random_word_list)
            
            num_replaced = 0
            for random_word in random_word_list:
                synonyms = get_synonyms(random_word)
                if len(synonyms) >= 1:
                    synonym = random.choice(synonyms)
                    new_words = [synonym if word == random_word else word for word in new_words]
                    num_replaced += 1
                if num_replaced >= n:
                    break
            
            return ' '.join(new_words)
        
        def random_insertion(text, n=1):
            """Случайная вставка слов"""
            words = text.split()
            for _ in range(n):
                if words:
                    new_word = random.choice(words)
                    random_idx = random.randint(0, len(words))
                    words.insert(random_idx, new_word)
            return ' '.join(words)
        
        def random_swap(text, n=1):
            """Случайная перестановка слов"""
            words = text.split()
            for _ in range(n):
                if len(words) >= 2:
                    idx1, idx2 = random.sample(range(len(words)), 2)
                    words[idx1], words[idx2] = words[idx2], words[idx1]
            return ' '.join(words)
        
        def random_deletion(text, p=0.1):
            """Случайное удаление слов"""
            words = text.split()
            if len(words) == 1:
                return text
            
            new_words = []
            for word in words:
                if random.uniform(0, 1) > p:
                    new_words.append(word)
            
            if len(new_words) == 0:
                return random.choice(words)
            
            return ' '.join(new_words)
        
        # Применение аугментации
        num_aug = int(len(texts) * augmentation_factor)
        indices = random.sample(range(len(texts)), min(num_aug, len(texts)))
        
        for idx in indices:
            original_text = texts[idx]
            label = labels[idx]
            
            # Применение различных техник аугментации
            if random.random() < MODEL_CONFIG["data_augmentation"]["synonym_replacement"]:
                aug_text = synonym_replacement(original_text, n=2)
                augmented_texts.append(aug_text)
                augmented_labels.append(label)
            
            if random.random() < MODEL_CONFIG["data_augmentation"]["random_insertion"]:
                aug_text = random_insertion(original_text, n=1)
                augmented_texts.append(aug_text)
                augmented_labels.append(label)
            
            if random.random() < MODEL_CONFIG["data_augmentation"]["random_swap"]:
                aug_text = random_swap(original_text, n=1)
                augmented_texts.append(aug_text)
                augmented_labels.append(label)
            
            if random.random() < MODEL_CONFIG["data_augmentation"]["random_deletion"]:
                aug_text = random_deletion(original_text, p=0.1)
                augmented_texts.append(aug_text)
                augmented_labels.append(label)
        
        # Объединение оригинальных и аугментированных данных
        all_texts = list(texts) + augmented_texts
        all_labels = list(labels) + augmented_labels
        
        print(f"📈 Аугментация: добавлено {len(augmented_texts)} примеров")
        return all_texts, all_labels
    
    def cross_validate_models(self, X, y, cv_folds=5):
        """Кросс-валидация моделей"""
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import accuracy_score, f1_score
        
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_results = {model_name: [] for model_name in ["lstm", "cnn_lstm", "transformer"]}
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            print(f"\n🔄 Кросс-валидация, фолд {fold + 1}/{cv_folds}")
            
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            # Создание и обучение моделей для этого фолда
            ensemble = SentimentModelEnsemble(vocab_size=self.model_ensemble.vocab_size)
            ensemble.build_all_models()
            
            for model_name, model in ensemble.models.items():
                # Быстрое обучение для кросс-валидации
                model.fit(
                    X_train_fold, y_train_fold,
                    batch_size=MODEL_CONFIG["batch_size"],
                    epochs=10,  # Меньше эпох для CV
                    validation_data=(X_val_fold, y_val_fold),
                    verbose=0
                )
                
                # Оценка
                predictions = model.predict(X_val_fold, verbose=0)
                predicted_classes = np.argmax(predictions, axis=1)
                
                accuracy = accuracy_score(y_val_fold, predicted_classes)
                f1 = f1_score(y_val_fold, predicted_classes, average='weighted')
                
                cv_results[model_name].append({
                    'accuracy': accuracy,
                    'f1_score': f1
                })
        
        # Вывод результатов кросс-валидации
        print("\n📊 Результаты кросс-валидации:")
        for model_name, results in cv_results.items():
            mean_acc = np.mean([r['accuracy'] for r in results])
            std_acc = np.std([r['accuracy'] for r in results])
            mean_f1 = np.mean([r['f1_score'] for r in results])
            
            print(f"{model_name}:")
            print(f"  Точность: {mean_acc:.4f} ± {std_acc:.4f}")
            print(f"  F1-score: {mean_f1:.4f}")
        
        return cv_results
    
    def validate_model_quality(self, model, X_test, y_test, model_name):
        """Валидация качества модели по целевым показателям"""
        predictions = model.predict(X_test, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)
        
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        
        accuracy = accuracy_score(y_test, predicted_classes)
        f1 = f1_score(y_test, predicted_classes, average='weighted')
        precision = precision_score(y_test, predicted_classes, average='weighted', zero_division=0)
        recall = recall_score(y_test, predicted_classes, average='weighted', zero_division=0)
        
        # Проверка соответствия целевым показателям
        quality_checks = {
            'accuracy': accuracy >= VALIDATION_CONFIG["target_accuracy"],
            'f1_score': f1 >= VALIDATION_CONFIG["target_f1_score"],
            'precision': precision >= VALIDATION_CONFIG["min_class_precision"]
        }
        
        all_passed = all(quality_checks.values())
        
        print(f"\n🎯 Валидация качества модели {model_name}:")
        print(f"  Точность: {accuracy:.4f} {'✅' if quality_checks['accuracy'] else '❌'}")
        print(f"  F1-score: {f1:.4f} {'✅' if quality_checks['f1_score'] else '❌'}")
        print(f"  Precision: {precision:.4f} {'✅' if quality_checks['precision'] else '❌'}")
        print(f"  Общий результат: {'✅ ПРОЙДЕНО' if all_passed else '❌ НЕ ПРОЙДЕНО'}")
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'quality_passed': all_passed
        }
    
    def train_production_model(self, X_train, y_train, X_val, y_val, X_test, y_test):
        """Полное обучение продакшн модели"""
        print("🚀 Начинаем обучение продакшн модели...")
        
        # Инициализация ансамбля
        vocab_size = len(self.tokenizer.word_index) + 1
        self.model_ensemble = SentimentModelEnsemble(vocab_size=vocab_size)
        
        # Аугментация данных
        if MODEL_CONFIG["data_augmentation"]["enabled"]:
            X_train_texts = [self.tokenizer.sequences_to_texts([seq])[0] for seq in X_train]
            X_train_aug, y_train_aug = self.augment_data(X_train_texts, y_train)
            
            # Преобразование обратно в последовательности
            X_train_aug_seq = self.tokenizer.texts_to_sequences(X_train_aug)
            from tensorflow.keras.preprocessing.sequence import pad_sequences
            X_train_aug_pad = pad_sequences(
                X_train_aug_seq, 
                maxlen=MODEL_CONFIG["max_sequence_length"]
            )
            
            # Объединение с оригинальными данными
            X_train = np.vstack([X_train, X_train_aug_pad])
            y_train = np.hstack([y_train, y_train_aug])
        
        # Построение всех моделей
        self.model_ensemble.build_all_models()
        
        # Кросс-валидация (опционально)
        if VALIDATION_CONFIG["cross_validation_folds"] > 0:
            print("🔄 Выполняем кросс-валидацию...")
            cv_results = self.cross_validate_models(X_train, y_train)
        
        # Обучение ансамбля
        print("🎯 Обучение основного ансамбля...")
        self.training_history = self.model_ensemble.train_ensemble(
            X_train, y_train, X_val, y_val
        )
        
        # Оценка на тестовых данных
        print("📊 Финальная оценка...")
        test_results = self.model_ensemble.evaluate_ensemble(X_test, y_test)
        
        # Валидация качества
        for model_name, model in self.model_ensemble.models.items():
            validation_result = self.validate_model_quality(
                model, X_test, y_test, model_name
            )
            self.validation_results[model_name] = validation_result
        
        # Валидация ансамбля
        if self.model_ensemble.ensemble_model:
            validation_result = self.validate_model_quality(
                self.model_ensemble.ensemble_model, X_test, y_test, "ensemble"
            )
            self.validation_results["ensemble"] = validation_result
        
        # Сохранение моделей
        self.model_ensemble.save_all_models()
        
        return self.training_history, test_results, self.validation_results
    
    def generate_training_report(self):
        """Генерация подробного отчета об обучении"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = {
            "training_timestamp": timestamp,
            "model_config": MODEL_CONFIG,
            "training_history": str(self.training_history),
            "validation_results": self.validation_results,
            "best_model": max(
                self.validation_results.items(),
                key=lambda x: x[1]["accuracy"]
            )[0] if self.validation_results else None
        }
        
        # Сохранение отчета
        report_path = MODEL_PATHS["main_model"].parent / f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        import json
        with open(report_path, 'w', encoding='utf-8') as f:
            # Конвертация numpy типов для JSON сериализации
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj
            
            json.dump(report, f, ensure_ascii=False, indent=2, default=convert_numpy)
        
        print(f"📄 Отчет сохранен: {report_path}")
        return report


# Пример использования
if __name__ == "__main__":
    # Создание и обучение ансамбля
    trainer = AdvancedSentimentTrainer()
    
    # Здесь должна быть загрузка данных
    # X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    
    # Обучение продакшн модели
    # history, results, validation = trainer.train_production_model(
    #     X_train, y_train, X_val, y_val, X_test, y_test
    # )
    
    # Генерация отчета
    # report = trainer.generate_training_report()
    
    print("✅ Улучшенная архитектура модели готова к использованию!")