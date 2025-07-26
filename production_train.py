"""
Продакшн скрипт для обучения высокоточной модели анализа настроений
"""
import os
import sys
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

# Добавляем путь к модулям
sys.path.append('src')

from config import (
    MODEL_CONFIG, MODEL_PATHS, TOKENIZER_CONFIG, VALIDATION_CONFIG,
    PROCESSED_DATA_DIR, DATA_SOURCES
)
from data_loader import EnhancedDataLoader
from model import SentimentModelEnsemble, AdvancedSentimentTrainer


class ProductionTrainer:
    """Продакшн тренер для высокоточного анализа настроений"""
    
    def __init__(self):
        self.data_loader = EnhancedDataLoader()
        self.advanced_trainer = AdvancedSentimentTrainer()
        self.tokenizer = None
        self.label_encoder = None
        self.final_results = {}
        
    def setup_environment(self):
        """Настройка окружения для обучения"""
        print("🔧 Настройка окружения...")
        
        # Создание необходимых директорий
        for path in MODEL_PATHS.values():
            if isinstance(path, Path):
                path.parent.mkdir(parents=True, exist_ok=True)
        
        # Настройка TensorFlow
        import tensorflow as tf
        
        # Настройка GPU (если доступно)
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"✅ Найдено {len(gpus)} GPU устройств")
            except RuntimeError as e:
                print(f"⚠️ Ошибка настройки GPU: {e}")
        else:
            print("ℹ️ Используется CPU")
        
        # Настройка воспроизводимости
        tf.random.set_seed(42)
        np.random.seed(42)
        
        print("✅ Окружение настроено")
    
    def load_and_prepare_data(self):
        """Загрузка и подготовка данных"""
        print("\n" + "="*60)
        print("📊 ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ")
        print("="*60)
        
        # Проверка существования обработанных данных
        train_path = PROCESSED_DATA_DIR / "train_data.csv"
        test_path = PROCESSED_DATA_DIR / "test_data.csv"
        
        if train_path.exists() and test_path.exists():
            print("📂 Загрузка существующих обработанных данных...")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            # Загрузка label encoder
            label_encoder_path = PROCESSED_DATA_DIR / "label_encoder.pkl"
            if label_encoder_path.exists():
                self.label_encoder = joblib.load(label_encoder_path)
            else:
                raise FileNotFoundError("Label encoder не найден")
        else:
            print("🔄 Подготовка новых данных...")
            # Подготовка данных с использованием расширенного загрузчика
            train_df, test_df = self.data_loader.prepare_final_dataset()
            self.label_encoder = self.data_loader.label_encoder
            
            # Сохранение
            self.data_loader.save_processed_data(train_df, test_df)
        
        print(f"\n📈 Статистика данных:")
        print(f"   Обучающая выборка: {len(train_df):,} примеров")
        print(f"   Тестовая выборка: {len(test_df):,} примеров")
        print(f"   Всего: {len(train_df) + len(test_df):,} примеров")
        
        # Анализ распределения классов
        print(f"\n📊 Распределение классов в обучающей выборке:")
        label_counts = train_df['label'].value_counts()
        for label, count in label_counts.items():
            percentage = count / len(train_df) * 100
            print(f"   {label}: {count:,} ({percentage:.1f}%)")
        
        return train_df, test_df
    
    def prepare_tokenizer_and_sequences(self, train_df, test_df):
        """Подготовка токенизатора и последовательностей"""
        print("\n" + "="*60)
        print("🔤 ТОКЕНИЗАЦИЯ И ПОДГОТОВКА ПОСЛЕДОВАТЕЛЬНОСТЕЙ")
        print("="*60)
        
        # Создание токенизатора
        self.tokenizer = Tokenizer(**TOKENIZER_CONFIG)
        
        # Обучение токенизатора
        print("🔄 Обучение токенизатора...")
        all_texts = list(train_df['processed_text']) + list(test_df['processed_text'])
        self.tokenizer.fit_on_texts(all_texts)
        
        vocab_size = len(self.tokenizer.word_index) + 1
        print(f"   Размер словаря: {vocab_size:,} слов")
        print(f"   Эффективный размер: {min(vocab_size, MODEL_CONFIG['max_features']):,}")
        
        # Преобразование в последовательности
        print("🔄 Преобразование текстов в последовательности...")
        X_train_seq = self.tokenizer.texts_to_sequences(train_df['processed_text'])
        X_test_seq = self.tokenizer.texts_to_sequences(test_df['processed_text'])
        
        # Паддинг последовательностей
        X_train = pad_sequences(
            X_train_seq,
            maxlen=MODEL_CONFIG["max_sequence_length"],
            padding='post',
            truncating='post'
        )
        X_test = pad_sequences(
            X_test_seq,
            maxlen=MODEL_CONFIG["max_sequence_length"],
            padding='post',
            truncating='post'
        )
        
        # Подготовка меток
        y_train = train_df['label_encoded'].values
        y_test = test_df['label_encoded'].values
        
        # Разделение обучающей выборки на train/validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train,
            test_size=VALIDATION_CONFIG["validation_size"],
            random_state=VALIDATION_CONFIG["random_state"],
            stratify=y_train
        )
        
        print(f"\n📊 Финальные размеры данных:")
        print(f"   Обучение: {X_train.shape}")
        print(f"   Валидация: {X_val.shape}")
        print(f"   Тест: {X_test.shape}")
        
        # Анализ длин последовательностей
        train_lengths = [len([w for w in seq if w != 0]) for seq in X_train]
        print(f"\n📏 Статистика длин последовательностей:")
        print(f"   Средняя длина: {np.mean(train_lengths):.1f}")
        print(f"   Медианная длина: {np.median(train_lengths):.1f}")
        print(f"   95-й перцентиль: {np.percentile(train_lengths, 95):.1f}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_production_models(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """Обучение продакшн моделей"""
        print("\n" + "="*60)
        print("🚀 ОБУЧЕНИЕ ПРОДАКШН МОДЕЛЕЙ")
        print("="*60)
        
        # Создание тренера
        from model import SentimentModelEnsemble
        vocab_size = len(self.tokenizer.word_index) + 1
        
        # Создание ансамбля моделей
        ensemble = SentimentModelEnsemble(vocab_size=vocab_size)
        ensemble.build_all_models()
        
        # Обучение всех моделей
        print("\n🎯 Обучение ансамбля моделей...")
        histories = ensemble.train_ensemble(X_train, y_train, X_val, y_val)
        
        # Оценка на тестовых данных
        print("\n📊 Оценка на тестовых данных...")
        test_results = ensemble.evaluate_ensemble(X_test, y_test)
        
        # Валидация качества
        validation_results = {}
        for model_name, model in ensemble.models.items():
            predictions = model.predict(X_test, verbose=0)
            predicted_classes = np.argmax(predictions, axis=1)
            
            from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
            
            accuracy = accuracy_score(y_test, predicted_classes)
            f1 = f1_score(y_test, predicted_classes, average='weighted')
            precision = precision_score(y_test, predicted_classes, average='weighted')
            recall = recall_score(y_test, predicted_classes, average='weighted')
            
            validation_results[model_name] = {
                'accuracy': accuracy,
                'f1_score': f1,
                'precision': precision,
                'recall': recall,
                'quality_passed': accuracy >= VALIDATION_CONFIG["target_accuracy"]
            }
        
        # Сохранение моделей
        ensemble.save_all_models()
        
        self.final_results = {
            'training_history': histories,
            'test_results': test_results,
            'validation_results': validation_results,
            'training_duration': str(datetime.now() - datetime.now()),
            'best_model': self._find_best_model(validation_results)
        }
        
        return self.final_results
    
    def _find_best_model(self, validation_results):
        """Определение лучшей модели"""
        if not validation_results:
            return None
        
        best_model = max(
            validation_results.items(),
            key=lambda x: x[1]['accuracy'] if isinstance(x[1], dict) else 0
        )
        
        return {
            'name': best_model[0],
            'accuracy': best_model[1]['accuracy'],
            'f1_score': best_model[1]['f1_score']
        }
    
    def save_production_artifacts(self):
        """Сохранение всех артефактов для продакшна"""
        print("\n" + "="*60)
        print("💾 СОХРАНЕНИЕ ПРОДАКШН АРТЕФАКТОВ")
        print("="*60)
        
        # Сохранение токенизатора
        tokenizer_path = MODEL_PATHS["tokenizer"]
        joblib.dump(self.tokenizer, tokenizer_path)
        print(f"✅ Токенизатор сохранен: {tokenizer_path}")
        
        # Сохранение label encoder
        label_encoder_path = MODEL_PATHS["label_encoder"]
        joblib.dump(self.label_encoder, label_encoder_path)
        print(f"✅ Label encoder сохранен: {label_encoder_path}")
        
        # Сохранение конфигурации
        config_path = MODEL_PATHS["main_model"].parent / "model_config.json"
        config_data = {
            'model_config': MODEL_CONFIG,
            'tokenizer_config': TOKENIZER_CONFIG,
            'vocab_size': len(self.tokenizer.word_index) + 1,
            'class_names': MODEL_CONFIG["class_names"],
            'training_timestamp': datetime.now().isoformat(),
            'final_results': self.final_results
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            # Конвертация numpy типов
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj
            
            json.dump(config_data, f, ensure_ascii=False, indent=2, default=convert_numpy)
        print(f"✅ Конфигурация сохранена: {config_path}")
        
        print("\n📄 Артефакты готовы для продакшна!")
    
    def create_visualizations(self):
        """Создание визуализаций результатов"""
        print("\n" + "="*60)
        print("📊 СОЗДАНИЕ ВИЗУАЛИЗАЦИЙ")
        print("="*60)
        
        # Создание директории для графиков
        plots_dir = MODEL_PATHS["main_model"].parent / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # График истории обучения
        if self.final_results.get('training_history'):
            self._plot_training_history(plots_dir)
        
        # График сравнения моделей
        if self.final_results.get('validation_results'):
            self._plot_model_comparison(plots_dir)
        
        # Матрица ошибок для лучшей модели
        self._plot_confusion_matrix(plots_dir)
        
        print(f"✅ Визуализации сохранены в: {plots_dir}")
    
    def _plot_training_history(self, plots_dir):
        """График истории обучения"""
        history = self.final_results['training_history']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('История обучения моделей', fontsize=16)
        
        models = ['lstm', 'cnn_lstm', 'transformer']
        colors = ['blue', 'green', 'red']
        
        for i, (model_name, color) in enumerate(zip(models, colors)):
            if model_name in history:
                hist = history[model_name].history
                
                # Точность
                axes[0, 0].plot(hist['accuracy'], label=f'{model_name} train', color=color)
                axes[0, 0].plot(hist['val_accuracy'], label=f'{model_name} val', color=color, linestyle='--')
                
                # Потери
                axes[0, 1].plot(hist['loss'], label=f'{model_name} train', color=color)
                axes[0, 1].plot(hist['val_loss'], label=f'{model_name} val', color=color, linestyle='--')
        
        axes[0, 0].set_title('Точность')
        axes[0, 0].set_xlabel('Эпоха')
        axes[0, 0].set_ylabel('Точность')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        axes[0, 1].set_title('Потери')
        axes[0, 1].set_xlabel('Эпоха')
        axes[0, 1].set_ylabel('Потери')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_model_comparison(self, plots_dir):
        """График сравнения моделей"""
        validation_results = self.final_results['validation_results']
        
        models = list(validation_results.keys())
        metrics = ['accuracy', 'f1_score', 'precision', 'recall']
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        x = np.arange(len(models))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            values = [validation_results[model][metric] for model in models]
            ax.bar(x + i * width, values, width, label=metric)
        
        ax.set_xlabel('Модели')
        ax.set_ylabel('Значение метрики')
        ax.set_title('Сравнение метрик моделей')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(models)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_confusion_matrix(self, plots_dir):
        """Матрица ошибок для лучшей модели"""
        # Здесь должна быть логика для создания матрицы ошибок
        # На данный момент создаем заглушку
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, 'Confusion Matrix\n(Implementation needed)', 
                ha='center', va='center', fontsize=16)
        plt.title('Матрица ошибок лучшей модели')
        plt.savefig(plots_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_production_training(self):
        """Полный цикл продакшн обучения"""
        print("🎯 ЗАПУСК ПРОДАКШН ОБУЧЕНИЯ МОДЕЛИ АНАЛИЗА НАСТРОЕНИЙ")
        print("="*80)
        
        start_time = datetime.now()
        
        try:
            # 1. Настройка окружения
            self.setup_environment()
            
            # 2. Загрузка и подготовка данных
            train_df, test_df = self.load_and_prepare_data()
            
            # 3. Токенизация
            X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_tokenizer_and_sequences(
                train_df, test_df
            )
            
            # 4. Обучение моделей
            results = self.train_production_models(
                X_train, X_val, X_test, y_train, y_val, y_test
            )
            
            # 5. Сохранение артефактов
            self.save_production_artifacts()
            
            # 6. Создание визуализаций
            self.create_visualizations()
            
            # 7. Финальный отчет
            self.print_final_report()
            
            end_time = datetime.now()
            total_duration = end_time - start_time
            
            print(f"\n🎉 ОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО!")
            print(f"⏱️ Общее время: {total_duration}")
            print(f"🏆 Лучшая модель: {self.final_results['best_model']['name']}")
            print(f"📊 Лучшая точность: {self.final_results['best_model']['accuracy']:.4f}")
            
            return True
            
        except Exception as e:
            print(f"\n❌ ОШИБКА ПРИ ОБУЧЕНИИ: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def print_final_report(self):
        """Печать финального отчета"""
        print("\n" + "="*60)
        print("📋 ФИНАЛЬНЫЙ ОТЧЕТ")
        print("="*60)
        
        if self.final_results.get('validation_results'):
            print("\n🏆 Результаты валидации моделей:")
            for model_name, metrics in self.final_results['validation_results'].items():
                if isinstance(metrics, dict):
                    quality_status = "✅ ПРОЙДЕНО" if metrics.get('quality_passed', False) else "❌ НЕ ПРОЙДЕНО"
                    print(f"\n{model_name.upper()}:")
                    print(f"  📊 Точность: {metrics['accuracy']:.4f}")
                    print(f"  📊 F1-score: {metrics['f1_score']:.4f}")
                    print(f"  📊 Precision: {metrics['precision']:.4f}")
                    print(f"  📊 Recall: {metrics['recall']:.4f}")
                    print(f"  🎯 Качество: {quality_status}")
        
        if self.final_results.get('best_model'):
            best = self.final_results['best_model']
            print(f"\n🥇 ЛУЧШАЯ МОДЕЛЬ: {best['name'].upper()}")
            print(f"   📊 Точность: {best['accuracy']:.4f}")
            print(f"   📊 F1-score: {best['f1_score']:.4f}")
        
        # Рекомендации
        print(f"\n💡 РЕКОМЕНДАЦИИ:")
        if self.final_results.get('best_model'):
            best_acc = self.final_results['best_model']['accuracy']
            if best_acc >= 0.92:
                print("   🎯 Отличное качество! Модель готова для продакшна.")
            elif best_acc >= 0.88:
                print("   ✅ Хорошее качество. Можно использовать в продакшне.")
            elif best_acc >= 0.85:
                print("   ⚠️ Приемлемое качество. Рекомендуется дополнительная настройка.")
            else:
                print("   ❌ Низкое качество. Требуется улучшение данных или архитектуры.")
        
        print(f"\n📁 Файлы сохранены в: {MODEL_PATHS['main_model'].parent}")
        print(f"🤖 Для запуска бота: python src/bot.py")


def main():
    """Главная функция"""
    trainer = ProductionTrainer()
    success = trainer.run_production_training()
    
    if success:
        print("\n🚀 Готово! Теперь можете запустить бота.")
    else:
        print("\n❌ Обучение не завершено. Проверьте ошибки выше.")
    
    return success


if __name__ == "__main__":
    main()