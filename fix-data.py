"""
Исправление структуры данных для корректной работы
"""
import sys
import os
sys.path.append('src')

import pandas as pd
from pathlib import Path
import shutil

# Пути
data_dir = Path("data")
processed_dir = data_dir / "processed"
raw_dir = data_dir / "raw"

# Создание директорий
processed_dir.mkdir(parents=True, exist_ok=True)
raw_dir.mkdir(parents=True, exist_ok=True)

print("🔧 Исправление структуры данных...")
print("="*60)

# Удаление старых файлов
print("\n1. Удаление старых данных...")
for file_path in processed_dir.glob("*.csv"):
    file_path.unlink()
    print(f"   Удален: {file_path.name}")

for file_path in processed_dir.glob("*.pkl"):
    file_path.unlink()
    print(f"   Удален: {file_path.name}")

# Импорт и запуск data_loader
print("\n2. Создание новых данных с правильной структурой...")
try:
    from data_loader import EnhancedDataLoader
    
    # Создание загрузчика
    loader = EnhancedDataLoader()
    
    # Подготовка данных
    print("\n3. Подготовка финального датасета...")
    train_df, test_df = loader.prepare_final_dataset()
    
    # Сохранение данных
    print("\n4. Сохранение обработанных данных...")
    loader.save_processed_data(train_df, test_df)
    
    # Проверка результата
    print("\n5. Проверка созданных файлов:")
    train_path = processed_dir / "train_data.csv"
    test_path = processed_dir / "test_data.csv"
    
    if train_path.exists():
        train_check = pd.read_csv(train_path)
        print(f"   ✅ train_data.csv: {len(train_check)} строк")
        print(f"      Колонки: {train_check.columns.tolist()}")
        if 'label' in train_check.columns:
            print(f"      Распределение классов:")
            print(f"      {train_check['label'].value_counts()}")
    
    if test_path.exists():
        test_check = pd.read_csv(test_path)
        print(f"\n   ✅ test_data.csv: {len(test_check)} строк")
        print(f"      Колонки: {test_check.columns.tolist()}")
    
    label_encoder_path = processed_dir / "label_encoder.pkl"
    if label_encoder_path.exists():
        print(f"\n   ✅ label_encoder.pkl создан")
    
    print("\n✅ Данные успешно исправлены!")
    print("\n🚀 Теперь можно запустить обучение:")
    print("   python production_train.py")
    
except Exception as e:
    print(f"\n❌ Ошибка: {e}")
    import traceback
    traceback.print_exc()
