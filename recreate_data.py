"""
Пересоздание данных с правильной структурой
"""
import os
import shutil
from pathlib import Path

# Удаление старых данных
data_dir = Path("data")
processed_dir = data_dir / "processed"

print("🗑️ Удаление старых данных...")
if processed_dir.exists():
    for file in processed_dir.glob("*"):
        if file.is_file():
            file.unlink()
            print(f"   Удален: {file}")

print("\n🔄 Запуск создания новых данных...")
print("="*60)

# Запуск data_loader для создания новых данных
os.system("python src/data_loader.py")

print("\n✅ Готово! Теперь можно запустить обучение:")
print("python production_train.py")