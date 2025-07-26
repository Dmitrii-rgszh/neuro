"""
Скрипт для очистки некорректных данных
"""
import os
from pathlib import Path

def clean_data_directory():
    """Очистка директории с данными"""
    data_dir = Path("data/raw")
    
    if data_dir.exists():
        # Удаление файла rusentiment.csv если он некорректный
        rusentiment_file = data_dir / "rusentiment.csv"
        if rusentiment_file.exists():
            try:
                # Проверяем первую строку
                with open(rusentiment_file, 'r', encoding='utf-8') as f:
                    first_line = f.readline()
                    if 'HTML' in first_line or 'DOCTYPE' in first_line:
                        os.remove(rusentiment_file)
                        print("✓ Удален некорректный файл rusentiment.csv")
                    else:
                        print("✓ Файл rusentiment.csv корректный")
            except Exception as e:
                print(f"Ошибка при проверке файла: {e}")
                os.remove(rusentiment_file)
                print("✓ Удален проблемный файл rusentiment.csv")
    
    print("\n✅ Очистка завершена!")
    print("Теперь можете запустить обучение: python src/train.py")

if __name__ == "__main__":
    clean_data_directory()