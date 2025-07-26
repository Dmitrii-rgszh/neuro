"""
Быстрое исправление data_loader.py
"""
import os

def fix_data_loader():
    """Исправляет проблему с progress_apply в data_loader.py"""
    
    data_loader_path = "src/data_loader.py"
    
    if not os.path.exists(data_loader_path):
        print("❌ Файл src/data_loader.py не найден!")
        return False
    
    # Читаем файл
    with open(data_loader_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Заменяем проблемную строку
    old_line = "df['processed_text'] = df['text'].progress_apply(self.preprocess_text)"
    new_line = "df['processed_text'] = df['text'].apply(self.preprocess_text)"
    
    if old_line in content:
        content = content.replace(old_line, new_line)
        
        # Сохраняем обратно
        with open(data_loader_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("✅ Файл исправлен!")
        return True
    else:
        print("⚠️ Проблемная строка не найдена. Возможно, уже исправлено.")
        return False

if __name__ == "__main__":
    print("Исправление data_loader.py...")
    fix_data_loader()
    print("\nТеперь запустите: python src/train.py")