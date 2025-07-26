"""
Тестирование установки и зависимостей
"""

def test_imports():
    """Проверка импорта всех необходимых библиотек"""
    print("Проверка установленных библиотек...\n")
    
    libraries = [
        ("tensorflow", "TensorFlow"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("sklearn", "Scikit-learn"),
        ("telegram", "Python-telegram-bot"),
        ("dotenv", "Python-dotenv"),
        ("nltk", "NLTK"),
        ("matplotlib", "Matplotlib"),
        ("seaborn", "Seaborn"),
        ("tqdm", "tqdm"),
        ("requests", "Requests"),
        ("joblib", "Joblib")
    ]
    
    all_ok = True
    
    for module_name, display_name in libraries:
        try:
            __import__(module_name)
            print(f"✓ {display_name} установлен")
        except ImportError:
            print(f"✗ {display_name} НЕ установлен")
            all_ok = False
    
    return all_ok

def test_nltk_data():
    """Проверка NLTK данных"""
    print("\nПроверка NLTK данных...\n")
    
    try:
        import nltk
        
        resources = ['stopwords', 'punkt']
        all_ok = True
        
        for resource in resources:
            try:
                nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' else f'corpora/{resource}')
                print(f"✓ NLTK {resource} загружен")
            except LookupError:
                print(f"✗ NLTK {resource} НЕ загружен")
                print(f"  Выполните: nltk.download('{resource}')")
                all_ok = False
        
        return all_ok
    except ImportError:
        print("✗ NLTK не установлен")
        return False

def test_project_structure():
    """Проверка структуры проекта"""
    print("\nПроверка структуры проекта...\n")
    
    from pathlib import Path
    
    required_dirs = [
        "data",
        "data/raw",
        "data/processed",
        "models",
        "src"
    ]
    
    all_ok = True
    base_dir = Path.cwd()
    
    for dir_name in required_dirs:
        dir_path = base_dir / dir_name
        if dir_path.exists():
            print(f"✓ Директория {dir_name} существует")
        else:
            print(f"✗ Директория {dir_name} НЕ существует")
            all_ok = False
    
    # Проверка наличия .env файла
    if (base_dir / ".env").exists():
        print("✓ Файл .env существует")
    else:
        print("✗ Файл .env НЕ существует")
        print("  Создайте его на основе .env.example")
        all_ok = False
    
    return all_ok

def test_tensorflow():
    """Проверка работы TensorFlow"""
    print("\nПроверка TensorFlow...\n")
    
    try:
        import tensorflow as tf
        print(f"✓ TensorFlow версия: {tf.__version__}")
        
        # Проверка GPU
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✓ Обнаружено GPU: {len(gpus)} устройств")
            for gpu in gpus:
                print(f"  - {gpu.name}")
        else:
            print("ℹ GPU не обнаружено, будет использоваться CPU")
        
        return True
    except Exception as e:
        print(f"✗ Ошибка TensorFlow: {e}")
        return False

def main():
    """Основная функция тестирования"""
    print("=" * 50)
    print("ТЕСТИРОВАНИЕ УСТАНОВКИ TELEGRAM SENTIMENT BOT")
    print("=" * 50)
    
    tests = [
        ("Импорт библиотек", test_imports),
        ("NLTK данные", test_nltk_data),
        ("Структура проекта", test_project_structure),
        ("TensorFlow", test_tensorflow)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{test_name}")
        print("-" * len(test_name))
        result = test_func()
        results.append(result)
    
    print("\n" + "=" * 50)
    print("РЕЗУЛЬТАТЫ:")
    print("=" * 50)
    
    all_passed = all(results)
    
    if all_passed:
        print("\n✅ ВСЕ ТЕСТЫ ПРОЙДЕНЫ!")
        print("\nСледующие шаги:")
        print("1. Добавьте токен бота в .env файл")
        print("2. Запустите обучение: python src/train.py")
        print("3. Запустите бота: python src/bot.py")
    else:
        print("\n⚠️  НЕКОТОРЫЕ ТЕСТЫ НЕ ПРОЙДЕНЫ!")
        print("\nИсправьте ошибки и запустите тест снова.")
        print("Для установки всех зависимостей выполните:")
        print("  pip install -r requirements.txt")
        print("  python setup_project.py")

if __name__ == "__main__":
    main()