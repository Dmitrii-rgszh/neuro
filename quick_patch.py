"""
Быстрое исправление проблемы с размерами в production_train.py
"""
import re
import shutil

print("🔧 Исправляем проблему размеров в модели...")

# Создаем резервную копию
shutil.copy("production_train.py", "production_train_backup.py")
print("📁 Создана резервная копия: production_train_backup.py")

# Читаем файл
with open("production_train.py", "r", encoding="utf-8") as f:
    content = f.read()

# Исправление 1: Размер fc1 слоя
# Меняем config["hidden_dim"] * 2 + 64 на правильный размер
content = re.sub(
    r'self\.fc1 = nn\.Linear\(config\["hidden_dim"\] \* 2 \+ 64, 512\)',
    'self.fc1 = nn.Linear(config["hidden_dim"] * 2 * 2 + 64, 512)  # Исправлено: 512*2 + 64 = 1088',
    content
)

# Исправление 2: Количество слоев LSTM для корректной работы dropout
# Меняем dropout=0.3 if config["epochs"] > 1 else 0
content = re.sub(
    r'dropout=0\.3 if config\["epochs"\] > 1 else 0\n\s*\)',
    'num_layers=2,\n            dropout=0.3\n        )',
    content
)

# Исправление 3: Добавляем num_layers для второго LSTM
content = re.sub(
    r'(self\.lstm2 = nn\.LSTM\([^)]+)',
    r'\1,\n            num_layers=2',
    content
)

# Сохраняем исправленный файл
with open("production_train.py", "w", encoding="utf-8") as f:
    f.write(content)

print("✅ Все исправления применены!")
print("\n📋 Что было исправлено:")
print("   1. Размер входа fc1: 576 → 1088 (правильный)")
print("   2. Добавлены num_layers=2 для LSTM (для корректной работы dropout)")
print("\n🚀 Теперь запустите: python production_train.py")