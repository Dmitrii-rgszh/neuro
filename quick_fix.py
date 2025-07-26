"""
Быстрое исправление ошибки verbose в production_train.py
"""
import re

print("🔧 Исправляем ошибку в production_train.py...")

# Читаем файл
with open("production_train.py", "r", encoding="utf-8") as f:
    content = f.read()

# Исправляем verbose=True
content = re.sub(
    r'(ReduceLROnPlateau\([^)]*),\s*verbose\s*=\s*True\s*\)',
    r'\1)',
    content
)

# Также исправляем verbose=1 если есть
content = re.sub(
    r'(ReduceLROnPlateau\([^)]*),\s*verbose\s*=\s*1\s*\)',
    r'\1)',
    content
)

# Сохраняем обратно
with open("production_train.py", "w", encoding="utf-8") as f:
    f.write(content)

print("✅ Исправлено! Теперь запустите снова:")
print("   python production_train.py")