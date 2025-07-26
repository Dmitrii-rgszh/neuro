"""
ПОЛНАЯ ДИАГНОСТИКА И АВТОМАТИЧЕСКОЕ РЕШЕНИЕ ПРОБЛЕМ С GPU
Для RTX 3060 Ti и машинного обучения
"""
import os
import sys
import subprocess
import platform
import ctypes
import json
from pathlib import Path

class GPUDiagnostics:
    def __init__(self):
        self.problems = []
        self.solutions = []
        self.gpu_ready = False
        
    def run_full_diagnostics(self):
        """Запуск полной диагностики"""
        print("="*70)
        print("🔍 ПОЛНАЯ ДИАГНОСТИКА GPU ДЛЯ МАШИННОГО ОБУЧЕНИЯ")
        print("="*70)
        
        # 1. Проверка системы
        self.check_system()
        
        # 2. Проверка драйверов
        self.check_nvidia_drivers()
        
        # 3. Проверка CUDA
        self.check_cuda()
        
        # 4. Проверка Python пакетов
        self.check_python_packages()
        
        # 5. Проверка переменных окружения
        self.check_environment()
        
        # 6. Тесты GPU
        self.test_gpu_frameworks()
        
        # 7. Вывод результатов
        self.print_results()
        
    def check_system(self):
        """Проверка системы"""
        print("\n1️⃣ ПРОВЕРКА СИСТЕМЫ")
        print("-"*50)
        
        # Windows версия
        if platform.system() != "Windows":
            self.problems.append("ОС не Windows")
            self.solutions.append("Эта инструкция только для Windows")
            
        # Проверка прав администратора
        is_admin = ctypes.windll.shell32.IsUserAnAdmin()
        print(f"🖥️  ОС: {platform.system()} {platform.release()}")
        print(f"👤 Права администратора: {'Да' if is_admin else 'Нет'}")
        
        if not is_admin:
            self.problems.append("Нет прав администратора")
            self.solutions.append("Запустите скрипт от имени администратора")
            
        # Python версия
        py_ver = sys.version_info
        print(f"🐍 Python: {py_ver.major}.{py_ver.minor}.{py_ver.micro}")
        
        if py_ver.major == 3 and py_ver.minor == 12:
            self.problems.append("Python 3.12 может иметь проблемы с CUDA")
            self.solutions.append("Рекомендуется Python 3.9-3.11")
            
    def check_nvidia_drivers(self):
        """Проверка драйверов NVIDIA"""
        print("\n2️⃣ ПРОВЕРКА ДРАЙВЕРОВ NVIDIA")
        print("-"*50)
        
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ nvidia-smi найден")
                
                # Парсим вывод для получения версии драйвера
                for line in result.stdout.split('\n'):
                    if 'Driver Version' in line:
                        driver_version = line.split('Driver Version:')[1].split()[0]
                        print(f"📊 Версия драйвера: {driver_version}")
                        
                        # Проверка минимальной версии для RTX 3060 Ti
                        if float(driver_version.split('.')[0]) < 516:
                            self.problems.append(f"Старая версия драйвера: {driver_version}")
                            self.solutions.append("Обновите драйвер NVIDIA до версии 516.01 или выше")
                            
                # Информация о GPU
                gpu_info = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], 
                                        capture_output=True, text=True)
                if gpu_info.returncode == 0:
                    print(f"🎮 GPU: {gpu_info.stdout.strip()}")
            else:
                self.problems.append("nvidia-smi не работает")
                self.solutions.append("Установите или переустановите драйверы NVIDIA")
                
        except FileNotFoundError:
            self.problems.append("nvidia-smi не найден")
            self.solutions.append("Установите драйверы NVIDIA с https://www.nvidia.com/Download/index.aspx")
            
    def check_cuda(self):
        """Проверка CUDA"""
        print("\n3️⃣ ПРОВЕРКА CUDA")
        print("-"*50)
        
        # Проверка CUDA_PATH
        cuda_path = os.environ.get('CUDA_PATH')
        if cuda_path:
            print(f"✅ CUDA_PATH: {cuda_path}")
            
            # Проверка версии
            version_file = Path(cuda_path) / "version.txt"
            if version_file.exists():
                with open(version_file, 'r') as f:
                    cuda_version = f.read().strip()
                    print(f"📊 CUDA версия: {cuda_version}")
            
            # Проверка nvcc
            nvcc_path = Path(cuda_path) / "bin" / "nvcc.exe"
            if nvcc_path.exists():
                nvcc_result = subprocess.run([str(nvcc_path), '--version'], capture_output=True, text=True)
                if nvcc_result.returncode == 0:
                    print("✅ nvcc найден и работает")
                else:
                    self.problems.append("nvcc не работает")
            else:
                self.problems.append("nvcc.exe не найден")
                self.solutions.append("Переустановите CUDA Toolkit")
        else:
            self.problems.append("CUDA_PATH не установлен")
            self.solutions.append("Установите CUDA Toolkit 11.7 или 11.8")
            
        # Проверка cudNN
        cudnn_paths = [
            Path(cuda_path) / "bin" / "cudnn64_8.dll" if cuda_path else None,
            Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.7/bin/cudnn64_8.dll"),
            Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/bin/cudnn64_8.dll"),
        ]
        
        cudnn_found = False
        for path in cudnn_paths:
            if path and path.exists():
                print(f"✅ cuDNN найден: {path}")
                cudnn_found = True
                break
                
        if not cudnn_found:
            self.problems.append("cuDNN не найден")
            self.solutions.append("Скачайте cuDNN с https://developer.nvidia.com/cudnn")
            
    def check_python_packages(self):
        """Проверка Python пакетов"""
        print("\n4️⃣ ПРОВЕРКА PYTHON ПАКЕТОВ")
        print("-"*50)
        
        # TensorFlow
        try:
            import tensorflow as tf
            print(f"✅ TensorFlow: {tf.__version__}")
            print(f"   CUDA построен: {tf.test.is_built_with_cuda()}")
            print(f"   GPU доступны: {len(tf.config.list_physical_devices('GPU'))}")
            
            if not tf.test.is_built_with_cuda():
                self.problems.append("TensorFlow собран без CUDA")
                self.solutions.append("Установите tensorflow-gpu или tensorflow>=2.10.0")
                
        except ImportError:
            self.problems.append("TensorFlow не установлен")
            self.solutions.append("pip install tensorflow==2.10.0")
            
        # PyTorch
        try:
            import torch
            print(f"✅ PyTorch: {torch.__version__}")
            print(f"   CUDA доступна: {torch.cuda.is_available()}")
            print(f"   CUDA версия: {torch.version.cuda}")
            
            if not torch.cuda.is_available():
                self.problems.append("PyTorch не видит CUDA")
                self.solutions.append("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
                
        except ImportError:
            print("⚠️  PyTorch не установлен (опционально)")
            
    def check_environment(self):
        """Проверка переменных окружения"""
        print("\n5️⃣ ПРОВЕРКА ПЕРЕМЕННЫХ ОКРУЖЕНИЯ")
        print("-"*50)
        
        important_vars = {
            'CUDA_PATH': 'Путь к CUDA Toolkit',
            'CUDNN_PATH': 'Путь к cuDNN (опционально)',
            'PATH': 'Должен содержать CUDA/bin'
        }
        
        for var, desc in important_vars.items():
            value = os.environ.get(var, '')
            if var == 'PATH':
                # Проверяем наличие CUDA в PATH
                cuda_in_path = any('cuda' in p.lower() for p in value.split(';'))
                print(f"{'✅' if cuda_in_path else '⚠️ '} {var}: {'CUDA найден' if cuda_in_path else 'CUDA не найден'} в PATH")
                
                if not cuda_in_path:
                    self.problems.append("CUDA/bin не в PATH")
                    self.solutions.append("Добавьте C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.X\\bin в PATH")
            else:
                print(f"{'✅' if value else '❌'} {var}: {value if value else 'Не установлен'}")
                
    def test_gpu_frameworks(self):
        """Тестирование фреймворков"""
        print("\n6️⃣ ТЕСТИРОВАНИЕ GPU")
        print("-"*50)
        
        # TensorFlow тест
        print("\n🔷 TensorFlow GPU тест:")
        tf_test = """
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ TensorFlow видит {len(gpus)} GPU")
    for gpu in gpus:
        print(f"   {gpu}")
    # Простой тест
    with tf.device('/GPU:0'):
        a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
        c = tf.matmul(a, b)
    print("✅ Вычисления на GPU работают")
else:
    print("❌ TensorFlow не видит GPU")
"""
        try:
            exec(tf_test)
            self.gpu_ready = True
        except Exception as e:
            print(f"❌ Ошибка TensorFlow: {str(e)}")
            
        # PyTorch тест
        print("\n🔶 PyTorch GPU тест:")
        pytorch_test = """
import torch

if torch.cuda.is_available():
    print(f"✅ PyTorch видит GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Память: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    # Простой тест
    a = torch.randn(100, 100).cuda()
    b = torch.randn(100, 100).cuda()
    c = torch.matmul(a, b)
    print("✅ Вычисления на GPU работают")
else:
    print("❌ PyTorch не видит GPU")
"""
        try:
            exec(pytorch_test)
            self.gpu_ready = True
        except Exception as e:
            print(f"❌ Ошибка PyTorch: {str(e)}")
            
    def print_results(self):
        """Вывод результатов диагностики"""
        print("\n" + "="*70)
        print("📋 РЕЗУЛЬТАТЫ ДИАГНОСТИКИ")
        print("="*70)
        
        if not self.problems:
            print("\n✅ ВСЕ ОТЛИЧНО! GPU ГОТОВ К РАБОТЕ!")
            print("\n🚀 Можете запускать обучение модели!")
        else:
            print(f"\n❌ Обнаружено проблем: {len(self.problems)}")
            print("\n🔧 ПРОБЛЕМЫ И РЕШЕНИЯ:")
            print("-"*50)
            
            for i, (problem, solution) in enumerate(zip(self.problems, self.solutions), 1):
                print(f"\n{i}. ❌ Проблема: {problem}")
                print(f"   💡 Решение: {solution}")
                
            # Автоматические исправления
            print("\n" + "="*70)
            print("🔧 АВТОМАТИЧЕСКИЕ ИСПРАВЛЕНИЯ")
            print("="*70)
            
            self.auto_fix()
            
    def auto_fix(self):
        """Попытка автоматического исправления проблем"""
        print("\n🤖 Пытаемся исправить проблемы автоматически...")
        
        # Создание скрипта установки
        fix_script = """@echo off
echo ====================================
echo АВТОМАТИЧЕСКОЕ ИСПРАВЛЕНИЕ GPU
echo ====================================
echo.

"""
        
        if "TensorFlow не установлен" in str(self.problems):
            fix_script += """
echo Устанавливаем TensorFlow GPU...
python -m pip install tensorflow==2.10.0 --upgrade --force-reinstall
echo.
"""
        
        if "PyTorch не видит CUDA" in str(self.problems):
            fix_script += """
echo Устанавливаем PyTorch с CUDA...
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
echo.
"""
        
        if "CUDA Toolkit" in str(self.solutions):
            fix_script += """
echo.
echo CUDA Toolkit нужно установить вручную:
echo 1. Скачайте CUDA 11.8: https://developer.nvidia.com/cuda-11-8-0-download-archive
echo 2. Запустите установщик
echo 3. Выберите Custom Installation
echo 4. Установите только CUDA Toolkit
echo.
pause
"""
        
        fix_script += """
echo.
echo Готово! Перезапустите диагностику.
pause
"""
        
        with open("fix_gpu_problems.bat", "w") as f:
            f.write(fix_script)
            
        print("✅ Создан fix_gpu_problems.bat")
        print("🔄 Запустите его от имени администратора!")

# Запуск диагностики
if __name__ == "__main__":
    diagnostics = GPUDiagnostics()
    diagnostics.run_full_diagnostics()
    
    print("\n" + "="*70)
    print("💡 ДОПОЛНИТЕЛЬНЫЕ СОВЕТЫ:")
    print("="*70)
    print("\n1. Если ничего не помогает - перезагрузите компьютер")
    print("2. Убедитесь что RTX 3060 Ti правильно установлена в слот PCIe")
    print("3. Проверьте питание GPU (требуется 8-pin коннектор)")
    print("4. Отключите встроенную графику в BIOS если есть")
    print("5. Используйте последние драйверы NVIDIA Game Ready или Studio")