#!/usr/bin/env python3
"""
🚀 INSTALADOR AUTOMÁTICO - TRADING BOT PROFESIONAL
Script que configura automáticamente todo el entorno necesario
"""

import os
import sys
import subprocess
import platform
import shutil
from pathlib import Path

def print_step(step_num, description):
    """Imprimir paso de instalación"""
    print(f"\n{'='*60}")
    print(f"🚀 PASO {step_num}: {description}")
    print(f"{'='*60}")

def run_command(command, description=""):
    """Ejecutar comando y manejar errores"""
    print(f"📝 Ejecutando: {command}")
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        if result.stdout:
            print(f"✅ {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stderr:
            print(f"❌ Detalle: {e.stderr}")
        return False

def check_python_version():
    """Verificar versión de Python"""
    version = sys.version_info
    print(f"🐍 Python detectado: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ ERROR: Se requiere Python 3.8 o superior")
        print("📥 Descarga Python desde: https://python.org/downloads/")
        return False
    
    print("✅ Versión de Python compatible")
    return True

def check_git():
    """Verificar que Git esté instalado"""
    try:
        result = subprocess.run(['git', '--version'], capture_output=True, text=True)
        print(f"✅ Git instalado: {result.stdout.strip()}")
        return True
    except FileNotFoundError:
        print("❌ Git no está instalado")
        print("📥 Instala Git desde: https://git-scm.com/downloads")
        return False

def create_virtual_environment():
    """Crear entorno virtual"""
    venv_path = Path(".venv")
    
    if venv_path.exists():
        print("📁 Entorno virtual ya existe")
        return True
    
    print("🔧 Creando entorno virtual...")
    if not run_command(f"{sys.executable} -m venv .venv"):
        return False
    
    print("✅ Entorno virtual creado")
    return True

def get_activation_command():
    """Obtener comando de activación según el OS"""
    if platform.system() == "Windows":
        return ".venv\\Scripts\\activate"
    else:
        return "source .venv/bin/activate"

def install_dependencies():
    """Instalar dependencias"""
    print("📦 Instalando dependencias...")
    
    # Buscar archivo de requirements
    req_files = [
        "requirements_professional_fixed.txt",
        "requirements_professional.txt", 
        "requirements.txt"
    ]
    
    req_file = None
    for req in req_files:
        if Path(req).exists():
            req_file = req
            break
    
    if not req_file:
        print("❌ No se encontró archivo requirements")
        return False
    
    print(f"📋 Usando: {req_file}")
    
    # Determinar comando pip según OS
    if platform.system() == "Windows":
        pip_cmd = ".venv\\Scripts\\pip"
    else:
        pip_cmd = ".venv/bin/pip"
    
    # Actualizar pip primero
    if not run_command(f"{pip_cmd} install --upgrade pip"):
        return False
    
    # Instalar dependencias
    if not run_command(f"{pip_cmd} install -r {req_file}"):
        return False
    
    print("✅ Dependencias instaladas")
    return True

def setup_environment_file():
    """Configurar archivo .env"""
    env_example = Path(".env.example")
    env_file = Path(".env")
    
    if env_file.exists():
        print("📄 Archivo .env ya existe")
        return True
    
    if not env_example.exists():
        print("⚠️ Creando .env.example...")
        env_content = """# BINANCE API (REQUERIDO)
BINANCE_API_KEY=tu_api_key_aqui
BINANCE_SECRET_KEY=tu_secret_key_aqui
BINANCE_BASE_URL=https://testnet.binance.vision

# ENVIRONMENT
ENVIRONMENT=testnet

# DISCORD (OPCIONAL)
DISCORD_WEBHOOK_URL=tu_webhook_url_aqui

# DATABASE
DATABASE_URL=sqlite:///trading_bot.db
"""
        with open(".env.example", "w") as f:
            f.write(env_content)
    
    # Copiar .env.example a .env
    shutil.copy(".env.example", ".env")
    print("✅ Archivo .env creado")
    print("⚠️ IMPORTANTE: Edita .env con tus credenciales de Binance")
    return True

def verify_installation():
    """Verificar que la instalación esté completa"""
    print("🔍 Verificando instalación...")
    
    # Verificar archivos críticos
    critical_files = [
        "run_trading_manager.py",
        "simple_professional_manager.py", 
        "professional_portfolio_manager.py",
        "advanced_risk_manager.py",
        "trading_database.py"
    ]
    
    missing_files = []
    for file in critical_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Archivos faltantes: {missing_files}")
        return False
    
    print("✅ Todos los archivos críticos presentes")
    return True

def print_next_steps():
    """Mostrar pasos siguientes"""
    activation_cmd = get_activation_command()
    
    print(f"\n{'🎯 INSTALACIÓN COMPLETADA':^60}")
    print("="*60)
    print("\n📋 PRÓXIMOS PASOS:")
    print(f"1️⃣ Activar entorno virtual:")
    print(f"   {activation_cmd}")
    print(f"\n2️⃣ Configurar credenciales Binance en .env:")
    print(f"   nano .env  # o usar tu editor preferido")
    print(f"\n3️⃣ Verificar configuración:")
    print(f"   python test_installation.py")
    print(f"\n4️⃣ Ejecutar sistema de trading:")
    print(f"   python run_trading_manager.py")
    print("\n" + "="*60)
    print("🚨 IMPORTANTE: Configura tus API keys antes de ejecutar")
    print("💡 Usa testnet para pruebas: https://testnet.binance.vision")
    print("📖 Lee SETUP_COMPLETO.md para más detalles")

def main():
    """Función principal de instalación"""
    print("🚀 INSTALADOR AUTOMÁTICO - TRADING BOT PROFESIONAL")
    print("="*60)
    
    try:
        # Paso 1: Verificar Python
        print_step(1, "Verificando Python")
        if not check_python_version():
            sys.exit(1)
        
        # Paso 2: Verificar Git
        print_step(2, "Verificando Git")
        if not check_git():
            sys.exit(1)
        
        # Paso 3: Crear entorno virtual
        print_step(3, "Configurando entorno virtual")
        if not create_virtual_environment():
            sys.exit(1)
        
        # Paso 4: Instalar dependencias
        print_step(4, "Instalando dependencias")
        if not install_dependencies():
            sys.exit(1)
        
        # Paso 5: Configurar .env
        print_step(5, "Configurando archivo de entorno")
        if not setup_environment_file():
            sys.exit(1)
        
        # Paso 6: Verificar instalación
        print_step(6, "Verificando instalación")
        if not verify_installation():
            sys.exit(1)
        
        # Mostrar próximos pasos
        print_next_steps()
        
    except KeyboardInterrupt:
        print("\n❌ Instalación cancelada por el usuario")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 