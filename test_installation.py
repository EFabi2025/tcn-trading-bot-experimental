#!/usr/bin/env python3
"""
🧪 VERIFICACIÓN DE INSTALACIÓN - TRADING BOT PROFESIONAL
Script que verifica que toda la instalación esté correcta
"""

import os
import sys
import importlib
from pathlib import Path
from dotenv import load_dotenv

def print_section(title):
    """Imprimir sección de verificación"""
    print(f"\n{'='*60}")
    print(f"🔍 {title}")
    print(f"{'='*60}")

def check_python_version():
    """Verificar versión de Python"""
    version = sys.version_info
    print(f"🐍 Python: {version.major}.{version.minor}.{version.micro}")
    
    if version.major >= 3 and version.minor >= 8:
        print("✅ Versión de Python correcta")
        return True
    else:
        print("❌ Se requiere Python 3.8+")
        return False

def check_dependencies():
    """Verificar dependencias críticas"""
    dependencies = [
        ('pandas', 'Análisis de datos'),
        ('numpy', 'Cálculos numéricos'),
        ('aiohttp', 'Cliente HTTP asíncrono'),
        ('python-dotenv', 'Variables de entorno'),
        ('tensorflow', 'Machine Learning'),
        ('scikit-learn', 'ML Tools'),
        ('sqlite3', 'Base de datos (built-in)'),
        ('asyncio', 'Programación asíncrona (built-in)')
    ]
    
    all_ok = True
    for module_name, description in dependencies:
        try:
            if module_name in ['sqlite3', 'asyncio']:
                importlib.import_module(module_name)
            else:
                __import__(module_name)
            print(f"✅ {module_name}: {description}")
        except ImportError:
            print(f"❌ {module_name}: FALTANTE - {description}")
            all_ok = False
    
    return all_ok

def check_core_files():
    """Verificar archivos core del sistema"""
    core_files = {
        'run_trading_manager.py': 'Script principal de ejecución',
        'simple_professional_manager.py': 'Manager principal de trading',
        'professional_portfolio_manager.py': 'Gestión de portafolio',
        'advanced_risk_manager.py': 'Sistema de gestión de riesgo',
        'trading_database.py': 'Sistema de base de datos',
        'smart_discord_notifier.py': 'Notificaciones Discord'
    }
    
    all_ok = True
    for file_name, description in core_files.items():
        if Path(file_name).exists():
            print(f"✅ {file_name}: {description}")
        else:
            print(f"❌ {file_name}: FALTANTE - {description}")
            all_ok = False
    
    return all_ok

def check_model_files():
    """Verificar archivos de modelos ML"""
    model_patterns = [
        'production_model_*.h5',
        'best_model_*.h5',
        'ultra_model_*.h5'
    ]
    
    models_found = []
    for pattern in model_patterns:
        models = list(Path('.').glob(pattern))
        models_found.extend(models)
    
    if models_found:
        print(f"✅ Modelos ML encontrados: {len(models_found)}")
        for model in models_found[:5]:  # Mostrar solo los primeros 5
            print(f"   📁 {model}")
        if len(models_found) > 5:
            print(f"   📁 ... y {len(models_found) - 5} más")
        return True
    else:
        print("⚠️ No se encontraron modelos ML (.h5)")
        print("   El sistema puede funcionar sin ML (modo básico)")
        return True  # No crítico

def check_environment_file():
    """Verificar archivo de configuración"""
    env_file = Path('.env')
    env_example = Path('.env.example')
    
    if not env_file.exists():
        print("❌ Archivo .env no encontrado")
        if env_example.exists():
            print("   💡 Ejecuta: cp .env.example .env")
        return False
    
    # Cargar variables de entorno
    load_dotenv()
    
    required_vars = [
        'BINANCE_API_KEY',
        'BINANCE_SECRET_KEY', 
        'BINANCE_BASE_URL',
        'ENVIRONMENT'
    ]
    
    all_configured = True
    for var in required_vars:
        value = os.getenv(var)
        if not value or value == f'tu_{var.lower()}_aqui':
            print(f"❌ {var}: No configurado")
            all_configured = False
        else:
            # Ocultar valores sensibles
            if 'KEY' in var:
                display_value = f"{value[:8]}...{value[-4:]}" if len(value) > 12 else "***"
            else:
                display_value = value
            print(f"✅ {var}: {display_value}")
    
    return all_configured

def check_binance_connection():
    """Verificar conexión básica con Binance"""
    try:
        import aiohttp
        import asyncio
        import hmac
        import hashlib
        import time
        
        load_dotenv()
        
        api_key = os.getenv('BINANCE_API_KEY')
        secret_key = os.getenv('BINANCE_SECRET_KEY')
        base_url = os.getenv('BINANCE_BASE_URL')
        
        if not all([api_key, secret_key, base_url]):
            print("❌ Credenciales de Binance no configuradas")
            return False
        
        async def test_connection():
            try:
                # Test básico de conectividad
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{base_url}/api/v3/time") as response:
                        if response.status == 200:
                            print("✅ Conexión con Binance exitosa")
                            return True
                        else:
                            print(f"❌ Error conexión Binance: {response.status}")
                            return False
            except Exception as e:
                print(f"❌ Error conectando con Binance: {e}")
                return False
        
        # Ejecutar test asíncrono
        return asyncio.run(test_connection())
        
    except ImportError as e:
        print(f"❌ Error importando módulos para test Binance: {e}")
        return False
    except Exception as e:
        print(f"❌ Error inesperado en test Binance: {e}")
        return False

def check_database():
    """Verificar sistema de base de datos"""
    try:
        import sqlite3
        
        # Test conexión SQLite
        conn = sqlite3.connect(':memory:')
        cursor = conn.cursor()
        cursor.execute('SELECT sqlite_version()')
        version = cursor.fetchone()[0]
        conn.close()
        
        print(f"✅ SQLite: {version}")
        return True
        
    except Exception as e:
        print(f"❌ Error en SQLite: {e}")
        return False

def run_system_test():
    """Ejecutar test rápido del sistema"""
    try:
        # Importar y crear instancia del manager
        from simple_professional_manager import SimpleProfessionalTradingManager
        
        manager = SimpleProfessionalTradingManager()
        print("✅ Sistema principal: Puede instanciarse")
        
        # Test configuración
        config = manager._load_config()
        if config.api_key and config.secret_key:
            print("✅ Configuración: Credenciales cargadas")
        else:
            print("⚠️ Configuración: Credenciales no configuradas")
        
        return True
        
    except ImportError as e:
        print(f"❌ Error importando sistema principal: {e}")
        return False
    except Exception as e:
        print(f"❌ Error en test del sistema: {e}")
        return False

def print_summary(results):
    """Imprimir resumen de verificación"""
    print(f"\n{'🎯 RESUMEN DE VERIFICACIÓN':^60}")
    print("="*60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test:.<40} {status}")
    
    print("="*60)
    print(f"📊 RESULTADO: {passed}/{total} verificaciones exitosas")
    
    if passed == total:
        print("🎉 ¡INSTALACIÓN COMPLETA Y FUNCIONAL!")
        print("🚀 Ya puedes ejecutar: python run_trading_manager.py")
    elif passed >= total - 2:
        print("⚠️ Instalación mayormente completa")
        print("🔧 Revisa los elementos fallidos arriba")
    else:
        print("❌ Instalación incompleta")
        print("📖 Revisa SETUP_COMPLETO.md para instrucciones")

def main():
    """Función principal de verificación"""
    print("🧪 VERIFICACIÓN DE INSTALACIÓN - TRADING BOT PROFESIONAL")
    print("="*60)
    print("🎯 Verificando que toda la instalación esté correcta...")
    
    results = {}
    
    # Verificación Python
    print_section("PYTHON Y DEPENDENCIAS")
    results['Python Version'] = check_python_version()
    results['Dependencies'] = check_dependencies()
    
    # Verificación archivos
    print_section("ARCHIVOS DEL SISTEMA")
    results['Core Files'] = check_core_files()
    results['ML Models'] = check_model_files()
    
    # Verificación configuración
    print_section("CONFIGURACIÓN")
    results['Environment File'] = check_environment_file()
    results['Database'] = check_database()
    
    # Verificación conectividad
    print_section("CONECTIVIDAD")
    results['Binance Connection'] = check_binance_connection()
    
    # Test del sistema
    print_section("SISTEMA PRINCIPAL")
    results['System Test'] = run_system_test()
    
    # Resumen final
    print_summary(results)

if __name__ == "__main__":
    main() 