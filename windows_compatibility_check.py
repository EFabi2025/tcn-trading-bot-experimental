#!/usr/bin/env python3
"""
🪟 VERIFICADOR DE COMPATIBILIDAD WINDOWS
Sistema de Trading Bot con Diversificación de Portafolio

Este script verifica que todos los componentes sean compatibles con Windows
y proporciona diagnósticos detallados.
"""

import os
import sys
import platform
import subprocess
import importlib
from pathlib import Path
from typing import Dict, List, Tuple, Any
import logging

# Configurar logging para Windows
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class WindowsCompatibilityChecker:
    """Verificador de compatibilidad para Windows"""

    def __init__(self):
        self.results = {}
        self.errors = []
        self.warnings = []

    def check_system_info(self) -> Dict[str, Any]:
        """Verificar información del sistema"""
        logger.info("🔍 Verificando información del sistema...")

        system_info = {
            'platform': platform.system(),
            'platform_release': platform.release(),
            'platform_version': platform.version(),
            'architecture': platform.architecture(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'python_version': sys.version,
            'python_executable': sys.executable
        }

        # Verificar que sea Windows
        if system_info['platform'] != 'Windows':
            self.warnings.append(f"⚠️  Sistema detectado: {system_info['platform']} (esperado: Windows)")
        else:
            logger.info(f"✅ Sistema Windows detectado: {system_info['platform_release']}")

        # Verificar versión Python
        python_version = sys.version_info
        if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
            self.errors.append(f"❌ Python {python_version.major}.{python_version.minor} no soportado. Requiere Python 3.8+")
        else:
            logger.info(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro} compatible")

        self.results['system_info'] = system_info
        return system_info

    def check_required_files(self) -> Dict[str, bool]:
        """Verificar archivos requeridos del sistema"""
        logger.info("📁 Verificando archivos del sistema...")

        required_files = [
            'run_trading_manager.py',
            'simple_professional_manager.py',
            'portfolio_diversification_manager.py',
            'config/trading_config.py',
            'requirements.txt',
            '.env.example'
        ]

        file_status = {}
        for file_path in required_files:
            exists = os.path.exists(file_path)
            file_status[file_path] = exists

            if exists:
                logger.info(f"✅ {file_path}")
            else:
                self.errors.append(f"❌ Archivo faltante: {file_path}")

        # Verificar archivo .env
        env_exists = os.path.exists('.env')
        file_status['.env'] = env_exists

        if env_exists:
            logger.info("✅ .env configurado")
        else:
            self.warnings.append("⚠️  Archivo .env no encontrado. Copiar desde .env.example")

        self.results['files'] = file_status
        return file_status

    def check_python_dependencies(self) -> Dict[str, Any]:
        """Verificar dependencias Python"""
        logger.info("📦 Verificando dependencias Python...")

        required_packages = [
            'tensorflow',
            'numpy',
            'pandas',
            'scikit-learn',
            'python-binance',
            'python-dotenv',
            'requests',
            'structlog',
            'pydantic',
            'sqlalchemy'
        ]

        dependency_status = {}

        for package in required_packages:
            try:
                module = importlib.import_module(package.replace('-', '_'))
                version = getattr(module, '__version__', 'unknown')
                dependency_status[package] = {
                    'installed': True,
                    'version': version
                }
                logger.info(f"✅ {package} v{version}")

            except ImportError:
                dependency_status[package] = {
                    'installed': False,
                    'version': None
                }
                self.errors.append(f"❌ Dependencia faltante: {package}")

        self.results['dependencies'] = dependency_status
        return dependency_status

    def check_tensorflow_windows(self) -> Dict[str, Any]:
        """Verificar configuración específica de TensorFlow en Windows"""
        logger.info("🧠 Verificando TensorFlow para Windows...")

        tf_status = {
            'installed': False,
            'version': None,
            'gpu_available': False,
            'cpu_optimized': False
        }

        try:
            import tensorflow as tf
            tf_status['installed'] = True
            tf_status['version'] = tf.__version__

            # Verificar GPU (opcional en Windows)
            try:
                gpus = tf.config.list_physical_devices('GPU')
                tf_status['gpu_available'] = len(gpus) > 0
                if tf_status['gpu_available']:
                    logger.info(f"✅ TensorFlow GPU disponible: {len(gpus)} dispositivos")
                else:
                    logger.info("ℹ️  TensorFlow usando CPU (normal en Windows)")
            except:
                tf_status['gpu_available'] = False

            # Verificar optimizaciones CPU
            try:
                tf.config.threading.set_inter_op_parallelism_threads(0)
                tf.config.threading.set_intra_op_parallelism_threads(0)
                tf_status['cpu_optimized'] = True
                logger.info("✅ TensorFlow optimizado para CPU")
            except:
                self.warnings.append("⚠️  No se pudo optimizar TensorFlow para CPU")

        except ImportError:
            self.errors.append("❌ TensorFlow no instalado")

        self.results['tensorflow'] = tf_status
        return tf_status

    def check_file_paths_windows(self) -> Dict[str, bool]:
        """Verificar compatibilidad de rutas en Windows"""
        logger.info("🛤️  Verificando compatibilidad de rutas...")

        path_tests = {}

        # Test 1: Crear directorio con os.path.join
        try:
            test_dir = os.path.join('temp_test', 'subdir')
            os.makedirs(test_dir, exist_ok=True)
            path_tests['create_nested_dir'] = True
            logger.info("✅ Creación de directorios anidados")

            # Limpiar
            import shutil
            shutil.rmtree('temp_test', ignore_errors=True)

        except Exception as e:
            path_tests['create_nested_dir'] = False
            self.errors.append(f"❌ Error creando directorios: {e}")

        # Test 2: Rutas con Path
        try:
            from pathlib import Path
            test_path = Path('logs') / 'test.log'
            test_path.parent.mkdir(exist_ok=True)
            path_tests['pathlib_support'] = True
            logger.info("✅ Soporte pathlib")

        except Exception as e:
            path_tests['pathlib_support'] = False
            self.errors.append(f"❌ Error con pathlib: {e}")

        self.results['paths'] = path_tests
        return path_tests

    def check_network_connectivity(self) -> Dict[str, bool]:
        """Verificar conectividad de red"""
        logger.info("🌐 Verificando conectividad de red...")

        network_tests = {}

        # Test Binance API
        try:
            import requests
            response = requests.get('https://api.binance.com/api/v3/ping', timeout=10)
            network_tests['binance_api'] = response.status_code == 200

            if network_tests['binance_api']:
                logger.info("✅ Conexión a Binance API")
            else:
                self.errors.append(f"❌ Error conectando a Binance API: {response.status_code}")

        except Exception as e:
            network_tests['binance_api'] = False
            self.errors.append(f"❌ Error de red Binance: {e}")

        self.results['network'] = network_tests
        return network_tests

    def check_environment_variables(self) -> Dict[str, bool]:
        """Verificar variables de entorno"""
        logger.info("🔐 Verificando variables de entorno...")

        env_vars = {}
        required_vars = [
            'BINANCE_API_KEY',
            'BINANCE_SECRET_KEY'
        ]

        optional_vars = [
            'ENVIRONMENT',
            'DISCORD_WEBHOOK_URL'
        ]

        # Cargar .env si existe
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except:
            pass

        # Verificar variables requeridas
        for var in required_vars:
            value = os.getenv(var)
            env_vars[var] = bool(value and value.strip())

            if env_vars[var]:
                logger.info(f"✅ {var} configurado")
            else:
                self.errors.append(f"❌ Variable de entorno faltante: {var}")

        # Verificar variables opcionales
        for var in optional_vars:
            value = os.getenv(var)
            env_vars[var] = bool(value and value.strip())

            if env_vars[var]:
                logger.info(f"✅ {var} configurado")
            else:
                logger.info(f"ℹ️  {var} no configurado (opcional)")

        self.results['environment'] = env_vars
        return env_vars

    def test_diversification_system(self) -> Dict[str, bool]:
        """Probar sistema de diversificación"""
        logger.info("🎯 Probando sistema de diversificación...")

        diversification_tests = {}

        try:
            # Test 1: Importar módulo
            from portfolio_diversification_manager import PortfolioDiversificationManager
            diversification_tests['import_manager'] = True
            logger.info("✅ Importación PortfolioDiversificationManager")

            # Test 2: Cargar configuración
            from config.trading_config import DIVERSIFICATION_CONFIG
            diversification_tests['load_config'] = True
            logger.info("✅ Carga de configuración")

            # Test 3: Inicializar manager
            manager = PortfolioDiversificationManager(DIVERSIFICATION_CONFIG)
            diversification_tests['initialize_manager'] = True
            logger.info("✅ Inicialización del manager")

            # Test 4: Verificar métodos principales
            test_positions = []  # Lista vacía para test
            score = manager.calculate_diversification_score(test_positions)
            diversification_tests['calculate_score'] = isinstance(score, (int, float))
            logger.info(f"✅ Cálculo de score: {score}")

        except Exception as e:
            diversification_tests['system_error'] = str(e)
            self.errors.append(f"❌ Error en sistema de diversificación: {e}")

        self.results['diversification'] = diversification_tests
        return diversification_tests

    def generate_report(self) -> str:
        """Generar reporte completo"""
        logger.info("📊 Generando reporte de compatibilidad...")

        report = []
        report.append("=" * 60)
        report.append("🪟 REPORTE DE COMPATIBILIDAD WINDOWS")
        report.append("Sistema de Trading Bot con Diversificación")
        report.append("=" * 60)
        report.append("")

        # Resumen
        total_errors = len(self.errors)
        total_warnings = len(self.warnings)

        if total_errors == 0:
            report.append("🎉 SISTEMA COMPATIBLE CON WINDOWS")
            report.append("✅ Todos los componentes verificados exitosamente")
        else:
            report.append(f"⚠️  PROBLEMAS DETECTADOS: {total_errors} errores, {total_warnings} advertencias")

        report.append("")

        # Errores críticos
        if self.errors:
            report.append("❌ ERRORES CRÍTICOS:")
            for error in self.errors:
                report.append(f"   {error}")
            report.append("")

        # Advertencias
        if self.warnings:
            report.append("⚠️  ADVERTENCIAS:")
            for warning in self.warnings:
                report.append(f"   {warning}")
            report.append("")

        # Detalles por categoría
        for category, data in self.results.items():
            report.append(f"📋 {category.upper()}:")
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, dict):
                        report.append(f"   {key}:")
                        for subkey, subvalue in value.items():
                            report.append(f"     {subkey}: {subvalue}")
                    else:
                        status = "✅" if value else "❌"
                        report.append(f"   {status} {key}: {value}")
            report.append("")

        # Recomendaciones
        report.append("💡 RECOMENDACIONES PARA WINDOWS:")
        report.append("   1. Usar PowerShell como administrador para instalación")
        report.append("   2. Instalar Visual Studio Build Tools si hay errores de compilación")
        report.append("   3. Agregar exclusiones de antivirus para la carpeta del proyecto")
        report.append("   4. Configurar firewall para permitir conexiones Python")
        report.append("   5. Usar encoding UTF-8 explícito en logs")
        report.append("")

        return "\n".join(report)

    def run_full_check(self) -> bool:
        """Ejecutar verificación completa"""
        logger.info("🚀 Iniciando verificación completa de compatibilidad Windows...")

        try:
            self.check_system_info()
            self.check_required_files()
            self.check_python_dependencies()
            self.check_tensorflow_windows()
            self.check_file_paths_windows()
            self.check_network_connectivity()
            self.check_environment_variables()
            self.test_diversification_system()

            # Generar y mostrar reporte
            report = self.generate_report()
            print(report)

            # Guardar reporte
            with open('windows_compatibility_report.txt', 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info("📄 Reporte guardado en: windows_compatibility_report.txt")

            return len(self.errors) == 0

        except Exception as e:
            logger.error(f"❌ Error durante verificación: {e}")
            return False

def main():
    """Función principal"""
    print("🪟 VERIFICADOR DE COMPATIBILIDAD WINDOWS")
    print("Sistema de Trading Bot con Diversificación de Portafolio")
    print("=" * 60)

    checker = WindowsCompatibilityChecker()
    success = checker.run_full_check()

    if success:
        print("\n🎉 ¡SISTEMA LISTO PARA WINDOWS!")
        print("Puedes proceder con la ejecución del trading bot.")
        return 0
    else:
        print("\n⚠️  PROBLEMAS DETECTADOS")
        print("Revisa los errores arriba y corrígelos antes de continuar.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
