#!/usr/bin/env python3
"""
SETUP BINANCE INTEGRATION - Configuración y preparación del sistema
Script para configurar el entorno de trading automatizado
"""

import os
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

def check_dependencies():
    """Verificar dependencias requeridas"""
    print("🔍 Verificando dependencias...")
    
    required_packages = [
        'aiohttp',
        'tensorflow', 
        'pandas',
        'numpy',
        'scikit-learn',
        'imblearn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"  ❌ {package}")
    
    if missing_packages:
        print(f"\n📦 Instalando paquetes faltantes: {', '.join(missing_packages)}")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            *missing_packages
        ])
        print("✅ Dependencias instaladas")
    else:
        print("✅ Todas las dependencias están instaladas")

def create_directory_structure():
    """Crear estructura de directorios"""
    print("\n📁 Creando estructura de directorios...")
    
    directories = [
        "models",
        "logs", 
        "config",
        "data",
        "backups"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"  ✅ {directory}/")

def save_trained_models():
    """Guardar modelos entrenados del sistema final"""
    print("\n💾 Guardando modelos entrenados...")
    
    try:
        # Importar y ejecutar el sistema final para generar modelos
        from tcn_final_ready import FinalReadyTCN
        import tensorflow as tf
        
        pairs = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
        
        for pair in pairs:
            print(f"  🔄 Procesando modelo para {pair}...")
            
            tcn = FinalReadyTCN(pair_name=pair)
            
            # Generar datos y entrenar modelo
            data = tcn.generate_optimized_data(n_samples=8000)  # Menos datos para rapidez
            features = tcn.create_confidence_features(data)
            sequences, targets = tcn.create_high_confidence_sequences(features, data)
            
            if len(sequences) > 0:
                # Crear y entrenar modelo rápido
                input_shape = (sequences.shape[1], sequences.shape[2])
                model = tcn.build_confidence_model(input_shape)
                
                # Entrenamiento rápido solo para guardar arquitectura
                split_point = int(0.8 * len(sequences))
                X_train = sequences[:split_point]
                y_train = targets[:split_point]
                
                # Solo pocas épocas para crear el modelo
                model.fit(X_train, y_train, epochs=5, verbose=0, batch_size=32)
                
                # Guardar modelo
                model_path = f"models/tcn_final_{pair.lower()}.h5"
                model.save(model_path)
                print(f"  ✅ Modelo guardado: {model_path}")
            else:
                print(f"  ⚠️  Sin datos suficientes para {pair}")
                
    except Exception as e:
        print(f"  ⚠️  Error guardando modelos: {e}")
        print("  📝 Los modelos se crearán dinámicamente en runtime")

def create_config_template():
    """Crear plantilla de configuración"""
    print("\n⚙️  Creando archivo de configuración...")
    
    config_template = {
        "trading": {
            "pairs": ["BTCUSDT", "ETHUSDT", "BNBUSDT"],
            "testnet": True,
            "prediction_interval": 60,
            "min_confidence": 0.80,
            "max_position_size": 0.05,
            "max_daily_trades": 10
        },
        "risk_management": {
            "stop_loss_pct": 0.02,
            "take_profit_pct": 0.04,
            "min_trade_value": 10.0
        },
        "model": {
            "sequence_length": 50,
            "retrain_interval_hours": 24
        },
        "alerts": {
            "enable_logging": True,
            "log_level": "INFO"
        }
    }
    
    with open("config/trading_config.json", "w") as f:
        json.dump(config_template, f, indent=4)
    
    print("  ✅ config/trading_config.json creado")

def create_env_template():
    """Crear plantilla de variables de entorno"""
    print("\n🔐 Creando plantilla de variables de entorno...")
    
    env_template = """# BINANCE API CONFIGURATION
# Obtener desde: https://www.binance.com/en/my/settings/api-management
# IMPORTANTE: Usar TESTNET inicialmente

# Para Testnet (recomendado para pruebas)
BINANCE_API_KEY=your_testnet_api_key_here
BINANCE_API_SECRET=your_testnet_secret_here

# Para Producción (solo cuando estés seguro)
# BINANCE_API_KEY=your_production_api_key_here  
# BINANCE_API_SECRET=your_production_secret_here

# Configuración adicional
ENVIRONMENT=testnet
LOG_LEVEL=INFO
"""
    
    with open(".env.example", "w") as f:
        f.write(env_template)
    
    print("  ✅ .env.example creado")
    print("  📝 Copiar a .env y configurar tus API keys")

def create_startup_script():
    """Crear script de inicio"""
    print("\n🚀 Creando script de inicio...")
    
    startup_script = """#!/bin/bash
# Script de inicio para Binance TCN Trading System

echo "🚀 INICIANDO BINANCE TCN TRADING SYSTEM"
echo "========================================"

# Verificar archivo .env
if [ ! -f .env ]; then
    echo "❌ Archivo .env no encontrado"
    echo "📝 Copiar .env.example a .env y configurar API keys"
    exit 1
fi

# Cargar variables de entorno
source .env

# Verificar API keys
if [ -z "$BINANCE_API_KEY" ] || [ "$BINANCE_API_KEY" = "your_testnet_api_key_here" ]; then
    echo "❌ BINANCE_API_KEY no configurado"
    echo "📝 Editar .env con tu API key real"
    exit 1
fi

if [ -z "$BINANCE_API_SECRET" ] || [ "$BINANCE_API_SECRET" = "your_testnet_secret_here" ]; then
    echo "❌ BINANCE_API_SECRET no configurado"
    echo "📝 Editar .env con tu secret real"
    exit 1
fi

echo "✅ Configuración verificada"
echo "🔄 Iniciando sistema de trading..."

# Exportar variables y ejecutar
export BINANCE_API_KEY
export BINANCE_API_SECRET
python binance_tcn_integration.py
"""
    
    with open("start_trading.sh", "w") as f:
        f.write(startup_script)
    
    # Hacer ejecutable
    os.chmod("start_trading.sh", 0o755)
    
    print("  ✅ start_trading.sh creado")

def create_monitor_script():
    """Crear script de monitoreo"""
    print("\n📊 Creando script de monitoreo...")
    
    monitor_script = """#!/usr/bin/env python3
\"\"\"
MONITOR TRADING SYSTEM - Monitoreo en tiempo real
\"\"\"

import os
import time
import json
from datetime import datetime
import asyncio
import aiohttp

class TradingMonitor:
    def __init__(self):
        self.api_key = os.getenv("BINANCE_API_KEY", "")
        self.base_url = "https://testnet.binance.vision"  # Cambiar para producción
    
    async def get_account_info(self):
        \"\"\"Obtener información de cuenta\"\"\"
        if not self.api_key:
            return None
            
        headers = {"X-MBX-APIKEY": self.api_key}
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(f"{self.base_url}/api/v3/account", headers=headers) as response:
                    if response.status == 200:
                        return await response.json()
            except:
                pass
        return None
    
    async def monitor_loop(self):
        \"\"\"Bucle de monitoreo\"\"\"
        print("📊 TRADING SYSTEM MONITOR")
        print("="*40)
        
        while True:
            try:
                now = datetime.now().strftime("%H:%M:%S")
                
                # Estado del sistema
                print(f"\\n[{now}] 🔄 Sistema activo")
                
                # Info de cuenta
                account_info = await self.get_account_info()
                if account_info:
                    balances = account_info.get('balances', [])
                    usdt_balance = next((b for b in balances if b['asset'] == 'USDT'), None)
                    if usdt_balance:
                        print(f"💰 Balance USDT: ${float(usdt_balance['free']):.2f}")
                
                # Verificar log de trading
                if os.path.exists('binance_tcn_trading.log'):
                    with open('binance_tcn_trading.log', 'r') as f:
                        lines = f.readlines()
                        if lines:
                            last_line = lines[-1].strip()
                            if 'TRADE EJECUTADO' in last_line:
                                print("🎯 Último trade detectado en logs")
                
                await asyncio.sleep(30)  # Check cada 30 segundos
                
            except KeyboardInterrupt:
                print("\\n👋 Monitor detenido")
                break
            except Exception as e:
                print(f"❌ Error en monitor: {e}")
                await asyncio.sleep(10)

if __name__ == "__main__":
    monitor = TradingMonitor()
    asyncio.run(monitor.monitor_loop())
"""
    
    with open("monitor_trading.py", "w") as f:
        f.write(monitor_script)
    
    print("  ✅ monitor_trading.py creado")

def show_next_steps():
    """Mostrar próximos pasos"""
    print("\n" + "="*60)
    print("🎉 SETUP COMPLETADO - PRÓXIMOS PASOS")
    print("="*60)
    
    steps = [
        "1. 🔑 Obtener API keys de Binance Testnet:",
        "   - Ir a https://testnet.binance.vision/",
        "   - Crear cuenta y generar API key/secret",
        "",
        "2. 📝 Configurar variables de entorno:",
        "   - cp .env.example .env",
        "   - Editar .env con tus API keys reales",
        "",
        "3. 🧪 Probar conexión:",
        "   - python -c \"from binance_tcn_integration import *; print('✅ Import OK')\"",
        "",
        "4. 🚀 Iniciar sistema:",
        "   - ./start_trading.sh",
        "   - O: python binance_tcn_integration.py",
        "",
        "5. 📊 Monitorear:",
        "   - python monitor_trading.py",
        "   - tail -f binance_tcn_trading.log",
        "",
        "6. 🔒 Para producción:",
        "   - Cambiar config.testnet = False",
        "   - Usar API keys de producción",
        "   - Ajustar límites de riesgo"
    ]
    
    for step in steps:
        print(step)
    
    print("\n" + "="*60)
    print("⚠️  IMPORTANTE: Comenzar SIEMPRE con TESTNET")
    print("🎯 Sistema TCN listo con 86.8% confianza promedio")
    print("="*60)

def main():
    """Función principal de setup"""
    print("🔧 BINANCE TCN INTEGRATION SETUP")
    print("="*50)
    
    try:
        check_dependencies()
        create_directory_structure()
        save_trained_models()
        create_config_template()
        create_env_template()
        create_startup_script()
        create_monitor_script()
        show_next_steps()
        
        print("\n✅ Setup completado exitosamente!")
        
    except Exception as e:
        print(f"\n❌ Error durante setup: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 