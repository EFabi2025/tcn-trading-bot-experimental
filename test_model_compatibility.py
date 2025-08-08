#!/usr/bin/env python3
"""
🔍 TEST DE COMPATIBILIDAD ENTRE ENTRENADOR Y PREDICTOR
Verifica que las mejoras implementadas funcionan correctamente
"""

import os
import sys
import json
import pandas as pd
from datetime import datetime

# Agregar el directorio actual al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_config_loading():
    """🔧 Probar carga de configuración de modelos"""
    print("🔧 Probando carga de configuración...")

    # Verificar archivos config.json en modelos
    model_dirs = [
        "models/adaptive_btcusdt_3m_6h_32w",
        "models/adaptive_ethusdt_3m_3h_24w",
        "models/adaptive_bnbusdt_5m_6h_48w",
        "models/adaptive_xrpusdt_3m_3h_24w",
        "models/adaptive_dotusdt_3m_6h_32w"
    ]

    for model_dir in model_dirs:
        config_path = os.path.join(model_dir, "config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                print(f"✅ {model_dir}:")
                print(f"   - lookback_window: {config.get('lookback_window', 'N/A')}")
                print(f"   - timeframe: {config.get('timeframe', 'N/A')}")
                print(f"   - feature_set: {config.get('feature_set', 'N/A')}")
            except Exception as e:
                print(f"❌ Error leyendo {config_path}: {e}")
        else:
            print(f"⚠️ No encontrado: {config_path}")

def test_predictor_initialization():
    """🔧 Probar inicialización del predictor mejorado"""
    print("\n🔧 Probando inicialización del predictor...")

    try:
        from tcn_definitivo_predictor import TCNDefinitivoPredictor

        predictor = TCNDefinitivoPredictor()
        print("✅ Predictor inicializado correctamente")

        # Verificar que tiene los atributos nuevos
        if hasattr(predictor, 'timeframe'):
            print(f"✅ Timeframe configurado: {predictor.timeframe}")
        else:
            print("⚠️ Timeframe no configurado")

        if hasattr(predictor, 'feature_set'):
            print(f"✅ Feature set configurado: {predictor.feature_set}")
        else:
            print("⚠️ Feature set no configurado")

        return predictor

    except Exception as e:
        print(f"❌ Error inicializando predictor: {e}")
        return None

def test_model_loading(predictor):
    """🔧 Probar carga de modelos con configuración"""
    print("\n🔧 Probando carga de modelos...")

    if not predictor:
        return

    symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'DOTUSDT']

    for symbol in symbols:
        try:
            print(f"\n📊 Probando {symbol}...")
            success = predictor._load_model_for_symbol(symbol)

            if success:
                print(f"✅ {symbol} cargado correctamente")

                # Verificar configuración cargada
                if symbol in predictor.sequence_lengths:
                    print(f"   - sequence_length: {predictor.sequence_lengths[symbol]}")
                else:
                    print("   ⚠️ sequence_length no configurado")

                if hasattr(predictor, 'timeframe'):
                    print(f"   - timeframe: {predictor.timeframe}")

                if hasattr(predictor, 'feature_set'):
                    print(f"   - feature_set: {predictor.feature_set}")

            else:
                print(f"❌ Error cargando {symbol}")

        except Exception as e:
            print(f"❌ Error probando {symbol}: {e}")

def test_feature_creation(predictor):
    """🔧 Probar creación de features con configuración correcta"""
    print("\n🔧 Probando creación de features...")

    if not predictor:
        return

    # Crear datos de prueba
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1min')
    test_data = pd.DataFrame({
        'open': [100 + i * 0.1 for i in range(100)],
        'high': [101 + i * 0.1 for i in range(100)],
        'low': [99 + i * 0.1 for i in range(100)],
        'close': [100.5 + i * 0.1 for i in range(100)],
        'volume': [1000 + i * 10 for i in range(100)]
    }, index=dates)

    try:
        # Probar creación de features
        features = predictor.create_features(test_data, 'BTCUSDT')
        print(f"✅ Features creados: {features.shape}")
        print(f"   - Filas: {features.shape[0]}")
        print(f"   - Columnas: {features.shape[1]}")

        # Verificar que no hay NaN
        nan_count = features.isna().sum().sum()
        if nan_count == 0:
            print("✅ Sin valores NaN en features")
        else:
            print(f"⚠️ {nan_count} valores NaN encontrados")

    except Exception as e:
        print(f"❌ Error creando features: {e}")

def test_compatibility_validation(predictor):
    """🔧 Probar validación de compatibilidad"""
    print("\n🔧 Probando validación de compatibilidad...")

    if not predictor:
        return

    # Crear datos de prueba
    dates = pd.date_range(start='2024-01-01', periods=50, freq='1min')
    test_data = pd.DataFrame({
        'open': [100 + i * 0.1 for i in range(50)],
        'high': [101 + i * 0.1 for i in range(50)],
        'low': [99 + i * 0.1 for i in range(50)],
        'close': [100.5 + i * 0.1 for i in range(50)],
        'volume': [1000 + i * 10 for i in range(50)]
    }, index=dates)

    try:
        features = predictor.create_features(test_data, 'BTCUSDT')

        # Probar validación
        is_compatible = predictor.validate_model_compatibility('BTCUSDT', features)

        if is_compatible:
            print("✅ Compatibilidad validada correctamente")
        else:
            print("❌ Incompatibilidad detectada")

    except Exception as e:
        print(f"❌ Error en validación: {e}")

def main():
    """🔧 Función principal de pruebas"""
    print("🔍 TEST DE COMPATIBILIDAD ENTRE ENTRENADOR Y PREDICTOR")
    print("=" * 60)

    # Probar carga de configuración
    test_config_loading()

    # Probar inicialización del predictor
    predictor = test_predictor_initialization()

    # Probar carga de modelos
    test_model_loading(predictor)

    # Probar creación de features
    test_feature_creation(predictor)

    # Probar validación de compatibilidad
    test_compatibility_validation(predictor)

    print("\n🎉 Pruebas completadas")

if __name__ == "__main__":
    main()
