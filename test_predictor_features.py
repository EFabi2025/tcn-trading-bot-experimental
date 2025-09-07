#!/usr/bin/env python3
"""
🔍 TEST DEL PREDICTOR TCN ENSEMBLE
=================================

Script para verificar qué features calcula el predictor en tiempo real
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Importar predictor y motor
try:
    from tcn_ensemble_predictor import TCNEnsemblePredictor
    from centralized_features_engine3 import CentralizedFeaturesEngine
    print("✅ Imports exitosos")
except ImportError as e:
    print(f"❌ Error importando: {e}")
    exit(1)

def generate_btc_test_data(periods=100):
    """Generar datos BTCUSDT realistas"""
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=periods, freq='5T')
    
    base_price = 45000
    returns = np.random.normal(0, 0.02, periods)
    prices = base_price * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices * (1 + np.random.normal(0, 0.005, periods)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.005, periods))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.005, periods))),
        'close': prices,
        'volume': np.random.lognormal(16, 0.3, periods)
    })
    
    # Asegurar consistencia OHLC
    df['high'] = np.maximum.reduce([df['open'], df['high'], df['low'], df['close']])
    df['low'] = np.minimum.reduce([df['open'], df['high'], df['low'], df['close']])
    
    return df

def test_motor_centralizado():
    """Test del motor centralizado solo"""
    print("\n🔍 TEST MOTOR CENTRALIZADO")
    print("=" * 40)
    
    engine = CentralizedFeaturesEngine()
    test_data = generate_btc_test_data()
    
    print(f"📊 Datos generados: {test_data.shape}")
    
    try:
        features = engine.calculate_features(test_data, feature_set='tcn_definitivo')
        print(f"✅ Features calculadas por motor: {features.shape}")
        print(f"📋 Columnas: {list(features.columns)}")
        return features.shape[1]
    except Exception as e:
        print(f"❌ Error en motor: {e}")
        return None

def test_predictor_features():
    """Test del predictor TCN"""
    print("\n🔍 TEST PREDICTOR TCN ENSEMBLE")
    print("=" * 40)
    
    try:
        predictor = TCNEnsemblePredictor()
        print("✅ Predictor inicializado")
        
        # Verificar modelos disponibles
        print(f"📊 Modelos disponibles: {list(predictor.models.keys())}")
        
        # Verificar si hay modelo de BTCUSDT
        btc_models = [k for k in predictor.models.keys() if 'btcusdt' in k.lower()]
        print(f"📊 Modelos de BTC: {btc_models}")
        
        if not btc_models:
            print("⚠️ No hay modelos de BTCUSDT disponibles")
            return None
            
        # Usar el primer modelo BTC disponible
        model_key = btc_models[0]
        print(f"🎯 Usando modelo: {model_key}")
        
        # Verificar input shape del modelo
        model = predictor.models[model_key]
        input_shape = model.input_shape
        print(f"📊 Input shape del modelo: {input_shape}")
        
        if len(input_shape) >= 3:
            expected_features = input_shape[-1]
            expected_sequence = input_shape[1]
            print(f"📊 Features esperadas: {expected_features}")
            print(f"📊 Secuencia esperada: {expected_sequence}")
            
            return expected_features
        else:
            print("⚠️ Input shape no reconocido")
            return None
            
    except Exception as e:
        print(f"❌ Error en predictor: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_features_compatibility():
    """Test de compatibilidad entre motor y predictor"""
    print("\n🔍 TEST COMPATIBILIDAD FEATURES")
    print("=" * 40)
    
    motor_features = test_motor_centralizado()
    predictor_features = test_predictor_features()
    
    if motor_features and predictor_features:
        print(f"\n📊 RESUMEN COMPATIBILIDAD:")
        print(f"   🔧 Motor calcula: {motor_features} features")
        print(f"   🤖 Predictor espera: {predictor_features} features")
        print(f"   📊 Diferencia: {motor_features - predictor_features}")
        
        if motor_features == predictor_features:
            print("   ✅ COMPATIBLE")
        else:
            print("   ❌ INCOMPATIBLE")
            if motor_features > predictor_features:
                print(f"   ⚠️ Motor calcula {motor_features - predictor_features} features EXTRA")
            else:
                print(f"   ⚠️ Motor calcula {predictor_features - motor_features} features MENOS")
                
        return motor_features, predictor_features
    else:
        print("❌ No se pudo completar la comparación")
        return None, None

def test_features_detailed_comparison():
    """Comparación detallada de features"""
    print("\n🔍 TEST DETALLADO DE FEATURES")
    print("=" * 40)
    
    try:
        # Motor centralizado
        engine = CentralizedFeaturesEngine()
        expected_features_list = engine.feature_sets['tcn_definitivo']
        
        # Calcular features
        test_data = generate_btc_test_data()
        calculated_features = engine.calculate_features(test_data, feature_set='tcn_definitivo')
        calculated_features_list = list(calculated_features.columns)
        
        print(f"📋 Features esperadas (tcn_definitivo): {len(expected_features_list)}")
        print(f"📋 Features calculadas: {len(calculated_features_list)}")
        
        # Comparar listas
        expected_set = set(expected_features_list)
        calculated_set = set(calculated_features_list)
        
        missing = expected_set - calculated_set
        extra = calculated_set - expected_set
        
        if missing:
            print(f"\n❌ FEATURES FALTANTES ({len(missing)}):")
            for feature in sorted(missing):
                print(f"   - {feature}")
                
        if extra:
            print(f"\n➕ FEATURES EXTRA ({len(extra)}):")
            for feature in sorted(extra):
                print(f"   - {feature}")
                
        if not missing and not extra:
            print("✅ Todas las features coinciden perfectamente")
            
        return expected_features_list, calculated_features_list
        
    except Exception as e:
        print(f"❌ Error en comparación detallada: {e}")
        return None, None

def main():
    """Función principal"""
    print("🚀 DIAGNÓSTICO COMPLETO DE FEATURES")
    print("=" * 50)
    
    # Test 1: Compatibilidad general
    motor_count, predictor_count = test_features_compatibility()
    
    # Test 2: Comparación detallada
    expected_list, calculated_list = test_features_detailed_comparison()
    
    # Resumen final
    print(f"\n📊 RESUMEN FINAL")
    print("=" * 30)
    if motor_count and predictor_count:
        if motor_count == predictor_count:
            print("✅ Sistema COMPATIBLE")
        else:
            print("❌ Sistema INCOMPATIBLE")
            print(f"🔧 Motor: {motor_count}, Predictor: {predictor_count}")
    else:
        print("⚠️ No se pudo determinar compatibilidad")

if __name__ == "__main__":
    main()
