#!/usr/bin/env python3
"""
🔍 VERIFICACIÓN SIMPLE DE FEATURES FALTANTES
============================================

Script simple para verificar qué features faltan en esta PC
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Importar el motor centralizado
try:
    from centralized_features_engine3 import CentralizedFeaturesEngine
    print("✅ Motor de features importado correctamente")
except ImportError as e:
    print(f"❌ Error importando motor: {e}")
    exit(1)

def generate_test_data(periods=100):
    """Generar datos de prueba"""
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

def main():
    print("🔍 VERIFICANDO FEATURES FALTANTES")
    print("=" * 40)
    
    # Crear motor de features
    engine = CentralizedFeaturesEngine()
    
    # Obtener features esperadas
    expected_features = engine.feature_sets['tcn_definitivo']
    print(f"📊 Features esperadas (tcn_definitivo): {len(expected_features)}")
    
    # Generar datos de prueba
    test_data = generate_test_data()
    print(f"📊 Datos de prueba generados: {test_data.shape}")
    
    # Calcular features
    try:
        features_df = engine.calculate_features(test_data, feature_set='tcn_definitivo')
        calculated_features = list(features_df.columns)
        print(f"📊 Features calculadas: {len(calculated_features)}")
        
        # Comparar
        expected_set = set(expected_features)
        calculated_set = set(calculated_features)
        
        missing_features = expected_set - calculated_set
        extra_features = calculated_set - expected_set
        
        print(f"\n🔍 ANÁLISIS DE DIFERENCIAS:")
        print(f"   📊 Esperadas: {len(expected_features)}")
        print(f"   📊 Calculadas: {len(calculated_features)}")
        print(f"   ❌ Faltantes: {len(missing_features)}")
        print(f"   ➕ Extras: {len(extra_features)}")
        
        if missing_features:
            print(f"\n❌ FEATURES FALTANTES ({len(missing_features)}):")
            for i, feature in enumerate(sorted(missing_features), 1):
                print(f"   {i}. {feature}")
                
        if extra_features:
            print(f"\n➕ FEATURES EXTRAS ({len(extra_features)}):")
            for i, feature in enumerate(sorted(extra_features), 1):
                print(f"   {i}. {feature}")
        
        # Análisis por categorías de features faltantes
        if missing_features:
            print(f"\n📊 ANÁLISIS POR CATEGORÍAS DE FEATURES FALTANTES:")
            categories = {
                'volatility': [f for f in missing_features if 'volatility' in f],
                'momentum': [f for f in missing_features if 'momentum' in f],
                'price': [f for f in missing_features if 'price' in f],
                'technical': [f for f in missing_features if any(x in f for x in ['rsi', 'macd', 'bb', 'sma', 'ema'])],
                'volume': [f for f in missing_features if 'volume' in f or 'ad' in f or 'obv' in f],
            }
            
            for category, features in categories.items():
                if features:
                    print(f"   📈 {category.upper()}: {len(features)} features")
                    for feature in features:
                        print(f"      - {feature}")
        
        # Verificar TA-Lib específicamente
        print(f"\n🔍 VERIFICACIÓN DE TA-LIB:")
        try:
            import talib
            print("   ✅ TA-Lib importado correctamente")
            
            # Probar una función simple
            test_prices = test_data['close'].values.astype(float)
            test_rsi = talib.RSI(test_prices, timeperiod=14)
            print(f"   ✅ RSI calculado: {len(test_rsi[~np.isnan(test_rsi)])} valores válidos")
            
        except ImportError:
            print("   ❌ TA-Lib NO disponible")
        except Exception as e:
            print(f"   ⚠️ Error usando TA-Lib: {e}")
            
    except Exception as e:
        print(f"❌ Error calculando features: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
