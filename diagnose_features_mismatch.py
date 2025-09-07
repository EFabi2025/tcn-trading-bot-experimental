#!/usr/bin/env python3
"""
🔍 DIAGNÓSTICO DE DISCREPANCIA DE FEATURES
==========================================

Script para identificar la diferencia entre:
- Features esperadas en entrenamiento (66)
- Features calculadas en predicción (62)
- Features que faltan o sobran

Problema identificado: Los modelos están "dormidos" porque hay inconsistencia
entre el número de features durante entrenamiento y predicción.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from centralized_features_engine3 import CentralizedFeaturesEngine

def diagnose_features_mismatch():
    """🔍 Diagnosticar la discrepancia de features"""
    
    print("🔍 DIAGNÓSTICO DE DISCREPANCIA DE FEATURES")
    print("=" * 50)
    
    # 1. Crear engine de features
    engine = CentralizedFeaturesEngine()
    
    # 2. Obtener features esperadas
    expected_features = engine.feature_sets['tcn_definitivo']
    print(f"\n📊 FEATURES ESPERADAS (66):")
    print(f"   Total: {len(expected_features)}")
    print(f"   Lista: {expected_features}")
    
    # 3. Crear datos de prueba
    print(f"\n🧪 CREANDO DATOS DE PRUEBA...")
    dates = pd.date_range('2024-01-01', periods=100, freq='1H')
    test_data = pd.DataFrame({
        'open': np.random.uniform(100, 200, 100),
        'high': np.random.uniform(200, 300, 100),
        'low': np.random.uniform(50, 100, 100),
        'close': np.random.uniform(100, 200, 100),
        'volume': np.random.uniform(1000, 10000, 100)
    }, index=dates)
    
    # 4. Calcular features
    print(f"\n⚙️ CALCULANDO FEATURES...")
    calculated_features_df = engine.calculate_features(test_data, 'tcn_definitivo')
    
    # 5. Analizar diferencias
    actual_features = list(calculated_features_df.columns)
    print(f"\n📈 FEATURES CALCULADAS:")
    print(f"   Total: {len(actual_features)}")
    print(f"   Lista: {actual_features}")
    
    # 6. Identificar features faltantes
    missing_features = set(expected_features) - set(actual_features)
    extra_features = set(actual_features) - set(expected_features)
    
    print(f"\n❌ FEATURES FALTANTES ({len(missing_features)}):")
    for feature in sorted(missing_features):
        print(f"   - {feature}")
    
    print(f"\n➕ FEATURES EXTRA ({len(extra_features)}):")
    for feature in sorted(extra_features):
        print(f"   - {feature}")
    
    # 7. Analizar por categorías
    print(f"\n📋 ANÁLISIS POR CATEGORÍAS:")
    
    categories = {
        'MOMENTUM': ['rsi_14', 'rsi_21', 'rsi_7', 'macd', 'macd_signal', 'macd_histogram', 
                     'stoch_k', 'stoch_d', 'williams_r', 'roc_10', 'roc_20', 'momentum_10', 
                     'momentum_20', 'cci_14', 'cci_20'],
        'TREND': ['sma_10', 'sma_20', 'sma_50', 'ema_10', 'ema_20', 'ema_50', 'adx_14', 
                  'plus_di', 'minus_di', 'psar', 'aroon_up', 'aroon_down'],
        'VOLATILITY': ['bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_position', 
                       'atr_14', 'atr_20', 'true_range', 'natr_14', 'natr_20'],
        'VOLUME': ['ad', 'adosc', 'obv', 'volume_sma_10', 'volume_sma_20', 'volume_ratio', 
                   'mfi_14', 'mfi_20'],
        'PRICE_PATTERNS': ['hl_ratio', 'oc_ratio', 'price_position', 'price_change_1', 
                           'price_change_5', 'price_change_10', 'volatility_10', 'volatility_20'],
        'MARKET_STRUCTURE': ['higher_high', 'lower_low', 'uptrend_strength', 'downtrend_strength', 
                             'resistance_touch', 'support_touch', 'efficiency_ratio', 'fractal_dimension'],
        'MOMENTUM_DERIVATIVES': ['rsi_momentum', 'macd_momentum', 'ad_momentum', 
                                 'volume_momentum', 'price_acceleration']
    }
    
    for category, features in categories.items():
        expected_in_category = [f for f in features if f in expected_features]
        actual_in_category = [f for f in features if f in actual_features]
        missing_in_category = [f for f in expected_in_category if f not in actual_features]
        
        print(f"\n   {category}:")
        print(f"     Esperadas: {len(expected_in_category)}")
        print(f"     Calculadas: {len(actual_in_category)}")
        print(f"     Faltantes: {len(missing_in_category)}")
        if missing_in_category:
            print(f"     Faltantes: {missing_in_category}")
    
    # 8. Verificar si TA-Lib está funcionando correctamente
    print(f"\n🔧 VERIFICACIÓN DE TA-LIB:")
    try:
        import talib
        print(f"   ✅ TA-Lib disponible")
        
        # Probar algunos indicadores específicos
        close_prices = test_data['close'].values.astype(float)
        high_prices = test_data['high'].values.astype(float)
        low_prices = test_data['low'].values.astype(float)
        volume_prices = test_data['volume'].values.astype(float)
        
        # Probar indicadores que podrían estar fallando
        test_indicators = {
            'rsi_14': talib.RSI(close_prices, timeperiod=14),
            'macd': talib.MACD(close_prices)[0],
            'stoch_k': talib.STOCH(high_prices, low_prices, close_prices)[0],
            'bb_upper': talib.BBANDS(close_prices)[0],
            'atr_14': talib.ATR(high_prices, low_prices, close_prices, timeperiod=14),
            'ad': talib.AD(high_prices, low_prices, close_prices, volume_prices),
            'mfi_14': talib.MFI(high_prices, low_prices, close_prices, volume_prices, timeperiod=14)
        }
        
        print(f"   ✅ Indicadores TA-Lib funcionando correctamente")
        
    except ImportError:
        print(f"   ❌ TA-Lib no disponible")
    except Exception as e:
        print(f"   ⚠️ Error en TA-Lib: {e}")
    
    # 9. Recomendaciones
    print(f"\n💡 RECOMENDACIONES:")
    print(f"   1. Verificar que todas las features esperadas se calculen correctamente")
    print(f"   2. Asegurar que TA-Lib esté instalado y funcionando en Windows")
    print(f"   3. Revisar las implementaciones manuales para features faltantes")
    print(f"   4. Validar que el entrenamiento use exactamente las mismas features")
    print(f"   5. Considerar usar un conjunto de features más pequeño y estable")
    
    return {
        'expected_count': len(expected_features),
        'actual_count': len(actual_features),
        'missing_features': list(missing_features),
        'extra_features': list(extra_features)
    }

if __name__ == "__main__":
    result = diagnose_features_mismatch()
    print(f"\n🎯 RESUMEN:")
    print(f"   Features esperadas: {result['expected_count']}")
    print(f"   Features calculadas: {result['actual_count']}")
    print(f"   Features faltantes: {len(result['missing_features'])}")
    print(f"   Features extra: {len(result['extra_features'])}") 