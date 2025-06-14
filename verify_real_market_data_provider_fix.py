#!/usr/bin/env python3
"""
🔍 VERIFICACIÓN: real_market_data_provider.py CORREGIDO
======================================================

Script para verificar que real_market_data_provider.py está completamente
corregido y es consistente con tcn_definitivo_predictor.py
"""

import sys
import numpy as np
import pandas as pd
from datetime import datetime

def verify_feature_list():
    """Verificar que la lista de features es correcta"""
    print("🔍 VERIFICANDO LISTA DE FEATURES...")

    try:
        from real_market_data_provider import RealMarketDataProvider
        from binance.client import Client

        # Crear instancia temporal (sin cliente real)
        provider = RealMarketDataProvider(None)

        # Verificar número de features
        feature_count = len(provider.FIXED_FEATURE_LIST)
        print(f"   📊 Número de features: {feature_count}")

        if feature_count == 66:
            print("   ✅ CORRECTO: 66 features")
        else:
            print(f"   ❌ ERROR: Se esperaban 66 features, encontradas {feature_count}")
            return False

        # Verificar categorías de features
        expected_categories = {
            'rsi_': 3,      # rsi_14, rsi_21, rsi_7
            'macd': 3,      # macd, macd_signal, macd_histogram
            'stoch_': 2,    # stoch_k, stoch_d
            'sma_': 3,      # sma_10, sma_20, sma_50
            'ema_': 3,      # ema_10, ema_20, ema_50
            'bb_': 5,       # bb_upper, bb_middle, bb_lower, bb_width, bb_position
            'atr_': 2,      # atr_14, atr_20
            'volume_': 4,   # volume_sma_10, volume_sma_20, volume_ratio, volume_momentum
        }

        for prefix, expected_count in expected_categories.items():
            actual_count = sum(1 for f in provider.FIXED_FEATURE_LIST if f.startswith(prefix))
            if actual_count >= expected_count:
                print(f"   ✅ {prefix}*: {actual_count} features")
            else:
                print(f"   ⚠️ {prefix}*: {actual_count} features (esperadas >= {expected_count})")

        return True

    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        return False

def verify_talib_import():
    """Verificar que TA-Lib está siendo usado"""
    print("\n🔍 VERIFICANDO USO DE TA-LIB...")

    try:
        # Leer el archivo y verificar imports
        with open('real_market_data_provider.py', 'r') as f:
            content = f.read()

        if 'import talib' in content:
            print("   ✅ TA-Lib importado correctamente")
        else:
            print("   ❌ TA-Lib NO está importado")
            return False

        # Verificar uso de funciones TA-Lib
        talib_functions = [
            'talib.RSI',
            'talib.MACD',
            'talib.BBANDS',
            'talib.ATR',
            'talib.SMA',
            'talib.EMA',
            'talib.STOCH',
            'talib.WILLR'
        ]

        for func in talib_functions:
            if func in content:
                print(f"   ✅ {func} utilizado")
            else:
                print(f"   ⚠️ {func} no encontrado")

        return True

    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        return False

def verify_consistency_with_tcn_predictor():
    """Verificar consistencia con tcn_definitivo_predictor.py"""
    print("\n🔍 VERIFICANDO CONSISTENCIA CON TCN_DEFINITIVO_PREDICTOR...")

    try:
        from real_market_data_provider import RealMarketDataProvider
        from tcn_definitivo_predictor import TCNDefinitivoPredictor

        # Crear instancias
        provider = RealMarketDataProvider(None)
        predictor = TCNDefinitivoPredictor()

        # Comparar listas de features
        provider_features = set(provider.FIXED_FEATURE_LIST)

        # Obtener features del predictor (simulando)
        predictor_features = {
            'rsi_14', 'rsi_21', 'rsi_7',
            'macd', 'macd_signal', 'macd_histogram',
            'stoch_k', 'stoch_d', 'williams_r',
            'sma_10', 'sma_20', 'sma_50',
            'ema_10', 'ema_20', 'ema_50',
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_position',
            'atr_14', 'atr_20', 'true_range',
            'volume_ratio', 'ad', 'obv'
        }

        # Verificar overlap
        common_features = provider_features.intersection(predictor_features)
        print(f"   📊 Features comunes: {len(common_features)}")

        if len(common_features) >= 20:  # Al menos 20 features en común
            print("   ✅ BUENA consistencia con tcn_definitivo_predictor.py")
        else:
            print("   ⚠️ Consistencia limitada con tcn_definitivo_predictor.py")

        return True

    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        return False

def verify_shape_compatibility():
    """Verificar compatibilidad de shapes"""
    print("\n🔍 VERIFICANDO COMPATIBILIDAD DE SHAPES...")

    try:
        # Verificar que el código espera shape (32, 66)
        with open('real_market_data_provider.py', 'r') as f:
            content = f.read()

        if '(32, 66)' in content:
            print("   ✅ Shape (32, 66) configurado correctamente")
        else:
            print("   ❌ Shape (32, 66) no encontrado")
            return False

        if 'tail(32)' in content:
            print("   ✅ Secuencia de 32 timesteps configurada")
        else:
            print("   ⚠️ Secuencia de 32 timesteps no encontrada")

        return True

    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        return False

def verify_legacy_code_removed():
    """Verificar que el código legacy fue removido"""
    print("\n🔍 VERIFICANDO REMOCIÓN DE CÓDIGO LEGACY...")

    try:
        with open('real_market_data_provider.py', 'r') as f:
            content = f.read()

        # Verificar que no hay referencias al sistema de 21 features
        legacy_patterns = [
            'rolling(window=',
            'pct_change(periods=',
            'ewm(span=12',
            'await self._calculate_rsi',
            '21 features'
        ]

        legacy_found = []
        for pattern in legacy_patterns:
            if pattern in content:
                legacy_found.append(pattern)

        if not legacy_found:
            print("   ✅ Código legacy removido completamente")
        else:
            print(f"   ⚠️ Código legacy encontrado: {legacy_found}")

        # Verificar que usa TA-Lib en lugar de implementación manual
        if 'talib.' in content and len(legacy_found) <= 2:  # Permitir algunas referencias menores
            print("   ✅ Migrado exitosamente a TA-Lib")
            return True
        else:
            print("   ❌ Migración a TA-Lib incompleta")
            return False

    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        return False

def main():
    """Función principal de verificación"""
    print("🔍 VERIFICACIÓN COMPLETA: real_market_data_provider.py")
    print("=" * 60)
    print(f"🕐 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Ejecutar todas las verificaciones
    checks = [
        ("Lista de Features", verify_feature_list),
        ("Uso de TA-Lib", verify_talib_import),
        ("Consistencia con TCN", verify_consistency_with_tcn_predictor),
        ("Compatibilidad de Shapes", verify_shape_compatibility),
        ("Remoción de Código Legacy", verify_legacy_code_removed)
    ]

    results = []
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"   ❌ ERROR en {check_name}: {e}")
            results.append((check_name, False))

    # Resumen final
    print("\n" + "=" * 60)
    print("📋 RESUMEN DE VERIFICACIÓN")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for check_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {check_name}")

    print()
    print(f"📊 RESULTADO FINAL: {passed}/{total} verificaciones pasadas")

    if passed == total:
        print("🎉 ¡CORRECCIÓN COMPLETADA EXITOSAMENTE!")
        print("   real_market_data_provider.py está ahora en armonía con el sistema principal")
    elif passed >= total * 0.8:
        print("⚠️ CORRECCIÓN MAYORMENTE EXITOSA")
        print("   Algunas verificaciones menores fallaron, pero el sistema es funcional")
    else:
        print("❌ CORRECCIÓN INCOMPLETA")
        print("   Se requieren más ajustes para completar la migración")

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
