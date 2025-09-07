#!/usr/bin/env python3
"""
🧪 TEST: VERIFICACIÓN DE PESOS OPTIMIZADOS UNIFORMES
Verifica que todos los pares tengan los mismos pesos optimizados
"""

import sys
import os

# Agregar el directorio raíz al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from predictor1m_talib import ProbabilisticPredictorTalib, SUPPORTED_PAIRS
    print("✅ Módulo predictor1m_talib importado correctamente")
except ImportError as e:
    print(f"❌ Error importando módulo: {e}")
    sys.exit(1)

def test_weights_uniformity():
    """🧪 Test: Verificar que todos los pares tengan pesos uniformes"""
    print("\n🧪 TEST: VERIFICACIÓN DE PESOS UNIFORMES")
    print("=" * 60)
    
    try:
        # Obtener pesos para todos los pares
        weights = ProbabilisticPredictorTalib.get_optimized_weights()
        
        if not weights:
            print("❌ Error: No se pudieron obtener los pesos")
            return False
        
        print(f"✅ Pesos obtenidos para {len(weights)} pares")
        
        # Verificar que todos los pares soportados estén incluidos
        missing_pairs = set(SUPPORTED_PAIRS) - set(weights.keys())
        if missing_pairs:
            print(f"❌ Pares faltantes: {missing_pairs}")
            return False
        
        print("✅ Todos los pares soportados están incluidos")
        
        # Verificar uniformidad de pesos
        reference_pair = SUPPORTED_PAIRS[0]
        reference_weights = weights[reference_pair]
        
        print(f"\n📊 Par de referencia: {reference_pair}")
        print(f"   Pesos: {reference_weights}")
        
        # Comparar con todos los demás pares
        for pair in SUPPORTED_PAIRS[1:]:
            if weights[pair] != reference_weights:
                print(f"❌ {pair} tiene pesos diferentes a {reference_pair}")
                print(f"   {pair}: {weights[pair]}")
                print(f"   {reference_pair}: {reference_weights}")
                return False
            else:
                print(f"✅ {pair}: Pesos idénticos a {reference_pair}")
        
        print("\n🎯 VERIFICACIÓN DE DISTRIBUCIÓN:")
        total_weight = sum(reference_weights.values())
        print(f"   Peso total: {total_weight:.3f}")
        
        if abs(total_weight - 1.0) < 0.001:
            print("✅ Peso total = 1.0 (correcto)")
        else:
            print(f"⚠️ Peso total = {total_weight:.3f} (debería ser 1.0)")
        
        # Mostrar distribución por grupos
        volume_group = {k: v for k, v in reference_weights.items() if k in ['volume_ratio', 'volume_delta', 'vwap']}
        momentum_group = {k: v for k, v in reference_weights.items() if k in ['heikin_ashi', 'williams_r', 'macd', 'stochastic', 'rsi_14']}
        volatility_group = {k: v for k, v in reference_weights.items() if k in ['bollinger', 'atr', 'pivots']}
        secondary_group = {k: v for k, v in reference_weights.items() if k in ['cci', 'mfi', 'roc']}
        
        print(f"\n📊 Distribución por grupos:")
        print(f"   💧 Volumen y Presión: {sum(volume_group.values()):.3f} ({sum(volume_group.values())*100:.1f}%)")
        print(f"   📈 Tendencia y Momentum: {sum(momentum_group.values()):.3f} ({sum(momentum_group.values())*100:.1f}%)")
        print(f"   📊 Volatilidad y Niveles: {sum(volatility_group.values()):.3f} ({sum(volatility_group.values())*100:.1f}%)")
        print(f"   🔧 Indicadores Secundarios: {sum(secondary_group.values()):.3f} ({sum(secondary_group.values())*100:.1f}%)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error durante la verificación: {e}")
        return False

def test_weight_consistency():
    """🧪 Test: Verificar consistencia de pesos entre diferentes llamadas"""
    print("\n🧪 TEST: VERIFICACIÓN DE CONSISTENCIA")
    print("=" * 60)
    
    try:
        # Llamar múltiples veces para verificar consistencia
        weights1 = ProbabilisticPredictorTalib.get_optimized_weights()
        weights2 = ProbabilisticPredictorTalib.get_optimized_weights()
        weights3 = ProbabilisticPredictorTalib.get_optimized_weights()
        
        if weights1 == weights2 == weights3:
            print("✅ Pesos consistentes entre múltiples llamadas")
            return True
        else:
            print("❌ Pesos inconsistentes entre llamadas")
            return False
            
    except Exception as e:
        print(f"❌ Error durante verificación de consistencia: {e}")
        return False

def main():
    """🏁 Función principal de testing"""
    print("🚀 TEST: VERIFICACIÓN DE PESOS OPTIMIZADOS UNIFORMES")
    print("=" * 80)
    
    # Mostrar pares soportados
    print(f"📋 Pares soportados: {SUPPORTED_PAIRS}")
    print(f"📊 Total de pares: {len(SUPPORTED_PAIRS)}")
    
    # Ejecutar tests
    test1_passed = test_weights_uniformity()
    test2_passed = test_weight_consistency()
    
    # Mostrar pesos para todos los pares
    print("\n🎯 PESOS OPTIMIZADOS PARA TODOS LOS PARES:")
    print("=" * 80)
    ProbabilisticPredictorTalib.print_weights_for_all_pairs()
    
    # Resumen de resultados
    print("\n📊 RESUMEN DE TESTS:")
    print("=" * 40)
    print(f"   🧪 Test Uniformidad: {'✅ PASÓ' if test1_passed else '❌ FALLÓ'}")
    print(f"   🧪 Test Consistencia: {'✅ PASÓ' if test2_passed else '❌ FALLÓ'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 TODOS LOS TESTS PASARON EXITOSAMENTE!")
        print("✅ Los pesos optimizados se aplican uniformemente a todos los pares")
        return True
    else:
        print("\n❌ ALGUNOS TESTS FALLARON")
        print("🔧 Revisar la implementación de pesos")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
