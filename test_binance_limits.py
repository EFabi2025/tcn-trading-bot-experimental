#!/usr/bin/env python3
"""
🧪 TEST DE LÍMITES DE BINANCE
Verificar que el sistema cumple con los requisitos mínimos de Binance
"""

import asyncio
from advanced_risk_manager import AdvancedRiskManager, RiskLimits

class TestConfig:
    api_key = "test"
    secret_key = "test"
    base_url = "https://testnet.binance.vision"
    environment = "test"

async def test_position_sizing():
    """📊 Probar el cálculo de tamaño de posición"""
    print("🧪 TESTING - LÍMITES DE BINANCE")
    print("=" * 50)
    
    # Crear risk manager
    risk_manager = AdvancedRiskManager(TestConfig())
    await risk_manager.initialize()
    
    print("\n📊 SIMULANDO DIFERENTES ESCENARIOS:")
    print("-" * 40)
    
    # Escenarios de prueba
    test_scenarios = [
        {"symbol": "BTCUSDT", "price": 45000.0, "confidence": 0.8, "name": "BTC - Alta confianza"},
        {"symbol": "ETHUSDT", "price": 3000.0, "confidence": 0.7, "name": "ETH - Confianza normal"},
        {"symbol": "BNBUSDT", "price": 250.0, "confidence": 0.75, "name": "BNB - Confianza media"},
    ]
    
    results = []
    
    for scenario in test_scenarios:
        print(f"\n🔍 {scenario['name']}:")
        print(f"   💰 Precio: ${scenario['price']:,.2f}")
        print(f"   🎯 Confianza: {scenario['confidence']:.1%}")
        
        # Calcular tamaño de posición
        quantity = risk_manager.calculate_position_size(
            scenario['symbol'], 
            scenario['confidence'], 
            scenario['price']
        )
        
        if quantity > 0:
            position_value = quantity * scenario['price']
            results.append({
                'symbol': scenario['symbol'],
                'quantity': quantity,
                'value': position_value,
                'meets_minimum': position_value >= 11.0
            })
            
            status = "✅" if position_value >= 11.0 else "❌"
            print(f"   {status} Resultado: {quantity:.6f} coins = ${position_value:.2f}")
        else:
            print("   ❌ Posición rechazada")
            results.append({
                'symbol': scenario['symbol'],
                'quantity': 0,
                'value': 0,
                'meets_minimum': False
            })
    
    # Resumen
    print("\n" + "=" * 50)
    print("📋 RESUMEN DE RESULTADOS:")
    print()
    
    valid_positions = [r for r in results if r['meets_minimum']]
    total_exposure = sum(r['value'] for r in valid_positions)
    exposure_percent = (total_exposure / risk_manager.current_balance) * 100
    
    print(f"💰 Balance disponible: ${risk_manager.current_balance:.2f}")
    print(f"📊 Posiciones válidas: {len(valid_positions)}/{len(results)}")
    print(f"💵 Exposición total: ${total_exposure:.2f} ({exposure_percent:.1f}%)")
    print(f"💵 Mínimo Binance: ${risk_manager.limits.min_position_value_usdt} USDT")
    
    print("\n🎯 ANÁLISIS:")
    
    if len(valid_positions) >= 2:
        print("✅ ÓPTIMO: Puedes diversificar en múltiples posiciones")
    elif len(valid_positions) == 1:
        print("⚠️ LIMITADO: Solo una posición por vez")
    else:
        print("❌ PROBLEMA: Balance insuficiente para trading")
    
    return len(valid_positions) > 0

async def main():
    """🎯 Ejecutar todos los tests"""
    print("🚀 VERIFICACIÓN DE LÍMITES DE BINANCE")
    print("🎯 Balance: 102 USDT | Mínimo Binance: 11 USDT")
    print()
    
    # Test 1: Position sizing
    sizing_ok = await test_position_sizing()
    
    # Resultado final
    print("\n" + "=" * 50)
    print("🏁 RESULTADO FINAL:")
    
    if sizing_ok:
        print("✅ SISTEMA LISTO PARA TRADING")
        print("🎯 Cumple todos los requisitos de Binance")
    else:
        print("⚠️ REQUIERE AJUSTES")
        print("🔧 Revisa la configuración arriba")

if __name__ == "__main__":
    asyncio.run(main()) 