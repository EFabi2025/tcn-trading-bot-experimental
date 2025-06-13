#!/usr/bin/env python3
"""
🚨 TEST EMERGENCY TCN INTEGRATION
Verificar que el sistema use los modelos de emergencia con 66 features
"""

import asyncio
from simple_professional_manager import SimpleProfessionalTradingManager

async def test_emergency_tcn_integration():
    """Test de integración de modelos TCN de emergencia"""
    print("🚨 TESTING EMERGENCY TCN INTEGRATION")
    print("="*50)

    # Inicializar manager
    manager = SimpleProfessionalTradingManager()
    await manager.initialize()

    print(f"💰 Balance: ${manager.current_balance:.2f}")

    # Obtener precios actuales
    print("\n📊 Obteniendo precios actuales...")
    prices = await manager._get_current_prices()
    print(f"   Precios obtenidos: {len(prices)} símbolos")

    for symbol, price in prices.items():
        print(f"   {symbol}: ${price:,.2f}")

    # Generar señales TCN con modelos de emergencia
    print("\n🧠 Generando señales TCN de emergencia...")
    signals = await manager._generate_tcn_signals(prices)

    print(f"\n🎯 RESULTADOS:")
    print(f"   Señales generadas: {len(signals)}")

    if signals:
        for symbol, signal_data in signals.items():
            print(f"\n   📈 {symbol}:")
            print(f"      Señal: {signal_data['signal']}")
            print(f"      Confianza: {signal_data['confidence']:.1%}")
            print(f"      Features usados: {signal_data['features_used']}")
            print(f"      Precio actual: ${signal_data['current_price']:,.2f}")

            # Mostrar probabilidades
            probs = signal_data['probabilities']
            print(f"      Probabilidades:")
            print(f"        SELL: {probs['SELL']:.3f}")
            print(f"        HOLD: {probs['HOLD']:.3f}")
            print(f"        BUY:  {probs['BUY']:.3f}")
    else:
        print("   ⚠️ No se generaron señales (correcto en mercado bajista)")

    print("\n✅ Test completado")

if __name__ == "__main__":
    asyncio.run(test_emergency_tcn_integration())
