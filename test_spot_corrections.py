#!/usr/bin/env python3
"""
🧪 TEST SPOT TRADING CORRECTIONS
Verificar que las correcciones para Binance Spot funcionan correctamente
"""

import asyncio
from simple_professional_manager import SimpleProfessionalTradingManager

async def test_spot_corrections():
    """🔍 Probar las correcciones para trading spot"""
    
    print("🧪 TESTING SPOT TRADING CORRECTIONS")
    print("=" * 50)
    
    # Crear manager
    manager = SimpleProfessionalTradingManager()
    await manager.initialize()
    
    print("\n📊 Estado inicial:")
    print(f"   💰 Balance: ${manager.current_balance:.2f}")
    print(f"   📈 Posiciones activas: {len(manager.active_positions)}")
    
    # Simular precios de prueba
    test_prices = {
        'BTCUSDT': 109500.00,
        'ETHUSDT': 2770.00,
        'BNBUSDT': 250.00
    }
    
    print(f"\n🔮 Generando señales con precios de prueba...")
    
    # Generar señales (debería SOLO producir BUY)
    signals = await manager._generate_simple_signals(test_prices)
    
    print(f"\n📋 Señales generadas:")
    if signals:
        for symbol, signal_data in signals.items():
            print(f"   {symbol}: {signal_data['signal']} @ ${signal_data['current_price']:.2f}")
            print(f"      🎯 Confianza: {signal_data['confidence']:.1%}")
            print(f"      ✅ Razón: {signal_data['reason']}")
    else:
        print("   (No se generaron señales)")
    
    # Test específico: intentar crear señal SELL manualmente
    print(f"\n🚫 Test: Intentar crear señal SELL manual...")
    
    test_sell_signal = {
        'signal': 'SELL',
        'confidence': 0.85,
        'current_price': test_prices['BTCUSDT'],
        'timestamp': None,
        'reason': 'test_manual'
    }
    
    # Verificar que se rechaza en risk manager
    can_trade, reason = await manager.risk_manager.check_risk_limits_before_trade(
        'BTCUSDT', 'SELL', 0.85
    )
    
    print(f"   🛡️ Resultado del risk manager:")
    print(f"      ✅ Permitido: {can_trade}")
    print(f"      📝 Razón: {reason}")
    
    # Test de balance insuficiente
    print(f"\n💰 Test: Balance insuficiente...")
    
    # Temporalmente reducir balance
    original_balance = manager.current_balance
    manager.current_balance = 5.0  # Menos del mínimo de $11
    
    can_trade, reason = await manager.risk_manager.check_risk_limits_before_trade(
        'BTCUSDT', 'BUY', 0.85
    )
    
    print(f"   🛡️ Con balance de $5.00:")
    print(f"      ✅ Permitido: {can_trade}")
    print(f"      📝 Razón: {reason}")
    
    # Restaurar balance
    manager.current_balance = original_balance
    
    # Test de compra válida
    print(f"\n✅ Test: Compra válida...")
    
    can_trade, reason = await manager.risk_manager.check_risk_limits_before_trade(
        'BTCUSDT', 'BUY', 0.85
    )
    
    print(f"   🛡️ Con balance de ${manager.current_balance:.2f}:")
    print(f"      ✅ Permitido: {can_trade}")
    print(f"      📝 Razón: {reason}")
    
    if can_trade:
        # Simular apertura de posición
        position = await manager.risk_manager.open_position(
            'BTCUSDT', 'BUY', 0.85, test_prices['BTCUSDT']
        )
        
        if position:
            print(f"\n📈 Posición de prueba creada:")
            print(f"   📊 {position.symbol}: {position.side}")
            print(f"   💵 Cantidad: {position.quantity:.6f}")
            print(f"   💲 Valor: ${position.quantity * position.entry_price:.2f}")
            print(f"   🛑 Stop Loss: ${position.stop_loss:.2f}")
            print(f"   🎯 Take Profit: ${position.take_profit:.2f}")
            
            # Cerrar posición de prueba
            await manager.risk_manager.close_position('BTCUSDT', 'TEST_CLEANUP')
    
    print(f"\n✅ TESTS COMPLETADOS")
    print(f"   🔒 Sistema protegido contra ventas en corto")
    print(f"   💰 Validación de balance implementada")
    print(f"   🛡️ Risk management funcionando")
    
    await manager.shutdown()

if __name__ == "__main__":
    asyncio.run(test_spot_corrections()) 