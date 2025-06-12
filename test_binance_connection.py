#!/usr/bin/env python3
"""
🧪 Tester para Simple Professional Trading Manager
"""
import asyncio
import os
from datetime import datetime
from dotenv import load_dotenv
from simple_professional_manager import SimpleProfessionalTradingManager

load_dotenv()

async def test_trading_manager():
    """🎯 Test básico del Trading Manager"""
    print("🧪 Iniciando test del Simple Professional Trading Manager...")

    manager = SimpleProfessionalTradingManager()

    try:
        # Inicializar
        await manager.initialize()
        print("✅ Manager inicializado correctamente")

        # Test de obtención de precios
        prices = await manager._get_current_prices()
        print(f"✅ Precios obtenidos: {prices}")

        # Test de generación de señales TCN
        print("\n🧠 Probando generación de señales TCN...")
        signals = await manager._generate_tcn_signals(prices)

        if signals:
            print(f"🎯 Señales generadas: {len(signals)}")
            for symbol, signal_data in signals.items():
                print(f"  📊 {symbol}: {signal_data['signal']} ({signal_data['confidence']:.1%})")
        else:
            print("📊 No se generaron señales en este ciclo")

        # Test de métricas
        await manager._update_metrics()
        status = await manager.get_system_status()
        print(f"\n📈 Estado del sistema: {status['status']}")
        print(f"💰 Balance: ${status['current_balance_usdt']:.2f}")
        print(f"📊 Posiciones activas: {status['active_positions']}")
        print(f"🎯 Trades realizados: {status['trade_count']}")

        print("\n✅ Test completado exitosamente")

    except Exception as e:
        print(f"❌ Error durante el test: {e}")
        raise
    finally:
        await manager.shutdown()
        print("🔄 Manager cerrado correctamente")

async def test_continuous_run(duration_minutes: int = 5):
    """🔄 Test de ejecución continua"""
    print(f"🔄 Iniciando test continuo por {duration_minutes} minutos...")

    manager = SimpleProfessionalTradingManager()

    try:
        await manager.initialize()

        # Ejecutar por tiempo limitado
        start_time = datetime.now()

        while True:
            current_time = datetime.now()
            elapsed = (current_time - start_time).total_seconds() / 60

            if elapsed >= duration_minutes:
                print(f"⏰ Tiempo completado: {duration_minutes} minutos")
                break

            # Una iteración del loop principal
            await manager._display_professional_info()
            await asyncio.sleep(30)  # Esperar 30 segundos entre iteraciones

    except KeyboardInterrupt:
        print("\n⏹️ Test detenido por el usuario")
    except Exception as e:
        print(f"\n❌ Error durante test continuo: {e}")

async def main():
    """🎯 Función principal"""
    print("🎯 Simple Professional Trading Manager - Tester")
    print("=" * 50)

    choice = input("""
Selecciona el tipo de test:
1. Test básico (inicialización y señales)
2. Test continuo (5 minutos)
3. Solo test de conexión Binance

Opción (1-3): """).strip()

    if choice == "1":
        await test_trading_manager()
    elif choice == "2":
        await test_continuous_run(5)
    elif choice == "3":
        from enhanced_real_predictor import AdvancedBinanceData
        async with AdvancedBinanceData() as binance_data:
            market_data = await binance_data.get_comprehensive_data('BTCUSDT')
            if market_data and market_data.get('klines_1m'):
                print(f'✅ Datos obtenidos: {len(market_data["klines_1m"])} velas')
                print(f'✅ Último precio BTC: ${market_data["klines_1m"][-1]["close"]}')
            else:
                print('❌ No se pudieron obtener datos')
    else:
        print("❌ Opción inválida")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Tester detenido por el usuario")
