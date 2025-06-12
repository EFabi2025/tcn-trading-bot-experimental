#!/usr/bin/env python3
"""
🚀 RUN TRADING MANAGER
Script principal para ejecutar el Professional Trading Manager
"""

import asyncio
import signal
import sys
from simple_professional_manager import SimpleProfessionalTradingManager

# Variable global para control del manager
trading_manager = None

def signal_handler(sig, frame):
    """🛑 Manejar señales del sistema para apagado seguro"""
    print(f"\n🛑 Señal recibida: {sig}")
    if trading_manager:
        print("🔄 Iniciando apagado seguro...")
        asyncio.create_task(trading_manager.shutdown())
    sys.exit(0)

async def main():
    """🎯 Función principal"""
    global trading_manager

    print("🚀 PROFESSIONAL TRADING MANAGER")
    print("=" * 50)
    print("💰 Balance: 102 USDT")
    print("🎯 Mínimo Binance: 11 USDT")
    print("📊 Posiciones máx: 2 simultáneas")
    print("=" * 50)

    try:
        # Configurar manejadores de señales
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Crear e inicializar trading manager
        trading_manager = SimpleProfessionalTradingManager()
        await trading_manager.initialize()

        # Mostrar estado inicial
        status = await trading_manager.get_system_status()
        print(f"\n📊 Estado del sistema:")
        print(f"   🔧 Estado: {status['status']}")
        print(f"   🌐 Entorno: {status['environment']}")
        print(f"   📈 Símbolos: {status['symbols_trading']}")
        print(f"   ⏱️ Intervalo: {status['check_interval']}s")
        print(f"   💰 Balance USDT: ${status['current_balance_usdt']:.2f}")
        print(f"   💼 Balance total: ${status['total_balance']:.2f}")

        # Mostrar precios iniciales si están disponibles
        if status.get('current_prices'):
            print(f"   💲 Precios actuales:")
            for symbol, price in status['current_prices'].items():
                print(f"      {symbol}: ${price:.4f}")

        # Mostrar información de cuenta si está disponible
        account_info = status.get('account_info')
        if account_info and account_info.get('other_balances'):
            print(f"   🪙 Otros activos:")
            for asset, balance in account_info['other_balances'].items():
                print(f"      {asset}: {balance['total']:.6f}")

        print(f"   📊 Métricas iniciales:")
        print(f"      🔧 API calls: {status['metrics'].get('api_calls_count', 0)}")
        print(f"      📈 Balance updates: {status['metrics'].get('balance_updates', 0)}")
        if status.get('last_balance_update'):
            print(f"      🕐 Último update: {status['last_balance_update']}")

        print(f"   🛡️ Configuración de riesgo:")
        print(f"      📊 Max posición: 15.0% (${trading_manager.current_balance * 0.15:.2f})")
        print(f"      🚨 Max pérdida diaria: 10.0%")
        print(f"      🛑 Stop Loss: 3.0%")
        print(f"      🎯 Take Profit: 6.0%")

        print(f"\n🎯 Iniciando trading automático...")
        print(f"⏸️ Presiona Ctrl+C para pausar/detener")
        print("-" * 30)

        # Ejecutar loop principal de trading
        await trading_manager.run()

    except KeyboardInterrupt:
        print("\n⏹️ Interrupción manual detectada")
        if trading_manager:
            await trading_manager.shutdown()
    except Exception as e:
        print(f"❌ Error crítico: {e}")
        if trading_manager:
            await trading_manager.emergency_stop()
            await trading_manager.shutdown()

if __name__ == "__main__":
    print("🚀 Iniciando Professional Trading Manager...")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Trading Manager detenido por el usuario")
    except Exception as e:
        print(f"\n❌ Error fatal: {e}")
        sys.exit(1)
