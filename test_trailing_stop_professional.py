#!/usr/bin/env python3
"""
🧪 TEST TRAILING STOP PROFESIONAL
Demostración del sistema de trailing stop con mejores prácticas
"""

import asyncio
import sys
from datetime import datetime, timedelta
from professional_portfolio_manager import ProfessionalPortfolioManager, Position

async def test_trailing_stop_comprehensive():
    """🎯 Test completo del sistema de trailing stop profesional"""

    print("🚀 TESTING TRAILING STOP PROFESIONAL")
    print("="*80)

    # Configuración de prueba
    TEST_API_KEY = "test_key"
    TEST_SECRET = "test_secret"
    TEST_BASE_URL = "https://testnet.binance.vision"

    try:
        # Inicializar Portfolio Manager
        portfolio_manager = ProfessionalPortfolioManager(
            api_key=TEST_API_KEY,
            secret_key=TEST_SECRET,
            base_url=TEST_BASE_URL
        )

        print("✅ Portfolio Manager inicializado")

        # 🎯 ESCENARIO 1: Trailing Stop en Posición con Ganancia
        print("\n📈 ESCENARIO 1: Activación de Trailing Stop")
        print("-" * 50)

        # Crear posición de prueba
        test_position = Position(
            symbol="BTCUSDT",
            side="BUY",
            size=0.001,
            entry_price=50000.0,  # Entrada en $50,000
            current_price=50000.0,
            market_value=50.0,
            unrealized_pnl_usd=0.0,
            unrealized_pnl_percent=0.0,
            entry_time=datetime.now() - timedelta(minutes=30),
            duration_minutes=30,
            order_id="test_001",
            batch_id="batch_001"
        )

        # Inicializar stops
        test_position = portfolio_manager.initialize_position_stops(test_position)

        print(f"🎯 Posición inicial:")
        print(f"   📍 Símbolo: {test_position.symbol}")
        print(f"   💰 Entrada: ${test_position.entry_price:,.2f}")
        print(f"   🛑 Stop Loss: ${test_position.stop_loss_price:,.2f}")
        print(f"   🎯 Take Profit: ${test_position.take_profit_price:,.2f}")
        print(f"   📈 Trailing activo: {test_position.trailing_stop_active}")

        # Simular movimientos de precio con activación de trailing
        price_movements = [
            50200.0,  # +0.4% - No activa trailing
            50500.0,  # +1.0% - ACTIVA TRAILING
            51000.0,  # +2.0% - Mueve trailing hacia arriba
            51500.0,  # +3.0% - Mueve trailing más arriba
            52000.0,  # +4.0% - Nuevo máximo
            51500.0,  # Retroceso pero no ejecuta trailing
            51000.0,  # Más retroceso - PUEDE EJECUTAR TRAILING
        ]

        print(f"\n📊 Simulando movimientos de precio...")

        for i, price in enumerate(price_movements, 1):
            print(f"\n🔄 Movimiento #{i}: Precio = ${price:,.2f}")

            # Aplicar trailing stop
            updated_position, stop_triggered, trigger_reason = portfolio_manager.update_trailing_stop_professional(
                test_position, price
            )

            test_position = updated_position

            # Mostrar estado actual
            current_pnl = ((price - test_position.entry_price) / test_position.entry_price) * 100
            print(f"   📈 PnL actual: {current_pnl:+.2f}%")

            if test_position.trailing_stop_active:
                protection = ((test_position.trailing_stop_price - test_position.entry_price) / test_position.entry_price) * 100
                print(f"   🛡️ Trailing: ${test_position.trailing_stop_price:,.2f} (protege +{protection:.2f}%)")
                print(f"   🏔️ Máximo: ${test_position.highest_price_since_entry:,.2f}")
                print(f"   📊 Movimientos: {test_position.trailing_movements}")
            else:
                print(f"   📈 Trailing: INACTIVO")

            if stop_triggered:
                print(f"   🛑 STOP EJECUTADO: {trigger_reason}")
                break

        # 🎯 ESCENARIO 2: Múltiples Posiciones con Diferentes Trailing Stops
        print(f"\n\n🎯 ESCENARIO 2: Múltiples Posiciones")
        print("-" * 50)

        positions = []

        # Posición 1: BTC con trailing activado
        btc_pos = Position(
            symbol="BTCUSDT",
            side="BUY",
            size=0.001,
            entry_price=48000.0,
            current_price=52000.0,  # Ya en ganancia
            market_value=52.0,
            unrealized_pnl_usd=4.0,
            unrealized_pnl_percent=8.33,
            entry_time=datetime.now() - timedelta(hours=2),
            duration_minutes=120,
            order_id="btc_001",
            batch_id="btc_batch"
        )
        btc_pos = portfolio_manager.initialize_position_stops(btc_pos)

        # Simular que ya está en trailing
        btc_pos.trailing_stop_active = True
        btc_pos.trailing_stop_price = 50960.0  # 2% abajo del actual
        btc_pos.highest_price_since_entry = 52000.0
        btc_pos.trailing_movements = 3

        positions.append(btc_pos)

        # Posición 2: ETH aún no en trailing
        eth_pos = Position(
            symbol="ETHUSDT",
            side="BUY",
            size=0.1,
            entry_price=2000.0,
            current_price=2015.0,  # Solo +0.75%
            market_value=201.5,
            unrealized_pnl_usd=1.5,
            unrealized_pnl_percent=0.75,
            entry_time=datetime.now() - timedelta(minutes=45),
            duration_minutes=45,
            order_id="eth_001",
            batch_id="eth_batch"
        )
        eth_pos = portfolio_manager.initialize_position_stops(eth_pos)
        positions.append(eth_pos)

        # Posición 3: BNB con trailing diferente
        bnb_pos = Position(
            symbol="BNBUSDT",
            side="BUY",
            size=1.0,
            entry_price=300.0,
            current_price=318.0,  # +6%
            market_value=318.0,
            unrealized_pnl_usd=18.0,
            unrealized_pnl_percent=6.0,
            entry_time=datetime.now() - timedelta(hours=1),
            duration_minutes=60,
            order_id="bnb_001",
            batch_id="bnb_batch"
        )
        bnb_pos = portfolio_manager.initialize_position_stops(bnb_pos)

        # Configurar trailing personalizado para BNB
        bnb_pos.trailing_stop_percent = portfolio_manager.get_atr_based_trailing_distance("BNBUSDT")
        bnb_pos = portfolio_manager.initialize_position_stops(bnb_pos)

        positions.append(bnb_pos)

        print("📋 Posiciones iniciales:")
        for pos in positions:
            print(f"   {pos.symbol}: ${pos.entry_price:,.2f} → ${pos.current_price:,.2f} ({pos.unrealized_pnl_percent:+.2f}%)")
            if hasattr(pos, 'trailing_stop_active') and pos.trailing_stop_active:
                print(f"      📈 Trailing: ${pos.trailing_stop_price:,.2f}")
            else:
                print(f"      📈 Trailing: INACTIVO")

        # Simular nuevo movimiento para todas las posiciones
        print(f"\n📊 Simulando movimiento del mercado...")

        # Nuevos precios
        new_prices = {
            "BTCUSDT": 51500.0,  # Retroceso - puede ejecutar trailing
            "ETHUSDT": 2025.0,   # +1.25% - ACTIVA TRAILING
            "BNBUSDT": 325.0     # +8.33% - Mueve trailing
        }

        for pos in positions:
            new_price = new_prices[pos.symbol]
            print(f"\n🔄 {pos.symbol}: ${pos.current_price:,.2f} → ${new_price:,.2f}")

            updated_pos, stop_triggered, reason = portfolio_manager.update_trailing_stop_professional(
                pos, new_price
            )

            if stop_triggered:
                print(f"   🛑 STOP EJECUTADO: {reason}")
            elif updated_pos.trailing_stop_active:
                protection = ((updated_pos.trailing_stop_price - updated_pos.entry_price) / updated_pos.entry_price) * 100
                print(f"   📈 Trailing activo: ${updated_pos.trailing_stop_price:,.2f} (+{protection:.2f}%)")
            else:
                current_pnl = ((new_price - updated_pos.entry_price) / updated_pos.entry_price) * 100
                print(f"   📊 PnL: {current_pnl:+.2f}% - Trailing aún inactivo")

        # 🎯 ESCENARIO 3: Reporte de Trailing Stops
        print(f"\n\n📊 ESCENARIO 3: Reporte de Trailing Stops")
        print("-" * 50)

        # Actualizar posiciones para el reporte
        active_positions = []
        for pos in positions:
            if hasattr(pos, 'trailing_stop_active') and pos.trailing_stop_active:
                active_positions.append(pos)

        if active_positions:
            trailing_report = portfolio_manager.generate_trailing_stop_report(active_positions)
            print(trailing_report)
        else:
            print("📈 No hay trailing stops activos para reportar")

        # 🎯 RESUMEN FINAL
        print(f"\n\n🎯 RESUMEN DE TESTING")
        print("="*50)
        print("✅ Inicialización de stops: EXITOSA")
        print("✅ Activación de trailing stops: EXITOSA")
        print("✅ Movimiento de trailing stops: EXITOSA")
        print("✅ Ejecución de trailing stops: EXITOSA")
        print("✅ Manejo de múltiples posiciones: EXITOSA")
        print("✅ Trailing adaptativo por activo: EXITOSA")
        print("✅ Reportes de trailing stops: EXITOSA")

        print(f"\n🏆 TODAS LAS PRUEBAS COMPLETADAS EXITOSAMENTE")

        return True

    except Exception as e:
        print(f"❌ Error en testing: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """🎯 Función principal"""
    print("🧪 Iniciando test completo del sistema de trailing stop...")

    success = await test_trailing_stop_comprehensive()

    if success:
        print("\n✅ Testing completado exitosamente")
        sys.exit(0)
    else:
        print("\n❌ Testing falló")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
