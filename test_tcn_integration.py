#!/usr/bin/env python3
"""
🧪 TEST DE INTEGRACIÓN TCN
Verificar que el sistema usa el modelo TCN real en lugar de señales aleatorias
"""

import asyncio
import sys
import os
from datetime import datetime

# Agregar el directorio actual al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from simple_professional_manager import SimpleProfessionalTradingManager

async def test_tcn_integration():
    """🧪 Probar integración del modelo TCN"""
    print("🧪 INICIANDO TEST DE INTEGRACIÓN TCN")
    print("="*60)

    # Crear manager
    manager = SimpleProfessionalTradingManager()

    try:
        # Inicializar (sin conectar a Binance real)
        print("🔧 Inicializando componentes...")

        # Inicializar risk manager para el test
        from advanced_risk_manager import AdvancedRiskManager

        # Crear configuración mock para Binance
        class MockBinanceConfig:
            def __init__(self):
                self.api_key = os.getenv('BINANCE_API_KEY', 'mock_key')
                self.secret_key = os.getenv('BINANCE_SECRET_KEY', 'mock_secret')
                self.base_url = "https://testnet.binance.vision"

        mock_config = MockBinanceConfig()
        manager.risk_manager = AdvancedRiskManager(mock_config)

        # Simular balance para testing
        manager.current_balance = 100.0

        # Crear precios de prueba
        test_prices = {
            "BTCUSDT": 45000.0,
            "ETHUSDT": 3000.0,
            "BNBUSDT": 400.0
        }

        print(f"💰 Balance simulado: ${manager.current_balance:.2f}")
        print(f"📊 Precios de prueba: {test_prices}")
        print()

        # Probar generación de señales TCN
        print("🧠 PROBANDO GENERACIÓN DE SEÑALES TCN...")
        print("-" * 40)

        signals = await manager._generate_tcn_signals(test_prices)

        print()
        print("📊 RESULTADOS DEL TEST:")
        print("-" * 40)

        if signals:
            print(f"✅ Se generaron {len(signals)} señales TCN")
            for symbol, signal_data in signals.items():
                print(f"🎯 {symbol}:")
                print(f"   📈 Señal: {signal_data['signal']}")
                print(f"   🎯 Confianza: {signal_data['confidence']:.1%}")
                print(f"   💰 Precio: ${signal_data['current_price']:,.2f}")
                print(f"   🔧 Razón: {signal_data['reason']}")
                print(f"   📊 Features usadas: {signal_data.get('features_used', 'N/A')}")
                print(f"   ⚡ Boost técnico: {signal_data.get('technical_boost', 1.0):.2f}x")
                print()
        else:
            print("📊 No se generaron señales en este test")
            print("   Esto puede ser normal si:")
            print("   - No hay señales BUY con confianza >70%")
            print("   - El modelo predice HOLD o SELL")
            print("   - No hay datos suficientes")

        # Verificar que NO hay código aleatorio
        print("🔍 VERIFICANDO AUSENCIA DE CÓDIGO ALEATORIO...")
        print("-" * 40)

        import inspect
        source_code = inspect.getsource(manager._generate_tcn_signals)

        if "random" in source_code.lower():
            print("❌ FALLO: Aún hay código aleatorio en _generate_tcn_signals")
            return False
        else:
            print("✅ ÉXITO: No se detectó código aleatorio")

        # Verificar que usa modelo TCN
        if "tcn_predictor" in source_code.lower():
            print("✅ ÉXITO: Usa predictor TCN")
        else:
            print("❌ FALLO: No se detectó uso del predictor TCN")
            return False

        # Verificar filtros de confianza
        if "confidence < 0.70" in source_code:
            print("✅ ÉXITO: Filtro de confianza 70% implementado")
        else:
            print("⚠️ ADVERTENCIA: No se detectó filtro de confianza 70%")

        # Verificar filtro de señales BUY only
        if "signal != 'BUY'" in source_code:
            print("✅ ÉXITO: Filtro BUY-only para Spot implementado")
        else:
            print("❌ FALLO: No se detectó filtro BUY-only")
            return False

        print()
        print("🎉 TEST DE INTEGRACIÓN TCN COMPLETADO")
        print("="*60)

        return True

    except Exception as e:
        print(f"❌ Error en test: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """🚀 Función principal"""
    success = await test_tcn_integration()

    if success:
        print("✅ TODOS LOS TESTS PASARON")
        print("🎯 El sistema ahora usa el modelo TCN real")
        print("🚫 Se eliminó la generación aleatoria de señales")
    else:
        print("❌ ALGUNOS TESTS FALLARON")
        print("⚠️ Revisar la implementación")

    return success

if __name__ == "__main__":
    result = asyncio.run(main())
    sys.exit(0 if result else 1)
