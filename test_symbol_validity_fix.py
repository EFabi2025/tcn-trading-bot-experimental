#!/usr/bin/env python3
"""
🧪 TEST: Símbolos Inválidos Fix
================================

Script de prueba para verificar que las mejoras en el manejo de símbolos inválidos
funcionen correctamente en el Professional Portfolio Manager.
"""

import asyncio
import os
from dotenv import load_dotenv
from professional_portfolio_manager import ProfessionalPortfolioManager

load_dotenv()

async def test_symbol_validity_fix():
    """🧪 Probar el fix de símbolos inválidos"""
    try:
        print("🧪 Iniciando prueba de fix de símbolos inválidos...")
        
        # Configuración
        api_key = os.getenv('BINANCE_API_KEY')
        secret_key = os.getenv('BINANCE_SECRET_KEY')
        base_url = os.getenv('BINANCE_BASE_URL', 'https://api.binance.com')
        
        if not api_key or not secret_key:
            print("❌ Error: Faltan credenciales de Binance en .env")
            return
        
        print(f"🔑 API Key configurada: {api_key[:10]}...")
        print(f"🌐 Base URL: {base_url}")
        
        # Crear instancia del portfolio manager
        portfolio_manager = ProfessionalPortfolioManager(
            api_key=api_key,
            secret_key=secret_key,
            base_url=base_url
        )
        
        # Inicializar
        print("\n🚀 Inicializando Portfolio Manager...")
        await portfolio_manager.initialize()
        
        # Probar verificación de símbolos válidos
        print("\n🔍 Probando verificación de símbolos válidos...")
        
        # Símbolos que deberían ser válidos
        valid_symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
        for symbol in valid_symbols:
            is_valid = await portfolio_manager._is_valid_symbol(symbol)
            print(f"   {symbol}: {'✅ Válido' if is_valid else '❌ Inválido'}")
        
        # Símbolos que probablemente sean inválidos
        invalid_symbols = ['INVALIDUSDT', 'FAKESYMBOLUSDT']
        for symbol in invalid_symbols:
            is_valid = await portfolio_manager._is_valid_symbol(symbol)
            print(f"   {symbol}: {'✅ Válido' if is_valid else '❌ Inválido'}")
        
        # Probar obtención de balances
        print("\n💰 Probando obtención de balances...")
        balances = await portfolio_manager.get_account_balances()
        print(f"   📊 Activos con balance: {len(balances)}")
        for asset, balance_info in balances.items():
            print(f"      {asset}: {balance_info['total']:.8f}")
        
        # Probar obtención de historial de órdenes
        print("\n📋 Probando obtención de historial de órdenes...")
        try:
            orders = await portfolio_manager.get_order_history(days_back=1)
            print(f"   📄 Órdenes encontradas: {len(orders)}")
        except Exception as e:
            print(f"   ⚠️ Error obteniendo historial: {e}")
        
        # Probar obtención de snapshot
        print("\n📊 Probando obtención de snapshot...")
        try:
            snapshot = await portfolio_manager.get_portfolio_snapshot()
            print(f"   📈 Snapshot obtenido exitosamente")
            print(f"      💰 Balance total: ${snapshot.total_balance_usd:.2f}")
            print(f"      📊 Posiciones activas: {snapshot.position_count}")
        except Exception as e:
            print(f"   ⚠️ Error obteniendo snapshot: {e}")
        
        # Mostrar estado del cache
        print(f"\n🗂️ Estado del cache de símbolos válidos:")
        print(f"   📝 Símbolos en cache: {len(portfolio_manager._symbol_validity_cache)}")
        print(f"   🔄 Última actualización: {portfolio_manager._last_cache_refresh}")
        
        print("\n✅ Prueba completada exitosamente!")
        
    except Exception as e:
        print(f"❌ Error en la prueba: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_symbol_validity_fix())
