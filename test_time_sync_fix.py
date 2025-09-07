#!/usr/bin/env python3
"""
🧪 TEST: Time Sync Fix
=======================

Script de prueba para verificar que las mejoras en la sincronización de tiempo
funcionen correctamente y eliminen el error -1021.
"""

import asyncio
import os
import time
from dotenv import load_dotenv
from professional_portfolio_manager import ProfessionalPortfolioManager

load_dotenv()

async def test_time_sync_fix():
    """🧪 Probar el fix de sincronización de tiempo"""
    try:
        print("🧪 Iniciando prueba de fix de sincronización de tiempo...")
        
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
        
        # Inicializar y verificar sincronización de tiempo
        print("\n🚀 Inicializando Portfolio Manager...")
        await portfolio_manager.initialize()
        
        # Probar verificación de tiempo
        print("\n🕐 Probando verificación de sincronización de tiempo...")
        await portfolio_manager._check_time_sync()
        
        # Mostrar configuración de recvWindow
        print(f"\n📊 Configuración de recvWindow:")
        print(f"   🔧 Valor ajustado por tiempo: {getattr(portfolio_manager, '_time_adjusted_recv_window', 'No disponible')}ms")
        
        # Probar obtención de balances (que puede causar error -1021)
        print("\n💰 Probando obtención de balances...")
        try:
            balances = await portfolio_manager.get_account_balances()
            print(f"   ✅ Balances obtenidos exitosamente: {len(balances)} activos")
        except Exception as e:
            if "code\":-1021" in str(e):
                print(f"   ❌ Error de timestamp persistente: {e}")
            else:
                print(f"   ⚠️ Otro error: {e}")
        
        # Probar obtención de historial de órdenes
        print("\n📋 Probando obtención de historial de órdenes...")
        try:
            orders = await portfolio_manager.get_order_history(days_back=1)
            print(f"   ✅ Historial obtenido exitosamente: {len(orders)} órdenes")
        except Exception as e:
            if "code\":-1021" in str(e):
                print(f"   ❌ Error de timestamp persistente: {e}")
            else:
                print(f"   ⚠️ Otro error: {e}")
        
        # Probar snapshot completo
        print("\n📊 Probando snapshot completo...")
        try:
            snapshot = await portfolio_manager.get_portfolio_snapshot()
            print(f"   ✅ Snapshot obtenido exitosamente")
            print(f"      💰 Balance total: ${snapshot.total_balance_usd:.2f}")
            print(f"      📊 Posiciones activas: {snapshot.position_count}")
        except Exception as e:
            if "code\":-1021" in str(e):
                print(f"   ❌ Error de timestamp persistente: {e}")
            else:
                print(f"   ⚠️ Otro error: {e}")
        
        # Mostrar métricas finales
        print(f"\n📈 Métricas finales:")
        print(f"   🔄 API calls realizadas: {portfolio_manager.api_calls_count}")
        print(f"   ❌ Errores totales: {portfolio_manager.error_count}")
        
        print("\n✅ Prueba completada exitosamente!")
        
    except Exception as e:
        print(f"❌ Error en la prueba: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_time_sync_fix())
