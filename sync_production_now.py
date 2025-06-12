#!/usr/bin/env python3
"""
🔴 SINCRONIZACIÓN DIRECTA DE PRODUCCIÓN
Sincroniza datos reales de tu cuenta de Binance
"""

import asyncio
from binance_historical_sync import BinanceHistoricalSync, BinanceConfig

async def sync_production():
    print("🔴 SINCRONIZACIÓN DE CUENTA REAL DE BINANCE")
    print("=" * 50)
    
    # Configurar para producción (no testnet)
    config = BinanceConfig(testnet=False)
    
    print(f"📡 URL: {config.base_url}")
    print(f"🔑 API Key configurada: {'✅' if config.api_key else '❌'}")
    print(f"🔐 Secret Key configurada: {'✅' if config.secret_key else '❌'}")
    
    if not config.api_key or not config.secret_key:
        print("❌ API keys no encontradas. Verifica tu archivo .env")
        return
    
    print(f"🔥 IMPORTANTE: Accediendo a tu cuenta REAL de Binance")
    print(f"📖 Solo lectura - NO se ejecutarán trades")
    
    try:
        async with BinanceHistoricalSync(config) as sync:
            # 1. Probar conexión
            print("\n🔗 Probando conexión...")
            account_info = await sync.get_account_info()
            
            if not account_info:
                print("❌ Error de conexión con Binance")
                return
            
            print("✅ Conexión exitosa con Binance")
            
            # 2. Ejecutar sincronización completa
            print("\n🚀 Iniciando sincronización completa...")
            symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
            days_back = 30
            
            results = await sync.full_sync(symbols, days_back)
            
            # 3. Mostrar resultados
            print(f"\n📊 RESULTADOS:")
            if results["success"]:
                print(f"🎉 ¡SINCRONIZACIÓN EXITOSA!")
                print(f"   💰 Balance sincronizado: {'✅' if results['balance_synced'] else '❌'}")
                print(f"   📈 Trades sincronizados: {results['total_trades_synced']}")
                print(f"   💹 PnL total: ${results['total_pnl']:.2f}")
                
                if results['symbols_synced']:
                    print(f"\n📊 Por símbolo:")
                    for sym in results['symbols_synced']:
                        print(f"   💱 {sym['symbol']}: {sym['synced_count']}/{sym['trades_count']} trades")
                
                print(f"\n✅ TU BASE DE DATOS AHORA TIENE DATOS REALES!")
                print(f"📂 Ejecuta: python verify_sync_results.py")
            else:
                print(f"❌ Errores durante sincronización:")
                for error in results['errors']:
                    print(f"   ❌ {error}")
                    
    except Exception as e:
        print(f"❌ Error durante sincronización: {e}")

if __name__ == "__main__":
    asyncio.run(sync_production()) 