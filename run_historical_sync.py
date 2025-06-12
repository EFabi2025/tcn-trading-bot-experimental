#!/usr/bin/env python3
"""
🚀 EJECUTOR DE SINCRONIZACIÓN HISTÓRICA
Script para verificar configuración y ejecutar sincronización de datos de Binance
"""

import asyncio
import os
from dotenv import load_dotenv
from binance_historical_sync import BinanceHistoricalSync, BinanceConfig

def check_environment():
    """🔍 Verificar configuración del entorno"""
    print("🔍 VERIFICANDO CONFIGURACIÓN DEL ENTORNO")
    print("=" * 50)
    
    # Cargar variables de entorno
    load_dotenv()
    
    api_key = os.getenv('BINANCE_API_KEY', '')
    api_secret = os.getenv('BINANCE_API_SECRET', '')
    
    print(f"🔑 API Key: {'✅ Configurada' if api_key else '❌ No encontrada'}")
    print(f"🔐 API Secret: {'✅ Configurada' if api_secret else '❌ No encontrada'}")
    
    if api_key:
        print(f"   📝 API Key (primeros 8 chars): {api_key[:8]}...")
    
    if not api_key or not api_secret:
        print("\n❌ CONFIGURACIÓN INCOMPLETA")
        print("📋 Para configurar tus API keys:")
        print("   1. Crea un archivo .env en el directorio del proyecto:")
        print("      touch .env")
        print("")
        print("   2. Añade tus credenciales de Binance:")
        print("      echo 'BINANCE_API_KEY=tu_api_key_aqui' >> .env")
        print("      echo 'BINANCE_API_SECRET=tu_api_secret_aqui' >> .env")
        print("")
        print("   3. O configura variables de entorno:")
        print("      export BINANCE_API_KEY='tu_api_key_aqui'")
        print("      export BINANCE_API_SECRET='tu_api_secret_aqui'")
        print("")
        print("🔗 Para obtener API keys de Binance:")
        print("   • Testnet: https://testnet.binance.vision/")
        print("   • Producción: https://www.binance.com/en/my/settings/api-management")
        return False
    
    print("✅ Configuración completa!")
    return True

async def run_sync_with_options():
    """⚙️ Ejecutar sincronización con opciones personalizables"""
    print("\n⚙️ CONFIGURACIÓN DE SINCRONIZACIÓN")
    print("=" * 50)
    
    # Opciones de configuración
    print("🎯 Opciones disponibles:")
    print("   1. Entorno de trading")
    print("   2. Símbolos a sincronizar") 
    print("   3. Días históricos")
    
    # Configuración del entorno
    print(f"\n📡 ENTORNO DE TRADING:")
    print(f"   🧪 Testnet (recomendado para pruebas)")
    print(f"   🔴 Producción (dinero real)")
    
    use_testnet = True  # Por defecto testnet por seguridad
    print(f"   ✅ Seleccionado: {'Testnet' if use_testnet else 'Producción'}")
    
    # Símbolos a sincronizar
    default_symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
    print(f"\n💱 SÍMBOLOS A SINCRONIZAR:")
    print(f"   📊 Símbolos: {', '.join(default_symbols)}")
    
    # Días históricos
    days_back = 30
    print(f"\n📅 PERÍODO HISTÓRICO:")
    print(f"   🗓️  Días atrás: {days_back}")
    
    # Configurar y ejecutar
    config = BinanceConfig(testnet=use_testnet)
    
    print(f"\n🚀 INICIANDO SINCRONIZACIÓN...")
    print(f"   📡 URL: {config.base_url}")
    print(f"   💱 Símbolos: {len(default_symbols)}")
    print(f"   📅 Período: {days_back} días")
    
    async with BinanceHistoricalSync(config) as sync:
        results = await sync.full_sync(default_symbols, days_back)
        
        # Mostrar resultados
        print(f"\n📊 RESULTADOS DE SINCRONIZACIÓN")
        print("=" * 50)
        
        if results['success']:
            print(f"🎉 SINCRONIZACIÓN EXITOSA!")
            print(f"   💰 Balance sincronizado: {'✅' if results['balance_synced'] else '❌'}")
            print(f"   📈 Total trades sincronizados: {results['total_trades_synced']}")
            print(f"   💹 PnL total calculado: ${results['total_pnl']:.2f}")
            
            if results['symbols_synced']:
                print(f"\n📊 Detalle por símbolo:")
                for symbol_data in results['symbols_synced']:
                    print(f"   💱 {symbol_data['symbol']}: {symbol_data['synced_count']}/{symbol_data['trades_count']} trades")
        else:
            print(f"❌ SINCRONIZACIÓN CON ERRORES:")
            for error in results['errors']:
                print(f"   ❌ {error}")
        
        return results

async def main():
    """🚀 Función principal"""
    print("🔄 BINANCE HISTORICAL SYNC - EJECUTOR")
    print("=" * 60)
    
    # 1. Verificar configuración
    if not check_environment():
        return
    
    # 2. Mostrar advertencias de seguridad
    print(f"\n⚠️  ADVERTENCIAS DE SEGURIDAD:")
    print(f"   🧪 Por defecto usa TESTNET (red de pruebas)")
    print(f"   🔒 API keys nunca se muestran completas en logs")
    print(f"   💾 Los datos se guardan en trading_bot.db local")
    print(f"   🔄 La sincronización puede tomar varios minutos")
    
    input(f"\n📝 Presiona ENTER para continuar...")
    
    # 3. Ejecutar sincronización
    await run_sync_with_options()
    
    print(f"\n✅ Proceso completado!")
    print(f"📂 Revisa trading_bot.db para ver los datos sincronizados")

if __name__ == "__main__":
    asyncio.run(main()) 