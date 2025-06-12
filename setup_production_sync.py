#!/usr/bin/env python3
"""
🔴 BINANCE PRODUCTION SYNC - Sincronización con Cuenta Real
⚠️  CUIDADO: Este script accede a tu cuenta REAL de Binance

ADVERTENCIAS CRÍTICAS:
- Solo lectura de datos (no trading automático)
- Usa API keys de PRODUCCIÓN (no testnet)
- Maneja datos de tu cuenta real
- Requiere permisos específicos en Binance
"""

import os
import asyncio
import shutil
from pathlib import Path
from binance_historical_sync import BinanceHistoricalSync, BinanceConfig
from dotenv import load_dotenv

def show_production_warnings():
    """🚨 Mostrar advertencias de seguridad para producción"""
    print("🔴 SINCRONIZACIÓN CON CUENTA REAL DE BINANCE")
    print("=" * 60)
    print("⚠️  ADVERTENCIAS CRÍTICAS:")
    print("   🔥 Esto accederá a tu cuenta REAL de Binance")
    print("   💰 Se verán tus balances y trades reales")
    print("   🔐 Asegúrate de que tus API keys sean correctas")
    print("   📖 Solo lectura - NO se ejecutarán trades")
    print("   🔒 Mantén tus credenciales seguras")
    print("")
    print("✅ VERIFICACIONES REQUERIDAS:")
    print("   1. API keys de producción configuradas")
    print("   2. Permisos 'Read Info' habilitados en Binance")
    print("   3. Red estable para evitar timeouts")
    print("   4. Backup de tu base de datos actual")
    print("")

def check_production_config():
    """🔍 Verificar configuración para producción"""
    print("🔍 VERIFICANDO CONFIGURACIÓN DE PRODUCCIÓN")
    print("=" * 50)
    
    # Verificar si existe .env
    env_file = Path(".env")
    env_example = Path("env_example")
    
    if not env_file.exists():
        if env_example.exists():
            print("📋 Creando .env desde env_example...")
            shutil.copy(env_example, env_file)
            print("✅ Archivo .env creado")
        else:
            print("❌ No se encuentra archivo de configuración")
            return False
    
    # Cargar variables
    load_dotenv()
    
    api_key = os.getenv('BINANCE_API_KEY', '').strip('"')
    secret_key = os.getenv('BINANCE_SECRET_KEY', '').strip('"') or os.getenv('BINANCE_API_SECRET', '').strip('"')
    environment = os.getenv('ENVIRONMENT', 'testnet').strip('"')
    
    print(f"🔑 API Key: {'✅ Configurada' if api_key and api_key != 'tu_api_key_de_binance_aqui' else '❌ No configurada'}")
    print(f"🔐 Secret Key: {'✅ Configurada' if secret_key and secret_key != 'tu_secret_key_de_binance_aqui' else '❌ No configurada'}")
    print(f"🌍 Entorno actual: {environment}")
    
    # Verificar que no sean valores por defecto
    if api_key == 'tu_api_key_de_binance_aqui' or not api_key:
        print("\n❌ ERROR: API Key no configurada")
        print("📝 Edita el archivo .env y reemplaza 'tu_api_key_de_binance_aqui' con tu API key real")
        return False
    
    if secret_key == 'tu_secret_key_de_binance_aqui' or not secret_key:
        print("\n❌ ERROR: Secret Key no configurada")
        print("📝 Edita el archivo .env y reemplaza 'tu_secret_key_de_binance_aqui' con tu secret key real")
        return False
    
    if api_key.startswith('tu_') or secret_key.startswith('tu_'):
        print("\n❌ ERROR: Credenciales no actualizadas")
        print("📝 Reemplaza los valores de ejemplo con tus credenciales reales de Binance")
        return False
    
    print("\n✅ Configuración básica correcta")
    return True

def update_env_for_production():
    """⚙️ Actualizar .env para producción"""
    print("\n⚙️ CONFIGURANDO PARA PRODUCCIÓN")
    print("=" * 50)
    
    env_file = Path(".env")
    if not env_file.exists():
        print("❌ Archivo .env no encontrado")
        return False
    
    # Leer contenido actual
    with open(env_file, 'r') as f:
        content = f.read()
    
    # Actualizar configuraciones críticas para producción
    updates = {
        'ENVIRONMENT=testnet': 'ENVIRONMENT=production',
        'TRADE_MODE=dry_run': 'TRADE_MODE=dry_run  # ✅ Mantenemos dry_run por seguridad',
    }
    
    updated = False
    for old, new in updates.items():
        if old in content:
            content = content.replace(old, new)
            updated = True
            print(f"✅ Actualizado: {old} → {new}")
    
    if updated:
        # Crear backup
        backup_file = f".env.backup.{int(asyncio.get_event_loop().time())}"
        shutil.copy(env_file, backup_file)
        print(f"💾 Backup creado: {backup_file}")
        
        # Escribir configuración actualizada
        with open(env_file, 'w') as f:
            f.write(content)
        print("✅ Configuración actualizada para producción")
    else:
        print("✅ Configuración ya está correcta")
    
    return True

async def test_production_connection():
    """🔗 Probar conexión con Binance producción"""
    print("\n🔗 PROBANDO CONEXIÓN CON BINANCE PRODUCCIÓN")
    print("=" * 50)
    
    try:
        config = BinanceConfig(testnet=False)  # 🔴 PRODUCCIÓN
        
        if not config.api_key or not config.secret_key:
            print("❌ API keys no configuradas correctamente")
            return False
        
        print(f"📡 Conectando a: {config.base_url}")
        
        async with BinanceHistoricalSync(config) as sync:
            # Solo probar conexión - no sincronizar aún
            account_info = await sync.get_account_info()
            
            if account_info:
                print("✅ Conexión exitosa con Binance")
                print(f"   💎 Tipo de cuenta: {account_info.get('accountType', 'UNKNOWN')}")
                print(f"   📈 Puede tradear: {account_info.get('canTrade', False)}")
                print(f"   💰 Puede retirar: {account_info.get('canWithdraw', False)}")
                
                # Mostrar algunos balances
                balances = account_info.get('balances', [])
                significant_balances = [b for b in balances if float(b['free']) > 0 or float(b['locked']) > 0]
                print(f"   💵 Assets con balance: {len(significant_balances)}")
                
                return True
            else:
                print("❌ Error conectando con Binance")
                return False
                
    except Exception as e:
        print(f"❌ Error de conexión: {e}")
        return False

async def execute_production_sync():
    """🚀 Ejecutar sincronización de producción"""
    print("\n🚀 EJECUTANDO SINCRONIZACIÓN DE PRODUCCIÓN")
    print("=" * 50)
    
    # Configurar para producción
    config = BinanceConfig(testnet=False)  # 🔴 PRODUCCIÓN
    
    # Parámetros de sincronización
    trading_symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
    days_back = 30  # Últimos 30 días
    
    print(f"🎯 Configuración de sincronización:")
    print(f"   🔴 Entorno: PRODUCCIÓN (cuenta real)")
    print(f"   💱 Símbolos: {', '.join(trading_symbols)}")
    print(f"   📅 Período: {days_back} días atrás")
    print(f"   🔒 Modo: Solo lectura (sin trading)")
    
    # Confirmación final
    print(f"\n⚠️  CONFIRMACIÓN FINAL:")
    print(f"   Estás a punto de sincronizar datos de tu cuenta REAL de Binance")
    print(f"   Esto es seguro - solo lectura, sin trading automático")
    
    user_input = input(f"\n📝 ¿Continuar? (escribe 'SI' para confirmar): ")
    
    if user_input.upper() != 'SI':
        print("❌ Operación cancelada por el usuario")
        return False
    
    try:
        async with BinanceHistoricalSync(config) as sync:
            results = await sync.full_sync(trading_symbols, days_back)
            
            print(f"\n📊 RESULTADOS DE SINCRONIZACIÓN")
            print("=" * 50)
            
            if results['success']:
                print(f"🎉 SINCRONIZACIÓN EXITOSA!")
                print(f"   ✅ Balance sincronizado: {'Sí' if results['balance_synced'] else 'No'}")
                print(f"   📈 Trades sincronizados: {results['total_trades_synced']}")
                print(f"   💹 PnL total: ${results['total_pnl']:.2f}")
                
                if results['symbols_synced']:
                    print(f"\n📊 Detalle por símbolo:")
                    for symbol_data in results['symbols_synced']:
                        print(f"   💱 {symbol_data['symbol']}: {symbol_data['synced_count']}/{symbol_data['trades_count']} trades")
                
                print(f"\n✅ TU BASE DE DATOS AHORA TIENE DATOS REALES!")
                print(f"📂 Ejecuta 'python verify_sync_results.py' para revisar")
                
                return True
            else:
                print(f"❌ SINCRONIZACIÓN CON ERRORES:")
                for error in results['errors']:
                    print(f"   ❌ {error}")
                return False
                
    except Exception as e:
        print(f"❌ Error durante sincronización: {e}")
        return False

async def main():
    """🚀 Función principal"""
    print("🔴 BINANCE PRODUCTION SYNC - CONFIGURACIÓN")
    print("=" * 60)
    
    # 1. Mostrar advertencias
    show_production_warnings()
    
    # 2. Verificar configuración
    if not check_production_config():
        print(f"\n❌ Configuración incorrecta. Configura tu .env primero.")
        print(f"📝 Edita el archivo .env con tus credenciales reales de Binance")
        return
    
    # 3. Actualizar para producción
    if not update_env_for_production():
        print(f"\n❌ Error actualizando configuración")
        return
    
    # 4. Probar conexión
    connection_ok = await test_production_connection()
    if not connection_ok:
        print(f"\n❌ No se pudo conectar a Binance. Verifica tus credenciales.")
        return
    
    # 5. Ejecutar sincronización
    success = await execute_production_sync()
    
    if success:
        print(f"\n🎉 ¡PROCESO COMPLETADO EXITOSAMENTE!")
        print(f"🗄️  Tu base de datos ahora contiene datos reales de tu cuenta")
        print(f"📊 Ejecuta 'python verify_sync_results.py' para ver el resumen")
    else:
        print(f"\n❌ Proceso completado con errores")

if __name__ == "__main__":
    asyncio.run(main()) 