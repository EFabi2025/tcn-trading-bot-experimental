#!/usr/bin/env python3
"""
🧪 TEST BINANCE CONNECTION
Script para probar la conectividad y obtención de datos de Binance
"""

import asyncio
import os
from dotenv import load_dotenv
from simple_professional_manager import SimpleProfessionalTradingManager

load_dotenv()

async def test_binance_connection():
    """🧪 Probar conexión completa con Binance"""
    print("🧪 TESTING BINANCE CONNECTION")
    print("=" * 50)
    
    try:
        # Verificar variables de entorno
        api_key = os.getenv('BINANCE_API_KEY')
        secret_key = os.getenv('BINANCE_SECRET_KEY')
        base_url = os.getenv('BINANCE_BASE_URL', 'https://testnet.binance.vision')
        
        if not api_key or not secret_key:
            print("❌ Error: Faltan BINANCE_API_KEY o BINANCE_SECRET_KEY en .env")
            return False
        
        print(f"✅ API Key configurada: {api_key[:8]}...")
        print(f"✅ Secret Key configurada: {secret_key[:8]}...")
        print(f"✅ Base URL: {base_url}")
        print()
        
        # Crear manager
        manager = SimpleProfessionalTradingManager()
        
        # Test 1: Obtener precios públicos
        print("🔍 Test 1: Obteniendo precios públicos...")
        btc_price = await manager.get_current_price('BTCUSDT')
        eth_price = await manager.get_current_price('ETHUSDT')
        bnb_price = await manager.get_current_price('BNBUSDT')
        
        if btc_price > 0 and eth_price > 0 and bnb_price > 0:
            print(f"   ✅ BTCUSDT: ${btc_price:.2f}")
            print(f"   ✅ ETHUSDT: ${eth_price:.2f}")
            print(f"   ✅ BNBUSDT: ${bnb_price:.2f}")
        else:
            print("   ❌ Error obteniendo precios públicos")
            return False
        print()
        
        # Test 2: Obtener información de cuenta (requiere autenticación)
        print("🔍 Test 2: Obteniendo información de cuenta...")
        account_info = await manager.get_account_info()
        
        if account_info:
            print(f"   ✅ Balance USDT: ${account_info.usdt_balance:.2f}")
            print(f"   ✅ Balance total USD: ${account_info.total_balance_usd:.2f}")
            print(f"   ✅ Activos encontrados: {len(account_info.balances)}")
            
            # Mostrar balances no-cero
            non_zero_balances = {k: v for k, v in account_info.balances.items() if v['total'] > 0}
            if non_zero_balances:
                print("   💰 Balances actuales:")
                for asset, balance in non_zero_balances.items():
                    print(f"      {asset}: {balance['total']:.8f} (libre: {balance['free']:.8f}, bloqueado: {balance['locked']:.8f})")
            else:
                print("   ⚠️ No se encontraron balances activos")
                
        else:
            print("   ❌ Error obteniendo información de cuenta")
            return False
        print()
        
        # Test 3: Verificar actualización de balance
        print("🔍 Test 3: Actualizando balance...")
        old_balance = manager.current_balance
        success = await manager.update_balance_from_binance()
        
        if success:
            print(f"   ✅ Balance actualizado: ${old_balance:.2f} → ${manager.current_balance:.2f}")
            print(f"   ✅ Información de cuenta cacheada: {'Sí' if manager.account_info else 'No'}")
        else:
            print("   ❌ Error actualizando balance")
            return False
        print()
        
        # Test 4: Verificar métricas
        print("🔍 Test 4: Verificando métricas...")
        print(f"   📊 API calls realizadas: {manager.metrics.get('api_calls_count', 0)}")
        print(f"   📈 Balance updates: {manager.metrics.get('balance_updates', 0)}")
        print(f"   ❌ Errores: {manager.metrics.get('error_count', 0)}")
        if manager.metrics.get('last_error'):
            print(f"   🚨 Último error: {manager.metrics.get('last_error')}")
        print()
        
        print("✅ TODOS LOS TESTS PASARON")
        print("🎯 El trading manager debería funcionar correctamente")
        return True
        
    except Exception as e:
        print(f"❌ Error durante los tests: {e}")
        return False

async def test_continuous_updates():
    """🔄 Probar actualizaciones continuas"""
    print("\n🔄 TEST ACTUALIZACIONES CONTINUAS")
    print("=" * 50)
    print("⚠️ Este test correrá por 3 minutos actualizando datos...")
    print("⏸️ Presiona Ctrl+C para detener\n")
    
    manager = SimpleProfessionalTradingManager()
    
    try:
        for i in range(18):  # 3 minutos (18 x 10 segundos)
            print(f"🔄 Actualización {i+1}/18...")
            
            # Obtener precios
            prices = await manager._get_current_prices()
            
            # Actualizar balance cada 5 actualizaciones
            if i % 5 == 0:
                await manager.update_balance_from_binance()
            
            # Mostrar resumen
            print(f"   💰 Balance: ${manager.current_balance:.2f}")
            print(f"   📊 API calls: {manager.metrics.get('api_calls_count', 0)}")
            print(f"   ❌ Errores: {manager.metrics.get('error_count', 0)}")
            print()
            
            await asyncio.sleep(10)
            
        print("✅ Test de actualizaciones continuas completado")
        
    except KeyboardInterrupt:
        print("\n⏹️ Test interrumpido por el usuario")
    except Exception as e:
        print(f"\n❌ Error durante test continuo: {e}")

async def main():
    """🎯 Función principal"""
    print("🧪 BINANCE CONNECTION TESTER")
    print("Verificando conectividad y funcionalidad básica...")
    print()
    
    # Test básico
    success = await test_binance_connection()
    
    if success:
        # Preguntar si hacer test continuo
        try:
            response = input("\n¿Ejecutar test de actualizaciones continuas? (y/N): ").lower()
            if response in ['y', 'yes', 'sí', 'si']:
                await test_continuous_updates()
        except KeyboardInterrupt:
            print("\n👋 Tests terminados")
    else:
        print("\n❌ Los tests básicos fallaron. Revisa tu configuración.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Tester detenido por el usuario")
    except Exception as e:
        print(f"\n❌ Error fatal en tester: {e}") 