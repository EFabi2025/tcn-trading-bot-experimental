#!/usr/bin/env python3
"""
🧹 Utilidad para limpiar balances antiguos y optimizar el sistema de trading
"""

import asyncio
import os
import sys
from datetime import datetime
from dotenv import load_dotenv

# Agregar el directorio actual al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from professional_portfolio_manager import ProfessionalPortfolioManager

async def cleanup_balances():
    """🧹 Limpiar balances antiguos y optimizar el sistema"""
    print("🧹 UTILIDAD DE LIMPIEZA DE BALANCES")
    print("=" * 50)
    
    try:
        # Cargar variables de entorno
        load_dotenv()
        
        # Configuración
        api_key = os.getenv('BINANCE_API_KEY')
        secret_key = os.getenv('BINANCE_SECRET_KEY')
        base_url = os.getenv('BINANCE_BASE_URL', 'https://testnet.binance.vision')
        
        if not api_key or not secret_key:
            print("❌ Faltan credenciales de Binance")
            return
        
        # Crear manager
        portfolio_manager = ProfessionalPortfolioManager(api_key, secret_key, base_url)
        
        # Inicializar
        await portfolio_manager.initialize()
        
        print("\n📊 ANÁLISIS DE BALANCES ACTUALES")
        print("-" * 30)
        
        # 1. Obtener balances RAW (sin filtros)
        raw_balances = await portfolio_manager.get_raw_account_balances()
        print(f"📋 Total de activos con balance: {len(raw_balances)}")
        
        # 2. Obtener balances filtrados (con validación)
        filtered_balances = await portfolio_manager.get_account_balances()
        print(f"✅ Activos válidos para trading: {len(filtered_balances)}")
        
        # 3. Identificar activos problemáticos
        problematic_assets = []
        for asset, balance in raw_balances.items():
            if asset != 'USDT' and asset not in filtered_balances:
                symbol = f"{asset}USDT"
                problematic_assets.append({
                    'asset': asset,
                    'symbol': symbol,
                    'balance': balance['total']
                })
        
        if problematic_assets:
            print(f"\n⚠️ ACTIVOS PROBLEMÁTICOS ENCONTRADOS: {len(problematic_assets)}")
            print("-" * 50)
            
            for item in problematic_assets:
                print(f"   • {item['asset']}: {item['balance']} (símbolo: {item['symbol']})")
            
            print(f"\n💡 RECOMENDACIONES:")
            print("   1. Estos activos tienen balance pero no son símbolos de trading válidos")
            print("   2. Considera convertirlos a USDT si es posible")
            print("   3. El sistema los omitirá automáticamente en futuras operaciones")
        else:
            print("\n✅ No se encontraron activos problemáticos")
        
        # 4. Mostrar estado del cache
        print(f"\n📊 ESTADO DEL CACHE")
        print("-" * 20)
        cache_status = portfolio_manager.get_cache_status()
        for key, value in cache_status.items():
            print(f"   • {key}: {value}")
        
        # 5. Opciones de limpieza
        print(f"\n🛠️ OPCIONES DE LIMPIEZA")
        print("-" * 25)
        print("   1. Limpiar cache de reportes (mostrar símbolos inválidos nuevamente)")
        print("   2. Forzar refresco completo de balances")
        print("   3. Limpiar cache de símbolos válidos/inválidos")
        print("   4. Salir")
        
        while True:
            try:
                choice = input("\nSelecciona una opción (1-4): ").strip()
                
                if choice == "1":
                    portfolio_manager.force_balance_refresh()
                    print("✅ Cache de reportes limpiado")
                    
                elif choice == "2":
                    fresh_balances = await portfolio_manager.get_fresh_balances_with_validation()
                    print(f"✅ Balances frescos obtenidos: {len(fresh_balances)} activos")
                    
                elif choice == "3":
                    portfolio_manager._clear_symbol_validity_cache()
                    print("✅ Cache de símbolos limpiado")
                    
                elif choice == "4":
                    print("👋 ¡Hasta luego!")
                    break
                    
                else:
                    print("❌ Opción inválida")
                    
            except KeyboardInterrupt:
                print("\n👋 ¡Hasta luego!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
        
    except Exception as e:
        print(f"❌ Error durante la limpieza: {e}")

async def main():
    """Función principal"""
    await cleanup_balances()

if __name__ == "__main__":
    asyncio.run(main())
