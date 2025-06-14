#!/usr/bin/env python3
"""
🧪 TEST TRAILING STOP PERSISTENCE
=================================

Script para verificar que el trailing stop NO se resetea cuando se abre
una nueva posición del mismo par.

PROBLEMA ORIGINAL:
- Nueva posición del mismo par → order_id cambia → trailing stop se resetea

SOLUCIÓN IMPLEMENTADA:
- order_id estable basado en buy_order.order_id original
- Cache persistente por order_id
- Logging detallado para debugging
"""

import asyncio
import os
from datetime import datetime, timedelta
from professional_portfolio_manager import ProfessionalPortfolioManager, Position
from dotenv import load_dotenv

load_dotenv()

class TrailingStopPersistenceTest:
    """🧪 Test de persistencia del trailing stop"""
    
    def __init__(self):
        self.api_key = os.getenv('BINANCE_API_KEY')
        self.secret_key = os.getenv('BINANCE_SECRET_KEY')
        
        if not self.api_key or not self.secret_key:
            raise ValueError("❌ Se requieren BINANCE_API_KEY y BINANCE_SECRET_KEY")
            
        # Usar testnet para testing seguro
        self.portfolio_manager = ProfessionalPortfolioManager(
            api_key=self.api_key,
            secret_key=self.secret_key,
            base_url="https://testnet.binance.vision"
        )
        
    async def test_trailing_persistence(self):
        """🧪 Test principal de persistencia del trailing stop"""
        print("🧪 INICIANDO TEST DE PERSISTENCIA DEL TRAILING STOP")
        print("=" * 60)
        
        try:
            # 1. Obtener snapshot inicial
            print("\n📊 PASO 1: Obteniendo snapshot inicial...")
            snapshot1 = await self.portfolio_manager.get_portfolio_snapshot()
            
            print(f"   📈 Posiciones encontradas: {len(snapshot1.active_positions)}")
            
            # 2. Mostrar estado inicial del cache
            print("\n🔍 PASO 2: Estado inicial del cache...")
            self.portfolio_manager.debug_trailing_cache()
            
            # 3. Simular activación de trailing stops en posiciones existentes
            print("\n📈 PASO 3: Simulando activación de trailing stops...")
            
            btc_positions = [pos for pos in snapshot1.active_positions if pos.symbol == 'BTCUSDT']
            
            if btc_positions:
                for i, position in enumerate(btc_positions):
                    print(f"\n   🎯 Procesando posición BTCUSDT #{i+1}:")
                    print(f"      📍 Order ID: {position.order_id}")
                    print(f"      💰 Entrada: ${position.entry_price:.2f}")
                    print(f"      📊 Actual: ${position.current_price:.2f}")
                    
                    # Simular precio que active el trailing stop (+2% ganancia)
                    simulated_price = position.entry_price * 1.02
                    print(f"      🎭 Simulando precio: ${simulated_price:.2f} (+2%)")
                    
                    # Activar trailing stop
                    updated_pos, triggered, reason = self.portfolio_manager.update_trailing_stop_professional(
                        position, simulated_price
                    )
                    
                    if updated_pos.trailing_stop_active:
                        print(f"      ✅ Trailing activado: ${updated_pos.trailing_stop_price:.4f}")
                    else:
                        print(f"      ⚠️ Trailing no activado")
            else:
                print("   ⚠️ No hay posiciones BTCUSDT para testing")
                
            # 4. Mostrar estado del cache después de activación
            print("\n🔍 PASO 4: Estado del cache después de activación...")
            self.portfolio_manager.debug_trailing_cache()
            
            # 5. Simular nueva consulta de snapshot (como si hubiera nueva posición)
            print("\n📊 PASO 5: Simulando nueva consulta de snapshot...")
            print("   (Esto simula lo que pasa cuando se abre nueva posición del mismo par)")
            
            snapshot2 = await self.portfolio_manager.get_portfolio_snapshot()
            print(f"   📈 Posiciones en segundo snapshot: {len(snapshot2.active_positions)}")
            
            # 6. Verificar que los trailing stops se mantienen
            print("\n🔍 PASO 6: Verificando persistencia de trailing stops...")
            
            btc_positions_2 = [pos for pos in snapshot2.active_positions if pos.symbol == 'BTCUSDT']
            
            trailing_preserved = 0
            trailing_lost = 0
            
            for position in btc_positions_2:
                if hasattr(position, 'trailing_stop_active') and position.trailing_stop_active:
                    trailing_preserved += 1
                    protection = ((position.trailing_stop_price - position.entry_price) / position.entry_price) * 100
                    print(f"   ✅ TRAILING PRESERVADO {position.symbol} Pos #{position.order_id}:")
                    print(f"      📈 Estado: ACTIVO ${position.trailing_stop_price:.4f} (+{protection:.2f}%)")
                    print(f"      🏔️ Máximo: ${position.highest_price_since_entry:.4f}")
                    print(f"      📊 Movimientos: {position.trailing_movements}")
                else:
                    trailing_lost += 1
                    print(f"   ❌ TRAILING PERDIDO {position.symbol} Pos #{position.order_id}")
                    
            # 7. Mostrar estado final del cache
            print("\n🔍 PASO 7: Estado final del cache...")
            self.portfolio_manager.debug_trailing_cache()
            
            # 8. Resultado del test
            print("\n🎯 RESULTADO DEL TEST:")
            print("=" * 40)
            
            if trailing_lost == 0 and trailing_preserved > 0:
                print("✅ TEST EXITOSO: Trailing stops preservados correctamente")
                print(f"   📈 Trailing stops activos: {trailing_preserved}")
                print(f"   ❌ Trailing stops perdidos: {trailing_lost}")
            elif trailing_lost > 0:
                print("❌ TEST FALLIDO: Se perdieron trailing stops")
                print(f"   📈 Trailing stops preservados: {trailing_preserved}")
                print(f"   ❌ Trailing stops perdidos: {trailing_lost}")
            else:
                print("⚠️ TEST INCONCLUSO: No había trailing stops para verificar")
                
        except Exception as e:
            print(f"❌ Error en test: {e}")
            
    async def test_order_id_stability(self):
        """🧪 Test específico de estabilidad del order_id"""
        print("\n🧪 TEST DE ESTABILIDAD DEL ORDER_ID")
        print("=" * 40)
        
        try:
            # Obtener múltiples snapshots y verificar que los order_ids son estables
            snapshots = []
            
            for i in range(3):
                print(f"\n📊 Snapshot #{i+1}...")
                snapshot = await self.portfolio_manager.get_portfolio_snapshot()
                snapshots.append(snapshot)
                
                # Mostrar order_ids de posiciones BTCUSDT
                btc_positions = [pos for pos in snapshot.active_positions if pos.symbol == 'BTCUSDT']
                for pos in btc_positions:
                    print(f"   📋 {pos.symbol} Order ID: {pos.order_id}")
                    
                await asyncio.sleep(1)  # Pequeña pausa
                
            # Verificar estabilidad
            if len(snapshots) >= 2:
                stable_ids = True
                
                for i in range(1, len(snapshots)):
                    prev_ids = {pos.order_id for pos in snapshots[i-1].active_positions if pos.symbol == 'BTCUSDT'}
                    curr_ids = {pos.order_id for pos in snapshots[i].active_positions if pos.symbol == 'BTCUSDT'}
                    
                    if prev_ids != curr_ids:
                        stable_ids = False
                        print(f"❌ Order IDs cambiaron entre snapshot {i} y {i+1}")
                        print(f"   Anterior: {prev_ids}")
                        print(f"   Actual: {curr_ids}")
                        
                if stable_ids:
                    print("✅ Order IDs estables entre snapshots")
                else:
                    print("❌ Order IDs inestables - PROBLEMA DETECTADO")
                    
        except Exception as e:
            print(f"❌ Error en test de estabilidad: {e}")

async def main():
    """🚀 Función principal del test"""
    try:
        test = TrailingStopPersistenceTest()
        
        # Test 1: Persistencia del trailing stop
        await test.test_trailing_persistence()
        
        # Test 2: Estabilidad del order_id
        await test.test_order_id_stability()
        
        print("\n🎉 TESTS COMPLETADOS")
        
    except Exception as e:
        print(f"❌ Error en tests: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 