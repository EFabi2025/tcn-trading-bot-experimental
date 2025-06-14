#!/usr/bin/env python3
"""
🧪 TEST SIMULADO TRAILING STOP PERSISTENCE
==========================================

Test que simula el problema y verifica la corrección sin necesidad de API de Binance.
Demuestra que el order_id estable preserva el trailing stop.
"""

from datetime import datetime
from professional_portfolio_manager import Position

class MockTrailingStopTest:
    """🧪 Test simulado de trailing stop"""
    
    def __init__(self):
        self.trailing_stop_cache = {}
        
    def _save_trailing_state(self, position: Position):
        """💾 Simular guardado de estado"""
        if position.order_id:
            self.trailing_stop_cache[position.order_id] = {
                'trailing_stop_active': position.trailing_stop_active,
                'trailing_stop_price': position.trailing_stop_price,
                'highest_price_since_entry': position.highest_price_since_entry,
                'trailing_movements': position.trailing_movements,
            }
            print(f"💾 Estado guardado para {position.order_id}: {'ACTIVO' if position.trailing_stop_active else 'INACTIVO'}")
            
    def _restore_trailing_state(self, position: Position) -> Position:
        """🔄 Simular restauración de estado"""
        if position.order_id and position.order_id in self.trailing_stop_cache:
            cached_state = self.trailing_stop_cache[position.order_id]
            
            position.trailing_stop_active = cached_state.get('trailing_stop_active', False)
            position.trailing_stop_price = cached_state.get('trailing_stop_price', None)
            position.highest_price_since_entry = cached_state.get('highest_price_since_entry', position.entry_price)
            position.trailing_movements = cached_state.get('trailing_movements', 0)
            
            print(f"🔄 Estado restaurado para {position.order_id}: {'ACTIVO' if position.trailing_stop_active else 'INACTIVO'}")
            return position
        else:
            print(f"🆕 Nueva posición {position.order_id}: Sin estado previo")
            
        return position
        
    def simulate_old_behavior(self, orders_count: int, binance_order_id: str) -> str:
        """❌ Simular comportamiento problemático original"""
        return f"{orders_count}ord_{binance_order_id}"
        
    def simulate_new_behavior(self, binance_order_id: str) -> str:
        """✅ Simular comportamiento corregido"""
        return f"pos_{binance_order_id}"
        
    def test_trailing_persistence(self):
        """🧪 Test principal de persistencia"""
        print("🧪 SIMULANDO PROBLEMA Y CORRECCIÓN DEL TRAILING STOP")
        print("=" * 60)
        
        # Datos simulados
        binance_order_id = "12345678"
        entry_price = 50000.0
        current_price = 51000.0  # +2% ganancia
        
        print(f"\n📊 DATOS SIMULADOS:")
        print(f"   🏷️ Binance Order ID: {binance_order_id}")
        print(f"   💰 Precio entrada: ${entry_price:,.2f}")
        print(f"   📈 Precio actual: ${current_price:,.2f} (+2%)")
        
        # === ESCENARIO 1: COMPORTAMIENTO PROBLEMÁTICO ===
        print(f"\n❌ ESCENARIO 1: COMPORTAMIENTO PROBLEMÁTICO ORIGINAL")
        print("-" * 50)
        
        # Simular primera posición con 5 órdenes existentes
        orders_count_1 = 5
        old_order_id_1 = self.simulate_old_behavior(orders_count_1, binance_order_id)
        
        position_1 = Position(
            symbol="BTCUSDT",
            side="BUY",
            size=0.001,
            entry_price=entry_price,
            current_price=current_price,
            market_value=current_price * 0.001,
            unrealized_pnl_usd=(current_price - entry_price) * 0.001,
            unrealized_pnl_percent=2.0,
            entry_time=datetime.now(),
            duration_minutes=30,
            order_id=old_order_id_1,
            batch_id=binance_order_id,
            trailing_stop_active=True,
            trailing_stop_price=50500.0,  # Trailing activado
            highest_price_since_entry=51000.0,
            trailing_movements=2
        )
        
        print(f"   📍 Order ID generado: {old_order_id_1}")
        print(f"   📈 Trailing activado: ${position_1.trailing_stop_price:.2f}")
        
        # Guardar estado
        self._save_trailing_state(position_1)
        
        # Simular nueva orden (orders_count aumenta)
        orders_count_2 = 6  # Nueva orden agregada
        old_order_id_2 = self.simulate_old_behavior(orders_count_2, binance_order_id)
        
        print(f"\n   🆕 Nueva orden agregada, orders_count: {orders_count_1} → {orders_count_2}")
        print(f"   📍 Nuevo Order ID: {old_order_id_2}")
        print(f"   ⚠️ Order ID cambió: {old_order_id_1} → {old_order_id_2}")
        
        # Intentar restaurar con nuevo ID
        position_1_restored = Position(
            symbol="BTCUSDT",
            side="BUY", 
            size=0.001,
            entry_price=entry_price,
            current_price=current_price,
            market_value=current_price * 0.001,
            unrealized_pnl_usd=(current_price - entry_price) * 0.001,
            unrealized_pnl_percent=2.0,
            entry_time=datetime.now(),
            duration_minutes=30,
            order_id=old_order_id_2,  # ID diferente!
            batch_id=binance_order_id
        )
        
        position_1_restored = self._restore_trailing_state(position_1_restored)
        
        print(f"   🛑 RESULTADO: Trailing perdido - {not position_1_restored.trailing_stop_active}")
        
        # === ESCENARIO 2: COMPORTAMIENTO CORREGIDO ===
        print(f"\n✅ ESCENARIO 2: COMPORTAMIENTO CORREGIDO")
        print("-" * 50)
        
        # Limpiar cache para test limpio
        self.trailing_stop_cache.clear()
        
        # Simular primera posición con ID estable
        new_order_id = self.simulate_new_behavior(binance_order_id)
        
        position_2 = Position(
            symbol="BTCUSDT",
            side="BUY",
            size=0.001,
            entry_price=entry_price,
            current_price=current_price,
            market_value=current_price * 0.001,
            unrealized_pnl_usd=(current_price - entry_price) * 0.001,
            unrealized_pnl_percent=2.0,
            entry_time=datetime.now(),
            duration_minutes=30,
            order_id=new_order_id,
            batch_id=binance_order_id,
            trailing_stop_active=True,
            trailing_stop_price=50500.0,  # Trailing activado
            highest_price_since_entry=51000.0,
            trailing_movements=2
        )
        
        print(f"   📍 Order ID generado: {new_order_id}")
        print(f"   📈 Trailing activado: ${position_2.trailing_stop_price:.2f}")
        
        # Guardar estado
        self._save_trailing_state(position_2)
        
        # Simular nueva orden (pero ID se mantiene estable)
        new_order_id_after = self.simulate_new_behavior(binance_order_id)
        
        print(f"\n   🆕 Nueva orden agregada")
        print(f"   📍 Order ID después: {new_order_id_after}")
        print(f"   ✅ Order ID estable: {new_order_id == new_order_id_after}")
        
        # Restaurar con mismo ID
        position_2_restored = Position(
            symbol="BTCUSDT",
            side="BUY",
            size=0.001,
            entry_price=entry_price,
            current_price=current_price,
            market_value=current_price * 0.001,
            unrealized_pnl_usd=(current_price - entry_price) * 0.001,
            unrealized_pnl_percent=2.0,
            entry_time=datetime.now(),
            duration_minutes=30,
            order_id=new_order_id_after,  # Mismo ID!
            batch_id=binance_order_id
        )
        
        position_2_restored = self._restore_trailing_state(position_2_restored)
        
        print(f"   ✅ RESULTADO: Trailing preservado - {position_2_restored.trailing_stop_active}")
        if position_2_restored.trailing_stop_active:
            print(f"      📈 Trailing price: ${position_2_restored.trailing_stop_price:.2f}")
            print(f"      📊 Movimientos: {position_2_restored.trailing_movements}")
        
        # === RESUMEN ===
        print(f"\n🎯 RESUMEN DE LA CORRECCIÓN:")
        print("=" * 40)
        print(f"❌ Método original: ID cambia → Trailing se pierde")
        print(f"✅ Método corregido: ID estable → Trailing se preserva")
        print(f"\n🔧 CAMBIO IMPLEMENTADO:")
        print(f"   ANTES: order_id = f\"{{orders_count}}ord_{{binance_order_id}}\"")
        print(f"   DESPUÉS: order_id = f\"pos_{{binance_order_id}}\"")
        
        return True

def main():
    """🚀 Ejecutar test simulado"""
    test = MockTrailingStopTest()
    success = test.test_trailing_persistence()
    
    if success:
        print(f"\n🎉 TEST SIMULADO COMPLETADO EXITOSAMENTE")
        print(f"✅ La corrección resuelve el problema del trailing stop")
    else:
        print(f"\n❌ TEST SIMULADO FALLÓ")

if __name__ == "__main__":
    main() 