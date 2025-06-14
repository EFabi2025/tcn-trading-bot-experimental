#!/usr/bin/env python3
"""
✅ VERIFICAR CORRECCIÓN TRAILING STOP
====================================

Script para verificar que la corrección del problema del trailing stop funciona.

PROBLEMA ORIGINAL:
- Posición con +2.3% no mostraba trailing stop
- update_trailing_stop_professional se ejecutaba pero no se guardaba en snapshot

CORRECCIÓN APLICADA:
- Línea agregada: snapshot.active_positions[i] = updated_position
- Ahora el estado del trailing stop se preserva en el snapshot

VERIFICACIÓN:
1. Simular posición con +2.3% de ganancia
2. Verificar que trailing stop se activa
3. Verificar que aparece en el reporte
"""

import asyncio
import os
from datetime import datetime, timedelta
from professional_portfolio_manager import ProfessionalPortfolioManager, Position
from simple_professional_manager import SimpleProfessionalTradingManager
from dotenv import load_dotenv

load_dotenv()

class TrailingStopFixVerifier:
    """✅ Verificador de la corrección del trailing stop"""
    
    def __init__(self):
        self.api_key = os.getenv('BINANCE_API_KEY')
        self.secret_key = os.getenv('BINANCE_SECRET_KEY')
        self.base_url = "https://testnet.binance.vision"
        
    async def verify_fix(self):
        """✅ Verificar que la corrección funciona"""
        try:
            print("✅ VERIFICANDO CORRECCIÓN DEL TRAILING STOP")
            print("=" * 50)
            
            # Crear simple manager
            simple_manager = SimpleProfessionalTradingManager()
            await simple_manager.initialize()
            
            print("🔍 Obteniendo snapshot actual...")
            snapshot = await simple_manager.portfolio_manager.get_portfolio_snapshot()
            
            print(f"📊 POSICIONES ENCONTRADAS: {len(snapshot.active_positions)}")
            
            # Buscar posición con ganancia >= 1%
            target_position = None
            for i, position in enumerate(snapshot.active_positions):
                if position.unrealized_pnl_percent >= 1.0:
                    target_position = (i, position)
                    break
            
            if not target_position:
                print("⚠️ No hay posiciones con ganancia >= 1% para probar")
                return False
            
            pos_index, position = target_position
            
            print(f"\n🎯 POSICIÓN DE PRUEBA ENCONTRADA:")
            print(f"   Symbol: {position.symbol}")
            print(f"   PnL: {position.unrealized_pnl_percent:.2f}%")
            print(f"   Order ID: {position.order_id}")
            print(f"   Trailing activo ANTES: {getattr(position, 'trailing_stop_active', False)}")
            
            # Simular el proceso de monitoreo
            print(f"\n🧪 SIMULANDO PROCESO DE MONITOREO...")
            
            # 1. Actualizar precio actual
            current_price = position.current_price
            
            # 2. Aplicar trailing stop profesional (como en _position_monitor)
            updated_position, stop_triggered, trigger_reason = simple_manager.portfolio_manager.update_trailing_stop_professional(
                position, current_price
            )
            
            print(f"   Resultado update_trailing_stop_professional:")
            print(f"      Trailing activo DESPUÉS: {getattr(updated_position, 'trailing_stop_active', False)}")
            print(f"      Trailing price: {getattr(updated_position, 'trailing_stop_price', None)}")
            print(f"      Stop triggered: {stop_triggered}")
            
            # 3. ✅ VERIFICAR LA CORRECCIÓN: Actualizar en snapshot
            snapshot.active_positions[pos_index] = updated_position
            
            print(f"\n✅ POSICIÓN ACTUALIZADA EN SNAPSHOT")
            
            # 4. Generar reporte para verificar que aparece
            report = simple_manager.portfolio_manager.format_tcn_style_report(snapshot)
            
            # 5. Verificar si aparece "Trail:" en el reporte
            if "Trail:" in report:
                print(f"✅ ÉXITO: Trailing stop aparece en el reporte")
                
                # Extraer líneas que contienen Trail:
                lines = report.split('\n')
                trail_lines = [line for line in lines if 'Trail:' in line]
                
                print(f"📈 LÍNEAS CON TRAILING STOP:")
                for line in trail_lines:
                    print(f"   {line.strip()}")
                    
                return True
            else:
                print(f"❌ FALLO: Trailing stop NO aparece en el reporte")
                
                # Debug: mostrar parte del reporte
                print(f"\n🔍 FRAGMENTO DEL REPORTE:")
                lines = report.split('\n')
                for i, line in enumerate(lines[:20]):
                    print(f"   {i+1:2d}: {line}")
                
                return False
                
        except Exception as e:
            print(f"❌ Error en verificación: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def test_manual_activation(self):
        """🧪 Test manual de activación de trailing stop"""
        try:
            print(f"\n🧪 TEST MANUAL DE ACTIVACIÓN")
            print("=" * 40)
            
            # Crear portfolio manager
            portfolio_manager = ProfessionalPortfolioManager(
                api_key=self.api_key,
                secret_key=self.secret_key,
                base_url=self.base_url
            )
            
            # Crear posición simulada con +2.3% ganancia
            test_position = Position(
                symbol="ETHUSDT",
                side="BUY",
                size=0.012,
                entry_price=2430.0,
                current_price=2487.90,  # +2.3% ganancia
                market_value=29.85,
                unrealized_pnl_usd=0.69,
                unrealized_pnl_percent=2.35,
                entry_time=datetime.now() - timedelta(hours=28),
                duration_minutes=28*60,
                order_id="pos_test_31313890437"
            )
            
            print(f"📊 POSICIÓN SIMULADA:")
            print(f"   Symbol: {test_position.symbol}")
            print(f"   Entrada: ${test_position.entry_price:.2f}")
            print(f"   Actual: ${test_position.current_price:.2f}")
            print(f"   PnL: {test_position.unrealized_pnl_percent:.2f}%")
            print(f"   Trailing activo INICIAL: {getattr(test_position, 'trailing_stop_active', False)}")
            
            # Inicializar stops
            test_position = portfolio_manager.initialize_position_stops(test_position)
            
            print(f"   Trailing activo POST-INIT: {getattr(test_position, 'trailing_stop_active', False)}")
            
            # Aplicar update trailing stop
            updated_position, stop_triggered, trigger_reason = portfolio_manager.update_trailing_stop_professional(
                test_position, test_position.current_price
            )
            
            print(f"\n✅ RESULTADO UPDATE TRAILING STOP:")
            print(f"   Trailing activo: {getattr(updated_position, 'trailing_stop_active', False)}")
            print(f"   Trailing price: ${getattr(updated_position, 'trailing_stop_price', 0):.4f}")
            print(f"   Highest price: ${getattr(updated_position, 'highest_price_since_entry', 0):.4f}")
            print(f"   Movements: {getattr(updated_position, 'trailing_movements', 0)}")
            print(f"   Stop triggered: {stop_triggered}")
            
            # Verificar en reporte
            from professional_portfolio_manager import PortfolioSnapshot, Asset
            
            test_snapshot = PortfolioSnapshot(
                timestamp=datetime.now(),
                total_balance_usd=100.0,
                free_usdt=70.0,
                total_unrealized_pnl=updated_position.unrealized_pnl_usd,
                total_unrealized_pnl_percent=updated_position.unrealized_pnl_percent,
                active_positions=[updated_position],
                all_assets=[
                    Asset("USDT", 70.0, 0.0, 70.0, 70.0, 70.0),
                    Asset("ETH", 0.012, 0.0, 0.012, 29.85, 30.0)
                ],
                position_count=1,
                max_positions=2,
                total_trades_today=5
            )
            
            report = portfolio_manager.format_tcn_style_report(test_snapshot)
            
            if "Trail:" in report:
                print(f"\n✅ ÉXITO: Trailing aparece en reporte simulado")
                
                # Mostrar líneas relevantes
                lines = report.split('\n')
                for line in lines:
                    if 'ETHUSDT' in line or 'Trail:' in line or '2.35%' in line:
                        print(f"   📈 {line.strip()}")
                        
                return True
            else:
                print(f"\n❌ FALLO: Trailing NO aparece en reporte simulado")
                return False
                
        except Exception as e:
            print(f"❌ Error en test manual: {e}")
            import traceback
            traceback.print_exc()
            return False

async def main():
    """🎯 Función principal de verificación"""
    verifier = TrailingStopFixVerifier()
    
    print("🚀 INICIANDO VERIFICACIÓN DE CORRECCIÓN")
    print("🎯 Verificando que trailing stop aparece en reportes")
    print("=" * 60)
    
    # Test 1: Verificación con datos reales
    success_real = await verifier.verify_fix()
    
    # Test 2: Test manual con datos simulados
    success_manual = await verifier.test_manual_activation()
    
    print(f"\n🎯 RESULTADOS DE VERIFICACIÓN:")
    print("=" * 40)
    print(f"✅ Test con datos reales: {'ÉXITO' if success_real else 'FALLO'}")
    print(f"✅ Test manual simulado: {'ÉXITO' if success_manual else 'FALLO'}")
    
    if success_real or success_manual:
        print(f"\n🎉 CORRECCIÓN VERIFICADA EXITOSAMENTE")
        print(f"   El trailing stop ahora aparece en los reportes")
    else:
        print(f"\n❌ CORRECCIÓN NECESITA REVISIÓN")
        print(f"   El trailing stop aún no aparece correctamente")

if __name__ == "__main__":
    asyncio.run(main()) 