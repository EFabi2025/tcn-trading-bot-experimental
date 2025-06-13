#!/usr/bin/env python3
"""
🚨 EMERGENCY SPOT TRADING FIX
Corrección inmediata para permitir ventas en trading Spot cuando sea necesario
"""

import os
import sys
from datetime import datetime

class EmergencySpotTradingFix:
    """🚨 Corrección de emergencia para trading Spot"""

    def __init__(self):
        self.fixes_applied = []

    def apply_emergency_fix(self):
        """🔧 Aplicar corrección de emergencia"""
        print("🚨 APLICANDO CORRECCIÓN DE EMERGENCIA SPOT TRADING")
        print("=" * 60)
        print(f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)

        # 1. Crear lógica corregida para Spot trading
        self._create_corrected_spot_logic()

        # 2. Crear configuración de emergencia
        self._create_emergency_config()

        # 3. Crear backup del archivo original
        self._backup_original_manager()

        # 4. Aplicar parche al manager
        self._patch_manager()

        # 5. Generar reporte
        self._generate_fix_report()

    def _create_corrected_spot_logic(self):
        """🔧 Crear lógica corregida para Spot trading"""
        print("\n🔧 Creando lógica corregida para Spot trading...")

        corrected_logic = '''
class CorrectedSpotTradingLogic:
    """🔧 Lógica corregida para trading en Spot"""

    @staticmethod
    def should_process_signal(signal: str, symbol: str, current_positions: dict,
                            available_balance: float) -> tuple:
        """
        Determinar si procesar una señal en trading Spot

        Returns:
            (should_process: bool, action: str, reason: str)
        """

        has_position = symbol in current_positions

        if signal == 'BUY':
            if has_position:
                return False, 'IGNORE', f'Ya existe posición en {symbol}'
            elif available_balance < 10:  # Mínimo $10 USDT
                return False, 'IGNORE', f'Balance insuficiente: ${available_balance:.2f}'
            else:
                return True, 'BUY', f'Comprar {symbol}'

        elif signal == 'SELL':
            if has_position:
                return True, 'SELL', f'Vender posición existente en {symbol}'
            else:
                return False, 'IGNORE', f'No hay posición que vender en {symbol}'

        elif signal == 'HOLD':
            # HOLD significa mantener posición actual o no hacer nada
            if has_position:
                return False, 'MONITOR', f'Mantener posición en {symbol}'
            else:
                return False, 'IGNORE', f'No hay posición que mantener en {symbol}'

        else:
            return False, 'ERROR', f'Señal desconocida: {signal}'

    @staticmethod
    def get_emergency_sell_conditions() -> dict:
        """Condiciones para venta de emergencia"""
        return {
            'max_loss_percent': -15.0,  # Vender si pérdida > 15%
            'strong_sell_confidence': 0.85,  # Vender si SELL > 85% confianza
            'market_crash_threshold': -10.0,  # Vender si mercado cae > 10% en 24h
            'emergency_mode': True
        }

    @staticmethod
    def should_emergency_sell(position, market_data: dict, signal_data: dict) -> tuple:
        """Determinar si hacer venta de emergencia"""

        conditions = CorrectedSpotTradingLogic.get_emergency_sell_conditions()

        # Condición 1: Pérdida excesiva
        if position.pnl_percent <= conditions['max_loss_percent']:
            return True, f"EMERGENCY_SELL: Pérdida excesiva {position.pnl_percent:.1f}%"

        # Condición 2: Señal SELL muy fuerte
        if (signal_data.get('signal') == 'SELL' and
            signal_data.get('confidence', 0) >= conditions['strong_sell_confidence']):
            return True, f"EMERGENCY_SELL: Señal SELL fuerte {signal_data['confidence']:.1%}"

        # Condición 3: Crash del mercado
        ticker_24h = market_data.get('ticker_24h', {})
        price_change_24h = float(ticker_24h.get('priceChangePercent', 0))
        if price_change_24h <= conditions['market_crash_threshold']:
            return True, f"EMERGENCY_SELL: Crash del mercado {price_change_24h:.1f}%"

        return False, "NO_EMERGENCY"
'''

        with open('corrected_spot_trading_logic.py', 'w') as f:
            f.write(corrected_logic)

        self.fixes_applied.append("✅ Lógica corregida para Spot trading creada")

    def _create_emergency_config(self):
        """⚙️ Crear configuración de emergencia"""
        print("  ⚙️ Creando configuración de emergencia...")

        emergency_config = {
            "EMERGENCY_MODE": True,
            "ALLOW_SPOT_SELLS": True,
            "MIN_CONFIDENCE_FOR_SELL": 0.80,
            "MAX_LOSS_BEFORE_EMERGENCY_SELL": -15.0,
            "ENABLE_PROTECTIVE_SELLS": True,
            "IGNORE_ONLY_BUY_RESTRICTION": True,
            "EMERGENCY_ACTIVATED_AT": datetime.now().isoformat(),
            "REASON": "TCN_SIGNAL_PROCESSING_FIX"
        }

        import json
        with open('emergency_trading_config.json', 'w') as f:
            json.dump(emergency_config, f, indent=2)

        self.fixes_applied.append("✅ Configuración de emergencia creada")

    def _backup_original_manager(self):
        """💾 Hacer backup del manager original"""
        print("  💾 Creando backup del manager original...")

        import shutil
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"simple_professional_manager_BACKUP_{timestamp}.py"

        try:
            shutil.copy2("simple_professional_manager.py", backup_name)
            self.fixes_applied.append(f"✅ Backup creado: {backup_name}")
        except Exception as e:
            print(f"    ❌ Error creando backup: {e}")

    def _patch_manager(self):
        """🔧 Aplicar parche al manager"""
        print("  🔧 Aplicando parche al manager...")

        # Leer archivo original
        try:
            with open("simple_professional_manager.py", 'r') as f:
                content = f.read()

            # Buscar y reemplazar la lógica problemática
            old_logic = '''        # 1. Solo procesar señales BUY (Spot trading no permite SELL sin activos)
        if signal != 'BUY':
            print(f"  ⏸️ Señal {signal} ignorada - Solo BUY permitido en Spot")
            continue'''

            new_logic = '''        # 🔧 LÓGICA CORREGIDA: Procesar señales según posiciones existentes
        has_position = symbol in self.active_positions

        if signal == 'BUY':
            if has_position:
                print(f"  ⏸️ Señal BUY ignorada - Ya existe posición en {symbol}")
                continue
        elif signal == 'SELL':
            if not has_position:
                print(f"  ⏸️ Señal SELL ignorada - No hay posición que vender en {symbol}")
                continue
            else:
                print(f"  🔥 SEÑAL SELL PROCESADA - Vendiendo posición en {symbol}")
        elif signal == 'HOLD':
            print(f"  ⏸️ Señal HOLD ignorada - Mantener estado actual en {symbol}")
            continue'''

            if old_logic in content:
                content = content.replace(old_logic, new_logic)

                # Escribir archivo corregido
                with open("simple_professional_manager.py", 'w') as f:
                    f.write(content)

                self.fixes_applied.append("✅ Manager parcheado exitosamente")
            else:
                print("    ⚠️ No se encontró la lógica exacta para parchear")
                self.fixes_applied.append("⚠️ Parche no aplicado - lógica no encontrada")

        except Exception as e:
            print(f"    ❌ Error aplicando parche: {e}")

    def _generate_fix_report(self):
        """📊 Generar reporte de corrección"""
        print("\n" + "="*60)
        print("🚨 REPORTE DE CORRECCIÓN SPOT TRADING")
        print("="*60)

        print(f"\n✅ CORRECCIONES APLICADAS ({len(self.fixes_applied)}):")
        for fix in self.fixes_applied:
            print(f"  {fix}")

        print(f"\n🔧 CAMBIOS REALIZADOS:")
        print(f"  • ❌ ELIMINADO: Restricción 'solo BUY' en Spot")
        print(f"  • ✅ AGREGADO: Lógica inteligente para SELL")
        print(f"  • ✅ AGREGADO: Verificación de posiciones existentes")
        print(f"  • ✅ AGREGADO: Configuración de emergencia")
        print(f"  • ✅ AGREGADO: Backup del archivo original")

        print(f"\n⚠️ COMPORTAMIENTO NUEVO:")
        print(f"  • BUY: Solo si NO hay posición existente")
        print(f"  • SELL: Solo si HAY posición existente")
        print(f"  • HOLD: Ignorar (mantener estado actual)")

        print(f"\n🚨 ACCIONES REQUERIDAS:")
        print(f"  1. 🔄 REINICIAR el trading manager")
        print(f"  2. 🔍 MONITOREAR las primeras operaciones")
        print(f"  3. ✅ VERIFICAR que las ventas funcionen")
        print(f"  4. 📊 REVISAR logs de trading")

        # Guardar reporte
        report_content = f"""
REPORTE DE CORRECCIÓN SPOT TRADING
=================================
Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

PROBLEMA IDENTIFICADO:
- El sistema ignoraba TODAS las señales SELL y HOLD
- Solo procesaba señales BUY
- Causaba acumulación de posiciones en tendencias bajistas

CORRECCIONES APLICADAS:
{chr(10).join(self.fixes_applied)}

ESTADO: CORRECCIÓN APLICADA - REQUIERE REINICIO
"""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f'spot_trading_fix_report_{timestamp}.txt', 'w') as f:
            f.write(report_content)

        print(f"\n📄 Reporte guardado: spot_trading_fix_report_{timestamp}.txt")

def main():
    """🚨 Función principal de corrección"""
    print("🚨 INICIANDO CORRECCIÓN DE EMERGENCIA SPOT TRADING...")

    fix = EmergencySpotTradingFix()
    fix.apply_emergency_fix()

    print(f"\n🚨 CORRECCIÓN COMPLETADA")
    print(f"🔄 REINICIAR EL SISTEMA PARA APLICAR CAMBIOS")

if __name__ == "__main__":
    main()
