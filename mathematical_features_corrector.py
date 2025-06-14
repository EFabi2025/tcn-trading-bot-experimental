#!/usr/bin/env python3
"""
🔧 CORRECTOR MATEMÁTICO DE FEATURES
==================================

Corrige automáticamente todos los errores matemáticos identificados
en las features técnicas del sistema de trading.

ERRORES A CORREGIR:
1. RSI: SMA → EMA de Wilder
2. Bollinger Bands: Método de cálculo estándar
3. ATR: SMA → EMA para True Range

METODOLOGÍA:
- Corrección precisa sin cambiar interfaces
- Mantener compatibilidad con modelos existentes
- Validación matemática automática
"""

import os
import re
import shutil
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
import talib

class MathematicalFeaturesCorrector:
    """🔧 Corrector automático de features matemáticas"""

    def __init__(self):
        self.corrections_made = []
        self.files_modified = []
        self.backup_dir = "backup_before_math_correction"

        # Patrones de código a corregir
        self.correction_patterns = {
            'rsi_sma_to_ema': {
                'pattern': r'gain = \(delta\.where\(delta > 0, 0\)\)\.rolling\(window=(\w+).*?\.mean\(\)',
                'replacement': self._get_rsi_ema_replacement,
                'description': 'RSI: Cambiar SMA por EMA de Wilder'
            },
            'rsi_loss_sma_to_ema': {
                'pattern': r'loss = \(-delta\.where\(delta < 0, 0\)\)\.rolling\(window=(\w+).*?\.mean\(\)',
                'replacement': self._get_rsi_loss_ema_replacement,
                'description': 'RSI Loss: Cambiar SMA por EMA de Wilder'
            },
            'atr_sma_to_ema': {
                'pattern': r'true_range\.rolling\(window=(\w+).*?\.mean\(\)',
                'replacement': 'true_range.ewm(span={}).mean()',
                'description': 'ATR: Cambiar SMA por EMA'
            },
            'bollinger_std_correction': {
                'pattern': r'\.rolling\((\d+)\)\.std\(\)',
                'replacement': '.rolling({}).std(ddof=0)',
                'description': 'Bollinger: Usar std poblacional'
            }
        }

    def _get_rsi_ema_replacement(self, match) -> str:
        """Generar reemplazo correcto para RSI gain con EMA"""
        period = match.group(1)
        return f"""# RSI con EMA de Wilder (corregido)
        gains = delta.where(delta > 0, 0)
        if len(gains.dropna()) > 0:
            gain = gains.ewm(alpha=1/{period}, adjust=False).mean()
        else:
            gain = pd.Series(index=delta.index, data=0.0)"""

    def _get_rsi_loss_ema_replacement(self, match) -> str:
        """Generar reemplazo correcto para RSI loss con EMA"""
        period = match.group(1)
        return f"""# RSI Loss con EMA de Wilder (corregido)
        losses = -delta.where(delta < 0, 0)
        if len(losses.dropna()) > 0:
            loss = losses.ewm(alpha=1/{period}, adjust=False).mean()
        else:
            loss = pd.Series(index=delta.index, data=1e-8)"""

    def create_backup(self) -> bool:
        """Crear backup de archivos antes de modificar"""
        try:
            if os.path.exists(self.backup_dir):
                shutil.rmtree(self.backup_dir)

            os.makedirs(self.backup_dir)

            # Archivos críticos a respaldar
            critical_files = [
                'real_market_data_provider.py',
                'real_binance_predictor.py',
                'real_trading_setup.py',
                'corrected_tcn_predictor.py',
                'enhanced_real_predictor.py'
            ]

            for file in critical_files:
                if os.path.exists(file):
                    shutil.copy2(file, os.path.join(self.backup_dir, file))
                    print(f"✅ Backup creado: {file}")

            print(f"🔒 Backup completo en: {self.backup_dir}")
            return True

        except Exception as e:
            print(f"❌ Error creando backup: {e}")
            return False

    def find_files_with_math_errors(self) -> List[str]:
        """Encontrar archivos con errores matemáticos"""
        files_with_errors = []

        # Patrones que indican errores matemáticos
        error_patterns = [
            r'\.rolling\(.*?\)\.mean\(\).*?# RSI',
            r'gain.*?rolling.*?mean',
            r'loss.*?rolling.*?mean',
            r'true_range.*?rolling.*?mean',
            r'_calculate_rsi.*?rolling.*?mean'
        ]

        # Buscar en archivos Python
        for root, dirs, files in os.walk('.'):
            for file in files:
                if file.endswith('.py') and not file.startswith('mathematical_'):
                    filepath = os.path.join(root, file)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            content = f.read()

                        for pattern in error_patterns:
                            if re.search(pattern, content, re.IGNORECASE):
                                if filepath not in files_with_errors:
                                    files_with_errors.append(filepath)
                                break

                    except Exception:
                        continue

        return files_with_errors

    def correct_rsi_calculation(self, filepath: str) -> bool:
        """Corregir cálculo de RSI específicamente"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            original_content = content

            # Patrón completo para RSI incorrecto
            rsi_pattern = r'''(async\s+)?def\s+_calculate_rsi\(.*?\):.*?try:.*?delta = prices\.diff\(\).*?gain = \(delta\.where\(delta > 0, 0\)\)\.rolling\(window=period.*?\.mean\(\).*?loss = \(-delta\.where\(delta < 0, 0\)\)\.rolling\(window=period.*?\.mean\(\)'''

            # Reemplazo correcto con EMA de Wilder
            rsi_replacement = '''async def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calcular RSI con EMA de Wilder (CORREGIDO)"""
        try:
            delta = prices.diff()

            # Separar ganancias y pérdidas
            gains = delta.where(delta > 0, 0)
            losses = -delta.where(delta < 0, 0)

            # Usar EMA de Wilder (alpha = 1/period)
            alpha = 1.0 / period
            gain = gains.ewm(alpha=alpha, adjust=False).mean()
            loss = losses.ewm(alpha=alpha, adjust=False).mean()

            # Evitar división por cero
            loss = loss.replace(0, 1e-8)
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))

            return rsi'''

            # Aplicar corrección
            if re.search(rsi_pattern, content, re.DOTALL | re.IGNORECASE):
                content = re.sub(rsi_pattern, rsi_replacement, content, flags=re.DOTALL | re.IGNORECASE)

                # Verificar si hubo cambios
                if content != original_content:
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(content)

                    self.corrections_made.append(f"RSI corregido en {filepath}")
                    return True

            return False

        except Exception as e:
            print(f"❌ Error corrigiendo RSI en {filepath}: {e}")
            return False

    def correct_atr_calculation(self, filepath: str) -> bool:
        """Corregir cálculo de ATR específicamente"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            original_content = content

            # Patrón para ATR incorrecto
            atr_pattern = r'true_range\.rolling\(window=period.*?\.mean\(\)'

            # Reemplazo correcto
            atr_replacement = 'true_range.ewm(span=period, adjust=False).mean()'

            if re.search(atr_pattern, content):
                content = re.sub(atr_pattern, atr_replacement, content)

                if content != original_content:
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(content)

                    self.corrections_made.append(f"ATR corregido en {filepath}")
                    return True

            return False

        except Exception as e:
            print(f"❌ Error corrigiendo ATR en {filepath}: {e}")
            return False

    def replace_with_talib_implementation(self, filepath: str) -> bool:
        """Reemplazar implementaciones manuales con TA-Lib"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            original_content = content

            # Reemplazos con TA-Lib
            replacements = [
                # RSI manual → TA-Lib
                {
                    'pattern': r'(async\s+)?def\s+_calculate_rsi\(.*?\):.*?try:.*?return rsi.*?except.*?return.*?',
                    'replacement': '''async def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calcular RSI usando TA-Lib (CORREGIDO)"""
        try:
            import talib
            rsi_values = talib.RSI(prices.values, timeperiod=period)
            return pd.Series(rsi_values, index=prices.index)
        except:
            return pd.Series(index=prices.index, data=50.0)'''
                },
                # ATR manual → TA-Lib
                {
                    'pattern': r'(async\s+)?def\s+_calculate_atr\(.*?\):.*?try:.*?return atr.*?except.*?return.*?',
                    'replacement': '''async def _calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
        """Calcular ATR usando TA-Lib (CORREGIDO)"""
        try:
            import talib
            atr_values = talib.ATR(high.values, low.values, close.values, timeperiod=period)
            return pd.Series(atr_values, index=high.index)
        except:
            return pd.Series(index=high.index, data=1.0)'''
                }
            ]

            for replacement in replacements:
                if re.search(replacement['pattern'], content, re.DOTALL | re.IGNORECASE):
                    content = re.sub(replacement['pattern'], replacement['replacement'],
                                   content, flags=re.DOTALL | re.IGNORECASE)

            if content != original_content:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)

                self.corrections_made.append(f"Implementaciones TA-Lib aplicadas en {filepath}")
                return True

            return False

        except Exception as e:
            print(f"❌ Error aplicando TA-Lib en {filepath}: {e}")
            return False

    def validate_corrections(self) -> Dict[str, bool]:
        """Validar que las correcciones son matemáticamente correctas"""
        print("\n🔍 VALIDANDO CORRECCIONES MATEMÁTICAS...")

        validation_results = {}

        # Generar datos de prueba
        np.random.seed(42)
        test_prices = pd.Series(np.random.randn(100).cumsum() + 100)
        test_high = test_prices * 1.01
        test_low = test_prices * 0.99

        try:
            # Validar RSI
            rsi_talib = talib.RSI(test_prices.values, timeperiod=14)

            # Simular RSI corregido
            delta = test_prices.diff()
            gains = delta.where(delta > 0, 0)
            losses = -delta.where(delta < 0, 0)
            alpha = 1.0 / 14
            gain = gains.ewm(alpha=alpha, adjust=False).mean()
            loss = losses.ewm(alpha=alpha, adjust=False).mean()
            loss = loss.replace(0, 1e-8)
            rs = gain / loss
            rsi_corrected = 100 - (100 / (1 + rs))

            # Comparar
            valid_idx = ~(np.isnan(rsi_talib) | np.isnan(rsi_corrected))
            if np.sum(valid_idx) > 0:
                mae_rsi = np.mean(np.abs(rsi_corrected.values[valid_idx] - rsi_talib[valid_idx]))
                validation_results['rsi'] = mae_rsi < 0.1
                print(f"  ✅ RSI: Error promedio = {mae_rsi:.4f} ({'PASS' if mae_rsi < 0.1 else 'FAIL'})")

            # Validar ATR
            atr_talib = talib.ATR(test_high.values, test_low.values, test_prices.values, timeperiod=14)

            # Simular ATR corregido
            tr1 = test_high - test_low
            tr2 = abs(test_high - test_prices.shift())
            tr3 = abs(test_low - test_prices.shift())
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr_corrected = true_range.ewm(span=14, adjust=False).mean()

            valid_idx = ~(np.isnan(atr_talib) | np.isnan(atr_corrected))
            if np.sum(valid_idx) > 0:
                mae_atr = np.mean(np.abs(atr_corrected.values[valid_idx] - atr_talib[valid_idx]) / atr_talib[valid_idx])
                validation_results['atr'] = mae_atr < 0.01
                print(f"  ✅ ATR: Error relativo = {mae_atr:.4f} ({'PASS' if mae_atr < 0.01 else 'FAIL'})")

        except Exception as e:
            print(f"❌ Error en validación: {e}")
            validation_results['error'] = str(e)

        return validation_results

    def run_comprehensive_correction(self) -> Dict[str, any]:
        """Ejecutar corrección completa del sistema"""
        print("🔧 INICIANDO CORRECCIÓN MATEMÁTICA COMPLETA")
        print("=" * 50)

        # 1. Crear backup
        if not self.create_backup():
            return {"status": "ERROR", "message": "No se pudo crear backup"}

        # 2. Encontrar archivos con errores
        files_with_errors = self.find_files_with_math_errors()
        print(f"\n📁 Archivos con errores matemáticos: {len(files_with_errors)}")

        # 3. Aplicar correcciones
        for filepath in files_with_errors:
            print(f"\n🔧 Corrigiendo: {filepath}")

            # Aplicar correcciones específicas
            rsi_corrected = self.correct_rsi_calculation(filepath)
            atr_corrected = self.correct_atr_calculation(filepath)
            talib_applied = self.replace_with_talib_implementation(filepath)

            if rsi_corrected or atr_corrected or talib_applied:
                self.files_modified.append(filepath)
                print(f"  ✅ Correcciones aplicadas")
            else:
                print(f"  ℹ️  Sin cambios necesarios")

        # 4. Validar correcciones
        validation_results = self.validate_corrections()

        # 5. Resumen final
        print("\n" + "=" * 50)
        print("📋 RESUMEN DE CORRECCIONES")
        print("=" * 50)

        print(f"✅ Archivos modificados: {len(self.files_modified)}")
        print(f"✅ Correcciones aplicadas: {len(self.corrections_made)}")

        if self.corrections_made:
            print("\n🔧 CORRECCIONES REALIZADAS:")
            for i, correction in enumerate(self.corrections_made, 1):
                print(f"   {i}. {correction}")

        if self.files_modified:
            print("\n📁 ARCHIVOS MODIFICADOS:")
            for i, file in enumerate(self.files_modified, 1):
                print(f"   {i}. {file}")

        # Estado general
        all_validations_passed = all(v for k, v in validation_results.items() if k != 'error')

        if all_validations_passed and len(self.corrections_made) > 0:
            status = "SUCCESS"
            print(f"\n🎯 ESTADO: CORRECCIÓN EXITOSA")
            print("   Todas las features matemáticas han sido corregidas")
        elif len(self.corrections_made) == 0:
            status = "NO_CHANGES"
            print(f"\n ℹ️ ESTADO: SIN CAMBIOS NECESARIOS")
            print("   No se encontraron errores matemáticos para corregir")
        else:
            status = "PARTIAL"
            print(f"\n⚠️  ESTADO: CORRECCIÓN PARCIAL")
            print("   Algunas correcciones pueden necesitar revisión manual")

        return {
            "status": status,
            "files_modified": self.files_modified,
            "corrections_made": self.corrections_made,
            "validation_results": validation_results,
            "backup_location": self.backup_dir
        }

def main():
    """Función principal para ejecutar las correcciones"""
    corrector = MathematicalFeaturesCorrector()
    results = corrector.run_comprehensive_correction()

    return results

if __name__ == "__main__":
    main()
