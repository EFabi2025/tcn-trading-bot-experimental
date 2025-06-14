#!/usr/bin/env python3
"""
🔍 AUDITOR MATEMÁTICO DE FEATURES TÉCNICAS
==========================================

Verifica la correctitud matemática de todos los indicadores técnicos
utilizados en el sistema de trading TCN.

Incluye:
- Verificación de fórmulas RSI, MACD, Bollinger Bands
- Validación de cálculos de volatilidad y momentum
- Detección de errores matemáticos comunes
- Comparación con implementaciones de referencia
"""

import numpy as np
import pandas as pd
import talib
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class MathematicalFeaturesAuditor:
    """
    🔍 Auditor matemático para verificar correctitud de features técnicas
    """

    def __init__(self):
        self.errors_found = []
        self.warnings_found = []
        self.validations_passed = []

    def generate_test_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """Generar datos de prueba realistas para validación"""
        np.random.seed(42)

        # Generar precio base con tendencia y volatilidad realista
        base_price = 45000
        trend = np.cumsum(np.random.normal(0, 0.001, n_samples))
        noise = np.random.normal(0, 0.02, n_samples)

        # Precio de cierre con comportamiento realista
        close_prices = base_price * np.exp(trend + noise)

        # Generar OHLCV coherente
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='5T'),
            'close': close_prices
        })

        # High y Low coherentes con close
        data['high'] = data['close'] * (1 + np.abs(np.random.normal(0, 0.005, n_samples)))
        data['low'] = data['close'] * (1 - np.abs(np.random.normal(0, 0.005, n_samples)))
        data['open'] = data['close'].shift(1).fillna(data['close'].iloc[0])

        # Volumen correlacionado con volatilidad
        volatility = data['close'].pct_change().rolling(20).std()
        data['volume'] = np.random.lognormal(8, 0.3, n_samples) * (1 + volatility * 10).fillna(1)

        return data.set_index('timestamp')

    def audit_rsi_calculation(self, data: pd.DataFrame) -> Dict[str, any]:
        """🔍 Auditar cálculo de RSI"""
        print("\n🔍 AUDITANDO RSI...")

        close = data['close'].values
        period = 14

        # === IMPLEMENTACIÓN TALIB ===
        rsi_talib = talib.RSI(close, timeperiod=period)

        # === IMPLEMENTACIÓN SISTEMA ACTUAL ===
        def calculate_rsi_system(prices: pd.Series, period: int = 14) -> pd.Series:
            """Implementación actual del sistema"""
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))

        rsi_system = calculate_rsi_system(data['close'], period)

        # === COMPARACIONES ===
        valid_idx = ~(np.isnan(rsi_talib) | np.isnan(rsi_system))

        if np.sum(valid_idx) == 0:
            self.errors_found.append("RSI: No hay valores válidos para comparar")
            return {"status": "ERROR", "details": "Sin valores válidos"}

        rsi_talib_clean = rsi_talib[valid_idx]
        rsi_system_clean = rsi_system.values[valid_idx]

        # Métricas de error
        mae_system_talib = np.mean(np.abs(rsi_system_clean - rsi_talib_clean))
        max_error_talib = np.max(np.abs(rsi_system_clean - rsi_talib_clean))

        results = {
            "status": "PASS",
            "mae_vs_talib": mae_system_talib,
            "max_error_talib": max_error_talib,
            "samples_compared": np.sum(valid_idx)
        }

        # Criterios de validación
        if mae_system_talib > 2.0:
            self.errors_found.append(f"RSI: Error promedio vs TA-Lib muy alto: {mae_system_talib:.2f}")
            results["status"] = "ERROR"
        elif mae_system_talib > 0.5:
            self.warnings_found.append(f"RSI: Error promedio vs TA-Lib moderado: {mae_system_talib:.2f}")
            results["status"] = "WARNING"

        if max_error_talib > 10.0:
            self.errors_found.append(f"RSI: Error máximo vs TA-Lib muy alto: {max_error_talib:.2f}")
            results["status"] = "ERROR"

        if np.any(rsi_system_clean < 0) or np.any(rsi_system_clean > 100):
            self.errors_found.append("RSI: Valores fuera del rango válido (0-100)")
            results["status"] = "ERROR"

        if results["status"] == "PASS":
            self.validations_passed.append("RSI: Cálculo matemáticamente correcto")

        print(f"  📊 Error promedio vs TA-Lib: {mae_system_talib:.4f}")
        print(f"  📊 Error máximo vs TA-Lib: {max_error_talib:.4f}")
        print(f"  📊 Muestras comparadas: {np.sum(valid_idx)}")
        print(f"  ✅ Estado: {results['status']}")

        return results

    def audit_macd_calculation(self, data: pd.DataFrame) -> Dict[str, any]:
        """🔍 Auditar cálculo de MACD"""
        print("\n🔍 AUDITANDO MACD...")

        close = data['close'].values

        # === IMPLEMENTACIÓN TALIB ===
        macd_talib, signal_talib, histogram_talib = talib.MACD(close)

        # === IMPLEMENTACIÓN SISTEMA ===
        def calculate_macd_system(prices: pd.Series):
            ema_12 = prices.ewm(span=12).mean()
            ema_26 = prices.ewm(span=26).mean()
            macd = ema_12 - ema_26
            signal_line = macd.ewm(span=9).mean()
            histogram = macd - signal_line
            return macd, signal_line, histogram

        macd_system, signal_system, histogram_system = calculate_macd_system(data['close'])

        # === COMPARACIONES ===
        valid_idx = ~(np.isnan(macd_talib) | np.isnan(macd_system))

        if np.sum(valid_idx) == 0:
            self.errors_found.append("MACD: No hay valores válidos para comparar")
            return {"status": "ERROR", "details": "Sin valores válidos"}

        # Normalizar por precio para comparación relativa
        price_norm = data['close'].values[valid_idx]

        macd_talib_norm = macd_talib[valid_idx] / price_norm
        macd_system_norm = macd_system.values[valid_idx] / price_norm

        signal_talib_norm = signal_talib[valid_idx] / price_norm
        signal_system_norm = signal_system.values[valid_idx] / price_norm

        # Errores relativos
        mae_macd = np.mean(np.abs(macd_system_norm - macd_talib_norm))
        mae_signal = np.mean(np.abs(signal_system_norm - signal_talib_norm))

        results = {
            "status": "PASS",
            "mae_macd": mae_macd,
            "mae_signal": mae_signal,
            "samples_compared": np.sum(valid_idx)
        }

        # Validaciones
        if mae_macd > 0.001:  # Error relativo > 0.1%
            self.errors_found.append(f"MACD: Error promedio muy alto: {mae_macd:.6f}")
            results["status"] = "ERROR"
        elif mae_macd > 0.0005:
            self.warnings_found.append(f"MACD: Error promedio moderado: {mae_macd:.6f}")
            results["status"] = "WARNING"

        if mae_signal > 0.001:
            self.errors_found.append(f"MACD Signal: Error promedio muy alto: {mae_signal:.6f}")
            results["status"] = "ERROR"

        if results["status"] == "PASS":
            self.validations_passed.append("MACD: Cálculo matemáticamente correcto")

        print(f"  📊 Error MACD (relativo): {mae_macd:.6f}")
        print(f"  📊 Error Signal (relativo): {mae_signal:.6f}")
        print(f"  📊 Muestras comparadas: {np.sum(valid_idx)}")
        print(f"  ✅ Estado: {results['status']}")

        return results

    def audit_bollinger_bands(self, data: pd.DataFrame) -> Dict[str, any]:
        """🔍 Auditar cálculo de Bollinger Bands"""
        print("\n🔍 AUDITANDO BOLLINGER BANDS...")

        close = data['close'].values

        # === IMPLEMENTACIÓN TALIB ===
        bb_upper_talib, bb_middle_talib, bb_lower_talib = talib.BBANDS(close, timeperiod=20)

        # === IMPLEMENTACIÓN SISTEMA ===
        def calculate_bollinger_system(prices: pd.Series):
            bb_ma = prices.rolling(20).mean()
            bb_std = prices.rolling(20).std()
            bb_upper = bb_ma + (bb_std * 2)
            bb_lower = bb_ma - (bb_std * 2)
            return bb_upper, bb_ma, bb_lower

        bb_upper_system, bb_middle_system, bb_lower_system = calculate_bollinger_system(data['close'])

        # === COMPARACIONES ===
        valid_idx = ~(np.isnan(bb_upper_talib) | np.isnan(bb_upper_system))

        if np.sum(valid_idx) == 0:
            self.errors_found.append("Bollinger Bands: No hay valores válidos para comparar")
            return {"status": "ERROR", "details": "Sin valores válidos"}

        # Errores relativos
        price_base = bb_middle_talib[valid_idx]
        mae_upper_rel = np.mean(np.abs(bb_upper_system.values[valid_idx] - bb_upper_talib[valid_idx]) / price_base)
        mae_middle_rel = np.mean(np.abs(bb_middle_system.values[valid_idx] - bb_middle_talib[valid_idx]) / price_base)
        mae_lower_rel = np.mean(np.abs(bb_lower_system.values[valid_idx] - bb_lower_talib[valid_idx]) / price_base)

        results = {
            "status": "PASS",
            "mae_upper_rel": mae_upper_rel,
            "mae_middle_rel": mae_middle_rel,
            "mae_lower_rel": mae_lower_rel,
            "samples_compared": np.sum(valid_idx)
        }

        # Validaciones
        if mae_upper_rel > 0.0001 or mae_lower_rel > 0.0001:
            self.errors_found.append(f"Bollinger Bands: Error relativo alto - Upper: {mae_upper_rel:.6f}, Lower: {mae_lower_rel:.6f}")
            results["status"] = "ERROR"

        if results["status"] == "PASS":
            self.validations_passed.append("Bollinger Bands: Cálculo matemáticamente correcto")

        print(f"  📊 Error Upper (relativo): {mae_upper_rel:.6f}")
        print(f"  📊 Error Middle (relativo): {mae_middle_rel:.6f}")
        print(f"  📊 Error Lower (relativo): {mae_lower_rel:.6f}")
        print(f"  ✅ Estado: {results['status']}")

        return results

    def audit_williams_r(self, data: pd.DataFrame) -> Dict[str, any]:
        """🔍 Auditar Williams %R"""
        print("\n🔍 AUDITANDO WILLIAMS %R...")

        high = data['high'].values
        low = data['low'].values
        close = data['close'].values

        # === IMPLEMENTACIÓN TALIB ===
        willr_talib = talib.WILLR(high, low, close, timeperiod=14)

        # === IMPLEMENTACIÓN SISTEMA ===
        def calculate_williams_r_system(high_s, low_s, close_s):
            highest_high = pd.Series(high_s).rolling(14).max()
            lowest_low = pd.Series(low_s).rolling(14).min()
            willr = -100 * (highest_high - close_s) / (highest_high - lowest_low)
            return willr.values

        willr_system = calculate_williams_r_system(high, low, close)

        # === COMPARACIONES ===
        valid_idx = ~(np.isnan(willr_talib) | np.isnan(willr_system))

        if np.sum(valid_idx) == 0:
            self.errors_found.append("Williams %R: No hay valores válidos para comparar")
            return {"status": "ERROR", "details": "Sin valores válidos"}

        mae = np.mean(np.abs(willr_system[valid_idx] - willr_talib[valid_idx]))
        max_error = np.max(np.abs(willr_system[valid_idx] - willr_talib[valid_idx]))

        results = {
            "status": "PASS",
            "mae": mae,
            "max_error": max_error,
            "samples_compared": np.sum(valid_idx)
        }

        # Validaciones
        if mae > 1.0:
            self.errors_found.append(f"Williams %R: Error promedio alto: {mae:.2f}")
            results["status"] = "ERROR"

        # Verificar rango válido (-100 a 0)
        willr_clean = willr_system[valid_idx]
        if np.any(willr_clean < -100) or np.any(willr_clean > 0):
            self.errors_found.append("Williams %R: Valores fuera del rango válido (-100 a 0)")
            results["status"] = "ERROR"

        if results["status"] == "PASS":
            self.validations_passed.append("Williams %R: Cálculo matemáticamente correcto")

        print(f"  📊 Error promedio vs TA-Lib: {mae:.4f}")
        print(f"  📊 Error máximo: {max_error:.4f}")
        print(f"  ✅ Estado: {results['status']}")

        return results

    def audit_momentum_calculations(self, data: pd.DataFrame) -> Dict[str, any]:
        """🔍 Auditar cálculos de momentum"""
        print("\n🔍 AUDITANDO MOMENTUM...")

        close = data['close']

        # === IMPLEMENTACIONES SISTEMA ===
        momentum_5 = close / close.shift(5)  # Sistema actual
        momentum_10 = close / close.shift(10)  # Sistema actual

        # === VERIFICACIONES ===
        results = {"status": "PASS", "checks": []}

        # 1. Verificar que momentum > 0 (precios siempre positivos)
        momentum_5_clean = momentum_5.dropna()
        momentum_10_clean = momentum_10.dropna()

        if np.any(momentum_5_clean <= 0) or np.any(momentum_10_clean <= 0):
            self.errors_found.append("Momentum: Valores negativos o cero encontrados")
            results["status"] = "ERROR"
        else:
            results["checks"].append("✅ Momentum siempre positivo")

        # 2. Verificar rangos razonables (0.5 a 2.0 para crypto)
        momentum_5_mean = momentum_5_clean.mean()
        momentum_10_mean = momentum_10_clean.mean()

        if momentum_5_mean < 0.5 or momentum_5_mean > 2.0:
            self.warnings_found.append(f"Momentum 5: Media fuera de rango razonable: {momentum_5_mean:.4f}")

        if momentum_10_mean < 0.5 or momentum_10_mean > 2.0:
            self.warnings_found.append(f"Momentum 10: Media fuera de rango razonable: {momentum_10_mean:.4f}")

        results["momentum_5_mean"] = momentum_5_mean
        results["momentum_10_mean"] = momentum_10_mean

        if results["status"] == "PASS":
            self.validations_passed.append("Momentum: Cálculos matemáticamente correctos")

        print(f"  📊 Momentum 5 promedio: {momentum_5_mean:.4f}")
        print(f"  📊 Momentum 10 promedio: {momentum_10_mean:.4f}")
        print(f"  ✅ Estado: {results['status']}")

        return results

    def audit_volatility_calculations(self, data: pd.DataFrame) -> Dict[str, any]:
        """🔍 Auditar cálculos de volatilidad"""
        print("\n🔍 AUDITANDO VOLATILIDAD...")

        # === IMPLEMENTACIÓN SISTEMA ===
        price_change = data['close'].pct_change()
        volatility = price_change.rolling(10).std()

        # === VERIFICACIONES ===
        results = {"status": "PASS", "checks": []}

        # 1. Verificar que volatilidad es positiva
        vol_clean = volatility.dropna()

        if np.any(vol_clean < 0):
            self.errors_found.append("Volatilidad: Valores negativos encontrados")
            results["status"] = "ERROR"
        else:
            results["checks"].append("✅ Volatilidad siempre positiva")

        # 2. Verificar rangos razonables para crypto (0.001 a 0.1)
        vol_mean = vol_clean.mean()
        vol_max = vol_clean.max()

        if vol_mean > 0.1:
            self.warnings_found.append(f"Volatilidad: Promedio muy alto: {vol_mean:.4f}")

        if vol_max > 0.5:
            self.warnings_found.append(f"Volatilidad: Máximo muy alto: {vol_max:.4f}")

        results["vol_mean"] = vol_mean
        results["vol_max"] = vol_max

        if results["status"] == "PASS":
            self.validations_passed.append("Volatilidad: Cálculos matemáticamente correctos")

        print(f"  📊 Volatilidad promedio: {vol_mean:.4f}")
        print(f"  📊 Volatilidad máxima: {vol_max:.4f}")
        print(f"  ✅ Estado: {results['status']}")

        return results

    def audit_atr_calculation(self, data: pd.DataFrame) -> Dict[str, any]:
        """🔍 Auditar cálculo de ATR"""
        print("\n🔍 AUDITANDO ATR...")

        high = data['high'].values
        low = data['low'].values
        close = data['close'].values

        # === IMPLEMENTACIÓN TALIB ===
        atr_talib = talib.ATR(high, low, close, timeperiod=14)

        # === IMPLEMENTACIÓN SISTEMA ===
        def calculate_atr_system(high_s, low_s, close_s):
            tr1 = pd.Series(high_s) - pd.Series(low_s)
            tr2 = abs(pd.Series(high_s) - pd.Series(close_s).shift())
            tr3 = abs(pd.Series(low_s) - pd.Series(close_s).shift())
            atr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(14).mean()
            return atr.values

        atr_system = calculate_atr_system(high, low, close)

        # === COMPARACIONES ===
        valid_idx = ~(np.isnan(atr_talib) | np.isnan(atr_system))

        if np.sum(valid_idx) == 0:
            self.errors_found.append("ATR: No hay valores válidos para comparar")
            return {"status": "ERROR", "details": "Sin valores válidos"}

        # Error relativo
        atr_talib_clean = atr_talib[valid_idx]
        atr_system_clean = atr_system[valid_idx]

        mae_rel = np.mean(np.abs(atr_system_clean - atr_talib_clean) / atr_talib_clean)
        max_error_rel = np.max(np.abs(atr_system_clean - atr_talib_clean) / atr_talib_clean)

        results = {
            "status": "PASS",
            "mae_rel": mae_rel,
            "max_error_rel": max_error_rel,
            "samples_compared": np.sum(valid_idx)
        }

        # Validaciones
        if mae_rel > 0.01:  # Error relativo > 1%
            self.errors_found.append(f"ATR: Error relativo promedio alto: {mae_rel:.4f}")
            results["status"] = "ERROR"
        elif mae_rel > 0.005:
            self.warnings_found.append(f"ATR: Error relativo moderado: {mae_rel:.4f}")
            results["status"] = "WARNING"

        if results["status"] == "PASS":
            self.validations_passed.append("ATR: Cálculo matemáticamente correcto")

        print(f"  📊 Error relativo promedio: {mae_rel:.4f}")
        print(f"  📊 Error relativo máximo: {max_error_rel:.4f}")
        print(f"  ✅ Estado: {results['status']}")

        return results

    def run_comprehensive_audit(self) -> Dict[str, any]:
        """🔍 Ejecutar auditoría matemática completa"""
        print("🔍 INICIANDO AUDITORÍA MATEMÁTICA COMPLETA DE FEATURES")
        print("=" * 60)

        # Generar datos de prueba
        test_data = self.generate_test_data(1000)
        print(f"📊 Datos de prueba generados: {len(test_data)} muestras")

        # Ejecutar todas las auditorías
        audit_results = {}

        try:
            audit_results['rsi'] = self.audit_rsi_calculation(test_data)
            audit_results['macd'] = self.audit_macd_calculation(test_data)
            audit_results['bollinger'] = self.audit_bollinger_bands(test_data)
            audit_results['williams_r'] = self.audit_williams_r(test_data)
            audit_results['momentum'] = self.audit_momentum_calculations(test_data)
            audit_results['volatility'] = self.audit_volatility_calculations(test_data)
            audit_results['atr'] = self.audit_atr_calculation(test_data)

        except Exception as e:
            self.errors_found.append(f"Error crítico en auditoría: {str(e)}")

        # === RESUMEN FINAL ===
        print("\n" + "=" * 60)
        print("📋 RESUMEN DE AUDITORÍA MATEMÁTICA")
        print("=" * 60)

        total_errors = len(self.errors_found)
        total_warnings = len(self.warnings_found)
        total_passed = len(self.validations_passed)

        print(f"✅ Validaciones pasadas: {total_passed}")
        print(f"⚠️  Advertencias: {total_warnings}")
        print(f"❌ Errores críticos: {total_errors}")

        if total_errors > 0:
            print(f"\n❌ ERRORES CRÍTICOS ENCONTRADOS:")
            for i, error in enumerate(self.errors_found, 1):
                print(f"   {i}. {error}")

        if total_warnings > 0:
            print(f"\n⚠️  ADVERTENCIAS:")
            for i, warning in enumerate(self.warnings_found, 1):
                print(f"   {i}. {warning}")

        if total_passed > 0:
            print(f"\n✅ VALIDACIONES EXITOSAS:")
            for i, validation in enumerate(self.validations_passed, 1):
                print(f"   {i}. {validation}")

        # Determinar estado general
        if total_errors > 0:
            overall_status = "CRÍTICO"
            print(f"\n🚨 ESTADO GENERAL: {overall_status}")
            print("   Se requiere corrección inmediata de errores matemáticos")
        elif total_warnings > 0:
            overall_status = "ADVERTENCIA"
            print(f"\n⚠️  ESTADO GENERAL: {overall_status}")
            print("   Se recomienda revisar las advertencias")
        else:
            overall_status = "EXCELENTE"
            print(f"\n🎯 ESTADO GENERAL: {overall_status}")
            print("   Todas las features son matemáticamente correctas")

        return {
            "overall_status": overall_status,
            "errors": self.errors_found,
            "warnings": self.warnings_found,
            "validations_passed": self.validations_passed,
            "audit_results": audit_results,
            "summary": {
                "total_errors": total_errors,
                "total_warnings": total_warnings,
                "total_passed": total_passed
            }
        }

def main():
    """Función principal para ejecutar la auditoría"""
    auditor = MathematicalFeaturesAuditor()
    results = auditor.run_comprehensive_audit()

    return results

if __name__ == "__main__":
    main()
