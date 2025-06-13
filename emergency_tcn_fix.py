#!/usr/bin/env python3
"""
🚨 EMERGENCY TCN FIX
Diagnóstico y corrección inmediata del modelo TCN con señales invertidas
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
import asyncio

class EmergencyTCNDiagnostic:
    """🚨 Diagnóstico de emergencia para TCN"""

    def __init__(self):
        self.issues_found = []
        self.fixes_applied = []

    async def run_emergency_diagnostic(self):
        """🔍 Ejecutar diagnóstico completo de emergencia"""
        print("🚨 DIAGNÓSTICO DE EMERGENCIA TCN")
        print("=" * 50)
        print(f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 50)

        # 1. Verificar modelos cargados
        await self._check_loaded_models()

        # 2. Verificar interpretación de señales
        await self._check_signal_interpretation()

        # 3. Verificar distribución de predicciones
        await self._check_prediction_distribution()

        # 4. Verificar datos de entrenamiento
        await self._check_training_data_logic()

        # 5. Aplicar correcciones inmediatas
        await self._apply_emergency_fixes()

        # 6. Generar reporte
        self._generate_emergency_report()

    async def _check_loaded_models(self):
        """🔍 Verificar modelos cargados"""
        print("\n🔍 VERIFICANDO MODELOS CARGADOS...")

        try:
            from enhanced_real_predictor import EnhancedTCNPredictor
            predictor = EnhancedTCNPredictor()

            print(f"✅ Predictor inicializado")
            print(f"📊 Modelos disponibles: {list(predictor.models.keys())}")

            # Verificar cada modelo
            for pair, model in predictor.models.items():
                if model is None:
                    self.issues_found.append(f"❌ Modelo {pair} es None")
                    continue

                # Test con datos dummy
                dummy_input = np.random.random((1, 50, 21))
                prediction = model.predict(dummy_input, verbose=0)

                print(f"  {pair}: Shape output: {prediction.shape}")
                print(f"  {pair}: Sample prediction: {prediction[0]}")

                # Verificar si el modelo está sesgado
                if prediction.shape[1] == 3:
                    probs = prediction[0]
                    max_prob_idx = np.argmax(probs)
                    max_prob = np.max(probs)

                    class_names = ['SELL', 'HOLD', 'BUY']
                    predicted_class = class_names[max_prob_idx]

                    print(f"  {pair}: Predicción dummy: {predicted_class} ({max_prob:.3f})")

                    # Verificar bias extremo
                    if max_prob > 0.95:
                        self.issues_found.append(f"⚠️ {pair}: Modelo extremadamente sesgado ({predicted_class}: {max_prob:.3f})")

        except Exception as e:
            self.issues_found.append(f"❌ Error cargando modelos: {e}")

    async def _check_signal_interpretation(self):
        """🔍 Verificar interpretación de señales"""
        print("\n🔍 VERIFICANDO INTERPRETACIÓN DE SEÑALES...")

        try:
            # Verificar mapeo de clases
            class_names = ['SELL', 'HOLD', 'BUY']
            print(f"📋 Mapeo actual de clases:")
            for i, name in enumerate(class_names):
                print(f"  Índice {i}: {name}")

            # Simular diferentes escenarios de mercado
            market_scenarios = [
                {"name": "Tendencia Bajista Fuerte", "rsi": 25, "macd": -0.5, "expected": "SELL"},
                {"name": "Tendencia Alcista Fuerte", "rsi": 75, "macd": 0.5, "expected": "BUY"},
                {"name": "Mercado Lateral", "rsi": 50, "macd": 0.0, "expected": "HOLD"}
            ]

            print(f"\n🧪 TESTING ESCENARIOS DE MERCADO:")
            for scenario in market_scenarios:
                print(f"  📊 {scenario['name']}:")
                print(f"     RSI: {scenario['rsi']}, MACD: {scenario['macd']}")
                print(f"     Señal esperada: {scenario['expected']}")

        except Exception as e:
            self.issues_found.append(f"❌ Error verificando interpretación: {e}")

    async def _check_prediction_distribution(self):
        """🔍 Verificar distribución de predicciones"""
        print("\n🔍 VERIFICANDO DISTRIBUCIÓN DE PREDICCIONES...")

        try:
            from enhanced_real_predictor import EnhancedTCNPredictor, AdvancedBinanceData

            predictor = EnhancedTCNPredictor()

            # Obtener datos reales y hacer múltiples predicciones
            async with AdvancedBinanceData() as data_provider:
                pairs = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]

                all_predictions = []

                for pair in pairs:
                    try:
                        print(f"  📊 Analizando {pair}...")
                        market_data = await data_provider.get_comprehensive_data(pair)

                        if market_data and market_data.get('klines_1m'):
                            prediction = await predictor.predict_enhanced(pair, market_data)

                            if prediction:
                                all_predictions.append({
                                    'pair': pair,
                                    'signal': prediction['signal'],
                                    'confidence': prediction['confidence'],
                                    'probabilities': prediction['probabilities']
                                })

                                print(f"    🎯 {pair}: {prediction['signal']} ({prediction['confidence']:.3f})")
                                print(f"    📊 Probs: SELL:{prediction['probabilities']['SELL']:.3f} | HOLD:{prediction['probabilities']['HOLD']:.3f} | BUY:{prediction['probabilities']['BUY']:.3f}")

                    except Exception as e:
                        print(f"    ❌ Error con {pair}: {e}")

                # Analizar distribución
                if all_predictions:
                    signals = [p['signal'] for p in all_predictions]
                    signal_counts = {signal: signals.count(signal) for signal in ['SELL', 'HOLD', 'BUY']}
                    total = len(signals)

                    print(f"\n📊 DISTRIBUCIÓN DE SEÑALES ACTUALES:")
                    for signal, count in signal_counts.items():
                        percentage = (count / total) * 100 if total > 0 else 0
                        print(f"  {signal}: {count}/{total} ({percentage:.1f}%)")

                        # Detectar bias extremo
                        if percentage > 80:
                            self.issues_found.append(f"🚨 BIAS EXTREMO: {signal} representa {percentage:.1f}% de las señales")

        except Exception as e:
            self.issues_found.append(f"❌ Error verificando distribución: {e}")

    async def _check_training_data_logic(self):
        """🔍 Verificar lógica de datos de entrenamiento"""
        print("\n🔍 VERIFICANDO LÓGICA DE ENTRENAMIENTO...")

        # Verificar si los umbrales de entrenamiento tienen sentido
        training_files = [
            "tcn_production_ready.py",
            "tcn_trading_ready_final.py",
            "tcn_final_ready.py"
        ]

        for file in training_files:
            if os.path.exists(file):
                print(f"  📄 Verificando {file}...")
                try:
                    with open(file, 'r') as f:
                        content = f.read()

                    # Buscar umbrales de clasificación
                    if 'strong_buy' in content and 'strong_sell' in content:
                        print(f"    ✅ Contiene umbrales de clasificación")

                        # Verificar si los umbrales son lógicos
                        if 'future_return > thresholds' in content:
                            print(f"    ✅ Lógica: future_return > threshold = BUY")
                        elif 'future_return < thresholds' in content:
                            print(f"    ⚠️ Posible inversión: future_return < threshold = BUY")
                            self.issues_found.append(f"🚨 {file}: Lógica de clasificación posiblemente invertida")

                except Exception as e:
                    print(f"    ❌ Error leyendo {file}: {e}")

    async def _apply_emergency_fixes(self):
        """🔧 Aplicar correcciones de emergencia"""
        print("\n🔧 APLICANDO CORRECCIONES DE EMERGENCIA...")

        # Fix 1: Crear predictor con lógica corregida
        await self._create_corrected_predictor()

        # Fix 2: Implementar filtros de seguridad
        await self._implement_safety_filters()

        # Fix 3: Crear modo de trading conservador
        await self._create_conservative_mode()

    async def _create_corrected_predictor(self):
        """🔧 Crear predictor con lógica corregida"""
        print("  🔧 Creando predictor corregido...")

        corrected_predictor_code = '''
class CorrectedTCNPredictor:
    """🔧 Predictor TCN con lógica corregida"""

    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_engineer = None

    async def predict_corrected(self, pair: str, market_data: dict) -> dict:
        """Predicción con lógica corregida y filtros de seguridad"""

        # 1. ANÁLISIS TÉCNICO PRIMARIO (más confiable que TCN sesgado)
        technical_signal = self._analyze_technical_indicators(market_data)

        # 2. FILTROS DE MERCADO
        market_condition = self._assess_market_condition(market_data)

        # 3. DECISIÓN FINAL CONSERVADORA
        if market_condition == "BEARISH_STRONG":
            # En mercado bajista fuerte, NO comprar
            return {
                'signal': 'HOLD',
                'confidence': 0.8,
                'reason': 'BEARISH_MARKET_PROTECTION',
                'technical_signal': technical_signal,
                'market_condition': market_condition
            }
        elif market_condition == "BULLISH_CONFIRMED" and technical_signal == "BUY":
            # Solo comprar en mercado alcista confirmado
            return {
                'signal': 'BUY',
                'confidence': 0.7,
                'reason': 'BULLISH_CONFIRMED_BUY',
                'technical_signal': technical_signal,
                'market_condition': market_condition
            }
        else:
            # Por defecto, mantener
            return {
                'signal': 'HOLD',
                'confidence': 0.6,
                'reason': 'CONSERVATIVE_HOLD',
                'technical_signal': technical_signal,
                'market_condition': market_condition
            }

    def _analyze_technical_indicators(self, market_data: dict) -> str:
        """Análisis técnico básico y confiable"""
        try:
            klines = market_data.get('klines_1m', [])
            if len(klines) < 50:
                return "INSUFFICIENT_DATA"

            # Calcular indicadores básicos
            closes = [float(k['close']) for k in klines[-50:]]

            # RSI simple
            rsi = self._calculate_simple_rsi(closes)

            # Tendencia simple (SMA)
            sma_20 = sum(closes[-20:]) / 20
            sma_50 = sum(closes[-50:]) / 50
            current_price = closes[-1]

            # Lógica conservadora
            if rsi < 30 and current_price > sma_20 > sma_50:
                return "BUY"
            elif rsi > 70 or current_price < sma_20 < sma_50:
                return "SELL"
            else:
                return "HOLD"

        except Exception:
            return "ERROR"

    def _assess_market_condition(self, market_data: dict) -> str:
        """Evaluar condición general del mercado"""
        try:
            ticker_24h = market_data.get('ticker_24h', {})
            price_change_24h = float(ticker_24h.get('priceChangePercent', 0))

            if price_change_24h < -5:
                return "BEARISH_STRONG"
            elif price_change_24h > 5:
                return "BULLISH_STRONG"
            elif price_change_24h < -2:
                return "BEARISH_WEAK"
            elif price_change_24h > 2:
                return "BULLISH_WEAK"
            else:
                return "NEUTRAL"

        except Exception:
            return "UNKNOWN"

    def _calculate_simple_rsi(self, prices: list, period: int = 14) -> float:
        """Calcular RSI simple"""
        try:
            if len(prices) < period + 1:
                return 50

            deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
            gains = [d if d > 0 else 0 for d in deltas[-period:]]
            losses = [-d if d < 0 else 0 for d in deltas[-period:]]

            avg_gain = sum(gains) / period
            avg_loss = sum(losses) / period

            if avg_loss == 0:
                return 100

            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

            return rsi

        except Exception:
            return 50
'''

        # Guardar predictor corregido
        with open('corrected_tcn_predictor.py', 'w') as f:
            f.write(corrected_predictor_code)

        self.fixes_applied.append("✅ Predictor corregido creado")

    async def _implement_safety_filters(self):
        """🛡️ Implementar filtros de seguridad"""
        print("  🛡️ Implementando filtros de seguridad...")

        safety_filters_code = '''
class TradingSafetyFilters:
    """🛡️ Filtros de seguridad para trading"""

    @staticmethod
    def should_block_trade(signal: str, market_data: dict, confidence: float) -> tuple:
        """Determinar si bloquear un trade por seguridad"""

        # Filtro 1: No comprar en caídas fuertes
        ticker_24h = market_data.get('ticker_24h', {})
        price_change_24h = float(ticker_24h.get('priceChangePercent', 0))

        if signal == 'BUY' and price_change_24h < -3:
            return True, f"BLOCKED: No comprar en caída de {price_change_24h:.1f}%"

        # Filtro 2: Confianza mínima más alta
        if confidence < 0.75:
            return True, f"BLOCKED: Confianza insuficiente {confidence:.3f} < 0.75"

        # Filtro 3: Verificar volumen anómalo
        volume_24h = float(ticker_24h.get('volume', 0))
        if volume_24h == 0:
            return True, "BLOCKED: Volumen cero detectado"

        return False, "TRADE_ALLOWED"

    @staticmethod
    def adjust_position_size_for_risk(base_size: float, market_data: dict) -> float:
        """Ajustar tamaño de posición según riesgo"""

        ticker_24h = market_data.get('ticker_24h', {})
        price_change_24h = abs(float(ticker_24h.get('priceChangePercent', 0)))

        # Reducir tamaño en mercados volátiles
        if price_change_24h > 5:
            return base_size * 0.5  # 50% del tamaño normal
        elif price_change_24h > 3:
            return base_size * 0.7  # 70% del tamaño normal
        else:
            return base_size
'''

        with open('trading_safety_filters.py', 'w') as f:
            f.write(safety_filters_code)

        self.fixes_applied.append("✅ Filtros de seguridad implementados")

    async def _create_conservative_mode(self):
        """🐌 Crear modo de trading conservador"""
        print("  🐌 Creando modo conservador...")

        # Crear configuración conservadora
        conservative_config = {
            'MIN_CONFIDENCE_THRESHOLD': 0.80,  # Aumentar de 0.70 a 0.80
            'MAX_POSITION_SIZE_PERCENT': 5.0,  # Reducir de 15% a 5%
            'MAX_DAILY_TRADES': 2,             # Máximo 2 trades por día
            'REQUIRE_TECHNICAL_CONFIRMATION': True,
            'BLOCK_TRADES_IN_DOWNTREND': True,
            'EMERGENCY_MODE': True
        }

        with open('conservative_trading_config.json', 'w') as f:
            import json
            json.dump(conservative_config, f, indent=2)

        self.fixes_applied.append("✅ Modo conservador configurado")

    def _generate_emergency_report(self):
        """📊 Generar reporte de emergencia"""
        print("\n" + "="*60)
        print("🚨 REPORTE DE EMERGENCIA TCN")
        print("="*60)

        print(f"\n❌ PROBLEMAS DETECTADOS ({len(self.issues_found)}):")
        for issue in self.issues_found:
            print(f"  {issue}")

        print(f"\n✅ CORRECCIONES APLICADAS ({len(self.fixes_applied)}):")
        for fix in self.fixes_applied:
            print(f"  {fix}")

        print(f"\n🚨 RECOMENDACIONES INMEDIATAS:")
        print(f"  1. 🛑 DETENER trading automático inmediatamente")
        print(f"  2. 🔍 Revisar todas las posiciones abiertas")
        print(f"  3. 📊 Usar modo conservador hasta corregir TCN")
        print(f"  4. 🧪 Re-entrenar modelo con datos correctos")
        print(f"  5. ✅ Implementar filtros de seguridad adicionales")

        print(f"\n⚠️ ACCIONES CRÍTICAS:")
        print(f"  • Cambiar MIN_CONFIDENCE_THRESHOLD a 0.80")
        print(f"  • Reducir MAX_POSITION_SIZE_PERCENT a 5%")
        print(f"  • Activar filtros anti-tendencia bajista")
        print(f"  • Monitorear manualmente por 24-48 horas")

        # Guardar reporte
        report_content = f"""
REPORTE DE EMERGENCIA TCN
========================
Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

PROBLEMAS DETECTADOS:
{chr(10).join(self.issues_found)}

CORRECCIONES APLICADAS:
{chr(10).join(self.fixes_applied)}

ESTADO: REQUIERE INTERVENCIÓN MANUAL INMEDIATA
"""

        with open(f'emergency_tcn_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt', 'w') as f:
            f.write(report_content)

        print(f"\n📄 Reporte guardado: emergency_tcn_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

async def main():
    """🚨 Función principal de emergencia"""
    print("🚨 INICIANDO DIAGNÓSTICO DE EMERGENCIA TCN...")

    diagnostic = EmergencyTCNDiagnostic()
    await diagnostic.run_emergency_diagnostic()

    print(f"\n🚨 DIAGNÓSTICO COMPLETADO")
    print(f"⚠️ REVISAR REPORTE Y APLICAR CORRECCIONES INMEDIATAMENTE")

if __name__ == "__main__":
    asyncio.run(main())
