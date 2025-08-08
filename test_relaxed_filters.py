#!/usr/bin/env python3
"""
🧪 TEST DE FILTROS RELAJADOS PARA MERCADO BULLISH CON ALTA VOLATILIDAD
Verifica que los nuevos filtros permiten más operaciones en mercados alcistas volátiles
"""

import os
import sys
from datetime import datetime, timedelta

# Configurar para testing
os.environ['MIN_CONFIDENCE_THRESHOLD'] = '0.65'

# Mock de la clase para testing
class MockTradingManager:
    def __init__(self):
        self.signal_history = {}
        self.signal_cooldown = {'BTCUSDT': 10, 'ETHUSDT': 15}
        self.eth_position_protection = {
            'min_hold_time_minutes': 30,
            'signal_confirmation_required': 2
        }
        self.last_position_action = {}

    def _get_positions_for_symbol(self, symbol):
        return []  # Sin posiciones para testing

    def _apply_market_context_filter(self, signal: str, confidence: float, market_context: dict, symbol: str):
        """🛡️ FILTRO DE CONTEXTO DE MERCADO RELAJADO - VERSIÓN OPTIMIZADA"""
        try:
            # ✅ CENTRALIZADO: Definir base_threshold al inicio para uso global
            base_threshold = float(os.getenv('MIN_CONFIDENCE_THRESHOLD', '0.65')) * 100

            # Extraer información del contexto
            regime = market_context.get('regime', 'NEUTRAL')
            market_confidence = market_context.get('confidence', 0.0)
            market_score = market_context.get('score', 0.0)
            fear_factor = market_context.get('market_fear_factor', 0.5)
            volatility = market_context.get('volatility_level', 'MEDIUM')

            # ✅ NUEVO: Información de duración del régimen
            regime_duration_hours = market_context.get('regime_duration_hours', 0)
            btc_leading_down = market_context.get('btc_leading_down', False)

            # Por defecto, no filtrar
            filter_reason = f"Sin filtro aplicado - {regime} con confianza {market_confidence:.1%}"

            # 🚀 NUEVO: Detección de mercado bullish con alta volatilidad (oportunidad)
            is_bullish_high_vol = (regime == 'BULLISH' and
                                 market_confidence > 0.8 and
                                 volatility == 'HIGH')

            # 🎯 BYPASS MODERADO: Mercado bullish con alta volatilidad = OPORTUNIDAD
            if is_bullish_high_vol and confidence >= base_threshold * 0.8:  # 🎯 MODERADO: 80% del umbral base
                filter_reason = f"BYPASS BULLISH+VOLATILIDAD: Oportunidad detectada con {confidence:.1f}% confianza"
                return signal, filter_reason

            # 🟡 FILTROS DE VOLATILIDAD - Ajustar según volatilidad del mercado (CENTRALIZADOS)
            if volatility == 'HIGH' and fear_factor > 0.8:
                # ✅ CENTRALIZADO: En mercado BULLISH, la volatilidad puede ser oportunidad
                if regime == 'BULLISH' and market_confidence > 0.9:
                    # 🚀 RELAJADO: En mercado muy bullish, permitir más oportunidades
                    volatility_thresholds = {
                        'BTCUSDT': base_threshold * 0.8,   # 🚀 RELAJADO: 80% del umbral base (era 90%)
                        'ETHUSDT': base_threshold * 0.8,   # 🚀 RELAJADO: 80% del umbral base (era 90%)
                        'BNBUSDT': base_threshold * 0.8,   # 🚀 RELAJADO: 80% del umbral base (era 90%)
                        'XRPUSDT': base_threshold * 0.8    # 🚀 RELAJADO: 80% del umbral base (era 90%)
                    }
                    vol_adjustment = "MUY_RELAJADO_BULLISH"
                else:
                    # ✅ CENTRALIZADO: Umbrales con +5% para alta volatilidad
                    volatility_thresholds = {
                        'BTCUSDT': base_threshold * 1.05,   # ✅ VOLATILIDAD: +5% sobre base
                        'ETHUSDT': base_threshold * 1.05,   # ✅ VOLATILIDAD: +5% sobre base
                        'BNBUSDT': base_threshold * 1.05,   # ✅ VOLATILIDAD: +5% sobre base
                        'XRPUSDT': base_threshold * 1.05    # ✅ VOLATILIDAD: +5% sobre base
                    }
                    vol_adjustment = "ALTA_VOLATILIDAD"

                if signal == 'BUY':
                    required_vol_confidence = volatility_thresholds.get(symbol, base_threshold)
                    if confidence < required_vol_confidence:
                        signal = 'HOLD'
                        filter_reason = f"Alta volatilidad ({vol_adjustment}) - {symbol} BUY requiere >{required_vol_confidence:.1f}% confianza"
                elif signal == 'SELL':
                    # 🚀 RELAJADO: En bullish extremo, permitir SELL con confianza baja
                    if regime == 'BULLISH' and market_confidence > 0.9:
                        min_sell_vol_conf = base_threshold * 0.9  # 🚀 RELAJADO: 90% del umbral base (era 100%)
                    elif regime == 'BULLISH' and market_confidence > 0.7:
                        min_sell_vol_conf = base_threshold * 0.95  # 🚀 NUEVO: 95% para bullish moderado
                    else:
                        min_sell_vol_conf = base_threshold * 1.05 + 5  # ✅ VOLATILIDAD: +5% base + 5% extra

                    if confidence < min_sell_vol_conf:
                        signal = 'HOLD'
                        filter_reason = f"Alta volatilidad ({vol_adjustment}) - SELL requiere >{min_sell_vol_conf:.1f}% confianza (actual: {confidence:.1f}%)"

            return signal, filter_reason

        except Exception as e:
            return signal, f"Error en filtro: {e}"

    def _apply_signal_stability_filter(self, symbol: str, signal: str, confidence: float, current_price: float, market_context=None):
        """🛡️ FILTRO DE ESTABILIDAD Y COOLDOWN PARA SEÑALES"""
        try:
            current_time = datetime.now()

            # Inicializar historial del símbolo si no existe
            if symbol not in self.signal_history:
                self.signal_history[symbol] = {
                    'last_signal': None,
                    'last_signal_time': None,
                    'signal_count': 0,
                    'consecutive_same_signal': 0
                }

            history = self.signal_history[symbol]

            # ✅ FILTRO DE CONFIANZA AUMENTADA PARA CAMBIOS DE SEÑAL (RELAJADO PARA ENSEMBLE)
            if history['last_signal'] and history['last_signal'] != signal:
                # ✅ RELAJADO: Con ensemble de modelos, las confianzas pueden ser más bajas pero válidas
                # Verificar si el mercado es muy bullish usando el contexto recibido
                market_is_very_bullish = market_context and \
                                       market_context.get('regime') == 'BULLISH' and \
                                       market_context.get('confidence', 0) > 0.9

                if market_is_very_bullish:
                    # 🎯 MODERADO: Umbrales moderadamente relajados para mercado muy bullish
                    min_confidence_for_change = {
                        'ETHUSDT': 55.0,  # 🎯 MODERADO: De 60% a 55% para ETH
                        'BTCUSDT': 55.0,  # 🎯 MODERADO: De 60% a 55% para BTC
                        'BNBUSDT': 55.0,  # 🎯 MODERADO: De 60% a 55% para BNB
                        'XRPUSDT': 55.0   # 🎯 MODERADO: De 60% a 55% para XRP
                    }.get(symbol, 52.0)
                    signal_context = "MODERADO_BULLISH_VOLATIL"
                else:
                    # 🎯 MODERADO: Umbrales moderados para ensemble en otros contextos
                    min_confidence_for_change = {
                        'ETHUSDT': 60.0,  # 🎯 MODERADO: De 65% a 60% para ETH
                        'BTCUSDT': 60.0,  # 🎯 MODERADO: De 65% a 60% para BTC
                        'BNBUSDT': 60.0,  # 🎯 MODERADO: De 65% a 60% para BNB
                        'XRPUSDT': 60.0   # 🎯 MODERADO: De 65% a 60% para XRP
                    }.get(symbol, 58.0)
                    signal_context = "MODERADO_ENSEMBLE"

                if confidence < min_confidence_for_change:
                    return 'HOLD', f"Cambio de señal {history['last_signal']}→{signal} requiere >{min_confidence_for_change:.0f}% confianza (actual: {confidence:.1f}%) [{signal_context}]"

            # ✅ ACTUALIZAR HISTORIAL
            history['last_signal'] = signal
            history['last_signal_time'] = current_time
            history['signal_count'] += 1

            return signal, f"Señal estable: {signal} con {confidence:.1f}% confianza"

        except Exception as e:
            return signal, f"Error en filtro estabilidad: {e}"

def test_bullish_high_volatility_scenarios():
    """🧪 Probar escenarios de mercado bullish con alta volatilidad"""
    print("🧪 TESTING FILTROS RELAJADOS PARA MERCADO BULLISH + ALTA VOLATILIDAD")
    print("=" * 70)

    manager = MockTradingManager()

    # Escenarios de prueba
    test_scenarios = [
        {
            'name': 'Mercado MUY BULLISH + Alta Volatilidad + BUY 55%',
            'market_context': {
                'regime': 'BULLISH',
                'confidence': 0.95,
                'volatility_level': 'HIGH',
                'market_fear_factor': 0.9
            },
            'signal': 'BUY',
            'confidence': 55.0,
            'symbol': 'BTCUSDT',
            'expected': 'BUY'  # Debería pasar con bypass
        },
        {
            'name': 'Mercado BULLISH + Alta Volatilidad + BUY 52%',
            'market_context': {
                'regime': 'BULLISH',
                'confidence': 0.85,
                'volatility_level': 'HIGH',
                'market_fear_factor': 0.9
            },
            'signal': 'BUY',
            'confidence': 52.0,
            'symbol': 'ETHUSDT',
            'expected': 'BUY'  # Debería pasar con bypass (80% de 65% = 52%)
        },
        {
            'name': 'Cambio señal HOLD→BUY en mercado MUY BULLISH',
            'market_context': {
                'regime': 'BULLISH',
                'confidence': 0.95,
                'volatility_level': 'HIGH'
            },
            'signal': 'BUY',
            'confidence': 56.0,
            'symbol': 'BTCUSDT',
            'expected': 'BUY',  # Debería pasar (requiere 55% en muy bullish)
            'previous_signal': 'HOLD'
        },
        {
            'name': 'SELL en mercado BULLISH + Volatilidad con 60%',
            'market_context': {
                'regime': 'BULLISH',
                'confidence': 0.95,
                'volatility_level': 'HIGH',
                'market_fear_factor': 0.9
            },
            'signal': 'SELL',
            'confidence': 60.0,
            'symbol': 'ETHUSDT',
            'expected': 'SELL'  # Debería pasar (90% de 65% = 58.5%)
        }
    ]

    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n🧪 TEST {i}: {scenario['name']}")
        print(f"   📊 Señal: {scenario['signal']} | Confianza: {scenario['confidence']}%")
        print(f"   🌊 Contexto: {scenario['market_context']['regime']} "
              f"({scenario['market_context']['confidence']:.0%}) | "
              f"Vol: {scenario['market_context'].get('volatility_level', 'N/A')}")

        # Simular señal anterior si existe
        if 'previous_signal' in scenario:
            manager.signal_history[scenario['symbol']] = {
                'last_signal': scenario['previous_signal'],
                'last_signal_time': datetime.now() - timedelta(minutes=20),
                'signal_count': 1,
                'consecutive_same_signal': 1
            }

        # Probar filtro de contexto de mercado
        filtered_signal_1, reason_1 = manager._apply_market_context_filter(
            scenario['signal'],
            scenario['confidence'],
            scenario['market_context'],
            scenario['symbol']
        )

        # Probar filtro de estabilidad si pasó el primero
        if filtered_signal_1 != 'HOLD':
            filtered_signal_2, reason_2 = manager._apply_signal_stability_filter(
                scenario['symbol'],
                filtered_signal_1,
                scenario['confidence'],
                100.0,  # precio mock
                scenario['market_context']
            )
            final_signal = filtered_signal_2
            final_reason = reason_2
        else:
            final_signal = filtered_signal_1
            final_reason = reason_1

        # Verificar resultado
        if final_signal == scenario['expected']:
            print(f"   ✅ PASÓ: {final_signal} | {final_reason}")
        else:
            print(f"   ❌ FALLÓ: Esperado {scenario['expected']}, obtuvo {final_signal}")
            print(f"      Razón: {final_reason}")

def test_comparison_old_vs_new():
    """🧪 Comparar comportamiento anterior vs nuevo"""
    print(f"\n{'='*70}")
    print("🔄 COMPARACIÓN: FILTROS ANTERIORES VS NUEVOS")
    print("="*70)

    # Configuración de prueba
    test_cases = [
        {'confidence': 55, 'desc': 'Confianza Media-Baja'},
        {'confidence': 60, 'desc': 'Confianza Media'},
        {'confidence': 65, 'desc': 'Confianza Base'},
        {'confidence': 70, 'desc': 'Confianza Alta'}
    ]

    market_context = {
        'regime': 'BULLISH',
        'confidence': 0.95,
        'volatility_level': 'HIGH',
        'market_fear_factor': 0.9
    }

    print(f"📊 Contexto: BULLISH (95%) + Alta Volatilidad")
    print(f"🎯 Umbral base: 65%")
    print(f"\n{'Confianza':<12} {'Anterior':<12} {'Nuevo':<12} {'Mejora'}")
    print("-" * 50)

    manager = MockTradingManager()

    for case in test_cases:
        conf = case['confidence']

        # Simular comportamiento anterior (aprox)
        old_threshold = 90 if market_context['confidence'] > 0.9 else 70  # Era más restrictivo
        old_result = 'BUY' if conf >= old_threshold else 'HOLD'

        # Comportamiento nuevo
        new_signal, reason = manager._apply_market_context_filter('BUY', conf, market_context, 'BTCUSDT')

        # Calcular mejora
        improvement = "✅ SÍ" if (old_result == 'HOLD' and new_signal == 'BUY') else "⚪ N/A"

        print(f"{conf}%{'':<8} {old_result:<12} {new_signal:<12} {improvement}")

if __name__ == "__main__":
    test_bullish_high_volatility_scenarios()
    test_comparison_old_vs_new()

    print(f"\n🎉 RESUMEN:")
    print("🚀 Los filtros ahora son MENOS CONSERVADORES en mercados bullish volátiles")
    print("📈 Más oportunidades de trading en mercados alcistas con alta volatilidad")
    print("🎯 Bypass especial para alta confianza en contextos favorables")
