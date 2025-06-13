#!/usr/bin/env python3
"""
📊 ANALIZADOR DE CONDICIONES DE MERCADO
Prevenir compras en tendencias bajistas extremas
"""

import asyncio
import aiohttp
from datetime import datetime
from typing import Dict, List, Optional

class MarketConditionAnalyzer:
    """📊 Analizador de condiciones de mercado"""

    def __init__(self):
        self.base_url = "https://api.binance.com"

    async def analyze_market_condition(self, symbol: str) -> Dict:
        """🔍 Analizar condición actual del mercado para un símbolo"""

        try:
            # Obtener datos de mercado
            market_data = await self._get_comprehensive_market_data(symbol)

            # Analizar múltiples indicadores
            analysis = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'price_analysis': await self._analyze_price_action(market_data),
                'technical_analysis': await self._analyze_technical_indicators(market_data),
                'volume_analysis': await self._analyze_volume(market_data),
                'market_sentiment': await self._analyze_market_sentiment(market_data),
                'overall_condition': None,
                'buy_recommendation': None,
                'risk_level': None
            }

            # Determinar condición general
            analysis['overall_condition'] = self._determine_overall_condition(analysis)
            analysis['buy_recommendation'] = self._should_allow_buy(analysis)
            analysis['risk_level'] = self._calculate_risk_level(analysis)

            return analysis

        except Exception as e:
            print(f"❌ Error analizando condición de mercado para {symbol}: {e}")
            return self._get_safe_default_analysis(symbol)

    async def _get_comprehensive_market_data(self, symbol: str) -> Dict:
        """📊 Obtener datos completos de mercado"""

        async with aiohttp.ClientSession() as session:
            # 1. Ticker 24h
            ticker_24h = await self._get_ticker_24h(session, symbol)

            # 2. Klines recientes (1h, últimas 24 horas)
            klines_1h = await self._get_klines(session, symbol, "1h", 24)

            # 3. Klines recientes (4h, últimas 7 días)
            klines_4h = await self._get_klines(session, symbol, "4h", 42)  # 7 días * 6 velas/día

            # 4. Klines diarias (últimos 30 días)
            klines_1d = await self._get_klines(session, symbol, "1d", 30)

            return {
                'ticker_24h': ticker_24h,
                'klines_1h': klines_1h,
                'klines_4h': klines_4h,
                'klines_1d': klines_1d,
                'current_price': float(ticker_24h.get('lastPrice', 0)) if ticker_24h else 0
            }

    async def _get_ticker_24h(self, session: aiohttp.ClientSession, symbol: str) -> Dict:
        """📊 Obtener ticker 24h"""
        try:
            url = f"{self.base_url}/api/v3/ticker/24hr"
            params = {'symbol': symbol}

            async with session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
        except Exception as e:
            print(f"❌ Error obteniendo ticker 24h: {e}")

        return {}

    async def _get_klines(self, session: aiohttp.ClientSession, symbol: str,
                         interval: str, limit: int) -> List:
        """📊 Obtener klines"""
        try:
            url = f"{self.base_url}/api/v3/klines"
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': limit
            }

            async with session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
        except Exception as e:
            print(f"❌ Error obteniendo klines: {e}")

        return []

    async def _analyze_price_action(self, market_data: Dict) -> Dict:
        """📈 Analizar acción del precio"""

        ticker_24h = market_data.get('ticker_24h', {})
        klines_1d = market_data.get('klines_1d', [])

        analysis = {
            'change_24h_percent': float(ticker_24h.get('priceChangePercent', 0)),
            'change_7d_percent': 0,
            'change_30d_percent': 0,
            'trend_direction': 'UNKNOWN',
            'trend_strength': 'UNKNOWN'
        }

        # Calcular cambios de 7 y 30 días
        if len(klines_1d) >= 7:
            price_7d_ago = float(klines_1d[-7][4])  # Close price 7 días atrás
            current_price = market_data.get('current_price', 0)
            if price_7d_ago > 0:
                analysis['change_7d_percent'] = ((current_price - price_7d_ago) / price_7d_ago) * 100

        if len(klines_1d) >= 30:
            price_30d_ago = float(klines_1d[-30][4])  # Close price 30 días atrás
            current_price = market_data.get('current_price', 0)
            if price_30d_ago > 0:
                                 analysis['change_30d_percent'] = ((current_price - price_30d_ago) / price_30d_ago) * 100

         # Determinar tendencia - UMBRALES CORREGIDOS
         change_24h = analysis['change_24h_percent']
         change_7d = analysis['change_7d_percent']

         # Condiciones bajistas más sensibles
         if change_24h < -8:  # ETH -9.13% debe ser BEARISH_EXTREME
             analysis['trend_direction'] = 'BEARISH_EXTREME'
             analysis['trend_strength'] = 'VERY_STRONG'
         elif change_24h < -4:  # BTC -4.69% debe ser BEARISH_STRONG
             analysis['trend_direction'] = 'BEARISH_STRONG'
             analysis['trend_strength'] = 'STRONG'
         elif change_24h < -2:
             analysis['trend_direction'] = 'BEARISH_WEAK'
             analysis['trend_strength'] = 'WEAK'
        elif change_24h > 5 and change_7d > 10:
            analysis['trend_direction'] = 'BULLISH_EXTREME'
            analysis['trend_strength'] = 'VERY_STRONG'
        elif change_24h > 3 and change_7d > 5:
            analysis['trend_direction'] = 'BULLISH_STRONG'
            analysis['trend_strength'] = 'STRONG'
        elif change_24h > 1 and change_7d > 2:
            analysis['trend_direction'] = 'BULLISH_WEAK'
            analysis['trend_strength'] = 'WEAK'
        else:
            analysis['trend_direction'] = 'SIDEWAYS'
            analysis['trend_strength'] = 'NEUTRAL'

        return analysis

    async def _analyze_technical_indicators(self, market_data: Dict) -> Dict:
        """📊 Analizar indicadores técnicos"""

        klines_1h = market_data.get('klines_1h', [])

        if len(klines_1h) < 14:
            return {'rsi': 50, 'sma_trend': 'UNKNOWN', 'support_resistance': 'UNKNOWN'}

        # Calcular RSI simple
        closes = [float(k[4]) for k in klines_1h[-14:]]  # Últimos 14 closes
        rsi = self._calculate_simple_rsi(closes)

        # Analizar tendencia de SMA
        sma_short = sum([float(k[4]) for k in klines_1h[-5:]]) / 5  # SMA 5
        sma_long = sum([float(k[4]) for k in klines_1h[-14:]]) / 14  # SMA 14

        if sma_short > sma_long * 1.02:
            sma_trend = 'BULLISH'
        elif sma_short < sma_long * 0.98:
            sma_trend = 'BEARISH'
        else:
            sma_trend = 'NEUTRAL'

        return {
            'rsi': rsi,
            'sma_trend': sma_trend,
            'support_resistance': 'UNKNOWN'  # Implementar en futuras versiones
        }

    def _calculate_simple_rsi(self, prices: List[float], period: int = 14) -> float:
        """📊 Calcular RSI simple"""
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

    async def _analyze_volume(self, market_data: Dict) -> Dict:
        """📊 Analizar volumen"""

        ticker_24h = market_data.get('ticker_24h', {})
        klines_1h = market_data.get('klines_1h', [])

        volume_24h = float(ticker_24h.get('volume', 0))

        # Calcular volumen promedio de las últimas 24 horas
        if len(klines_1h) >= 24:
            volumes = [float(k[5]) for k in klines_1h[-24:]]  # Volume
            avg_volume = sum(volumes) / len(volumes)
            volume_ratio = volume_24h / avg_volume if avg_volume > 0 else 1
        else:
            volume_ratio = 1

        return {
            'volume_24h': volume_24h,
            'volume_ratio': volume_ratio,
            'volume_trend': 'HIGH' if volume_ratio > 1.5 else 'NORMAL' if volume_ratio > 0.5 else 'LOW'
        }

    async def _analyze_market_sentiment(self, market_data: Dict) -> Dict:
        """📊 Analizar sentimiento del mercado"""

        ticker_24h = market_data.get('ticker_24h', {})

        # Análisis básico basado en cambios de precio
        change_24h = float(ticker_24h.get('priceChangePercent', 0))

        if change_24h < -10:
            sentiment = 'PANIC_SELL'
        elif change_24h < -5:
            sentiment = 'FEAR'
        elif change_24h < -2:
            sentiment = 'BEARISH'
        elif change_24h > 10:
            sentiment = 'EUPHORIA'
        elif change_24h > 5:
            sentiment = 'GREED'
        elif change_24h > 2:
            sentiment = 'BULLISH'
        else:
            sentiment = 'NEUTRAL'

        return {
            'sentiment': sentiment,
            'fear_greed_index': max(0, min(100, 50 + change_24h * 2))  # Aproximación simple
        }

    def _determine_overall_condition(self, analysis: Dict) -> str:
        """🎯 Determinar condición general del mercado"""

        price_analysis = analysis.get('price_analysis', {})
        technical_analysis = analysis.get('technical_analysis', {})
        sentiment = analysis.get('market_sentiment', {})

        trend_direction = price_analysis.get('trend_direction', 'UNKNOWN')
        rsi = technical_analysis.get('rsi', 50)
        sentiment_value = sentiment.get('sentiment', 'NEUTRAL')

                 # Condiciones bajistas extremas - SIMPLIFICADO
         if trend_direction == 'BEARISH_EXTREME':
             return 'BEARISH_EXTREME'

         # Condiciones bajistas fuertes - SIMPLIFICADO
         elif trend_direction in ['BEARISH_STRONG', 'BEARISH_WEAK']:
             return 'BEARISH_STRONG'

        # Condiciones alcistas
        elif (trend_direction in ['BULLISH_STRONG', 'BULLISH_EXTREME'] and
              rsi > 60):
            return 'BULLISH_STRONG'

        # Condiciones neutrales
        else:
            return 'NEUTRAL'

    def _should_allow_buy(self, analysis: Dict) -> Dict:
        """🚨 Determinar si permitir compras"""

        overall_condition = analysis.get('overall_condition', 'UNKNOWN')
        price_analysis = analysis.get('price_analysis', {})

        # PROHIBIR compras en condiciones bajistas extremas
        if overall_condition == 'BEARISH_EXTREME':
            return {
                'allow_buy': False,
                'reason': 'BEARISH_EXTREME_CONDITION',
                'confidence': 0.95,
                'recommendation': 'WAIT_FOR_REVERSAL'
            }

        # PROHIBIR compras en condiciones bajistas fuertes
        elif overall_condition == 'BEARISH_STRONG':
            return {
                'allow_buy': False,
                'reason': 'BEARISH_STRONG_CONDITION',
                'confidence': 0.85,
                'recommendation': 'WAIT_FOR_STABILIZATION'
            }

        # PERMITIR compras con precaución en condiciones neutrales
        elif overall_condition == 'NEUTRAL':
            return {
                'allow_buy': True,
                'reason': 'NEUTRAL_CONDITION',
                'confidence': 0.60,
                'recommendation': 'PROCEED_WITH_CAUTION'
            }

        # PERMITIR compras en condiciones alcistas
        elif overall_condition == 'BULLISH_STRONG':
            return {
                'allow_buy': True,
                'reason': 'BULLISH_CONDITION',
                'confidence': 0.80,
                'recommendation': 'FAVORABLE_CONDITIONS'
            }

        # Por defecto, ser conservador
        else:
            return {
                'allow_buy': False,
                'reason': 'UNKNOWN_CONDITION',
                'confidence': 0.50,
                'recommendation': 'WAIT_FOR_CLARITY'
            }

    def _calculate_risk_level(self, analysis: Dict) -> str:
        """⚠️ Calcular nivel de riesgo"""

        overall_condition = analysis.get('overall_condition', 'UNKNOWN')
        price_analysis = analysis.get('price_analysis', {})
        volume_analysis = analysis.get('volume_analysis', {})

        change_24h = abs(price_analysis.get('change_24h_percent', 0))
        volume_trend = volume_analysis.get('volume_trend', 'NORMAL')

        # Riesgo muy alto
        if (overall_condition == 'BEARISH_EXTREME' or
            change_24h > 10 or
            volume_trend == 'HIGH'):
            return 'VERY_HIGH'

        # Riesgo alto
        elif (overall_condition == 'BEARISH_STRONG' or
              change_24h > 5):
            return 'HIGH'

        # Riesgo medio
        elif (overall_condition == 'NEUTRAL' or
              change_24h > 2):
            return 'MEDIUM'

        # Riesgo bajo
        else:
            return 'LOW'

    def _get_safe_default_analysis(self, symbol: str) -> Dict:
        """🛡️ Análisis por defecto seguro en caso de error"""
        return {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'overall_condition': 'UNKNOWN',
            'buy_recommendation': {
                'allow_buy': False,
                'reason': 'ERROR_OCCURRED',
                'confidence': 0.0,
                'recommendation': 'WAIT_FOR_ANALYSIS'
            },
            'risk_level': 'VERY_HIGH',
            'error': True
        }

async def test_market_analyzer():
    """🧪 Test del analizador de mercado"""
    print("🧪 TESTING ANALIZADOR DE CONDICIONES DE MERCADO")
    print("=" * 60)

    analyzer = MarketConditionAnalyzer()

    symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]

    for symbol in symbols:
        print(f"\n📊 Analizando {symbol}...")
        analysis = await analyzer.analyze_market_condition(symbol)

        print(f"  🎯 Condición general: {analysis['overall_condition']}")
        print(f"  📈 Cambio 24h: {analysis.get('price_analysis', {}).get('change_24h_percent', 0):.2f}%")
        print(f"  📊 RSI: {analysis.get('technical_analysis', {}).get('rsi', 0):.1f}")
        print(f"  🚨 Permitir BUY: {analysis['buy_recommendation']['allow_buy']}")
        print(f"  ⚠️ Razón: {analysis['buy_recommendation']['reason']}")
        print(f"  🎯 Recomendación: {analysis['buy_recommendation']['recommendation']}")
        print(f"  ⚠️ Nivel de riesgo: {analysis['risk_level']}")

if __name__ == "__main__":
    asyncio.run(test_market_analyzer())
