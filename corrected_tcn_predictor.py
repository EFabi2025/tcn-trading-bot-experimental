
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
