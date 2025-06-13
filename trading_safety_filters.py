
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
