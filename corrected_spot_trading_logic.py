
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
