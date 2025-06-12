"""
Servicio de gestión de riesgos para trading automatizado.

Implementa validaciones de riesgo, cálculo de posiciones y
circuit breakers para proteger el capital.
"""
import asyncio
from typing import List, Dict, Any, Optional
from decimal import Decimal, ROUND_DOWN
from datetime import datetime, timedelta
from dataclasses import dataclass

from ..interfaces.trading_interfaces import (
    IRiskManager, OrderRequest, Balance, TradingSignal,
    OrderSide
)
from ..core.config import get_settings
from ..core.logging_config import get_trading_logger
from ..database.models import RiskEvent


@dataclass
class RiskCheckResult:
    """Resultado de una verificación de riesgo."""
    passed: bool
    reason: Optional[str] = None
    risk_score: float = 0.0
    recommended_action: Optional[str] = None


class RiskManager(IRiskManager):
    """
    Gestor de riesgos profesional para trading automatizado.

    Implementa múltiples capas de validación de riesgo:
    - Validación de balance y posición
    - Límites diarios de pérdida
    - Circuit breakers
    - Análisis de correlación de posiciones
    """

    def __init__(self):
        """Inicializa el gestor de riesgos."""
        self._settings = get_settings()
        self._logger = get_trading_logger(__name__)

        # Estado interno para tracking
        self._daily_pnl: Dict[str, Decimal] = {}
        self._position_sizes: Dict[str, Decimal] = {}
        self._last_order_times: Dict[str, datetime] = {}
        self._circuit_breaker_active = False
        self._risk_events: List[RiskEvent] = []

        # Configuración de límites
        self._max_position_percent = Decimal(str(self._settings.max_position_percent))
        self._max_daily_loss = Decimal(str(self._settings.max_daily_loss_percent))
        self._stop_loss_percent = Decimal(str(self._settings.stop_loss_percent))

        self._logger.info(
            "risk_manager_initialized",
            max_position_percent=float(self._max_position_percent),
            max_daily_loss_percent=float(self._max_daily_loss),
            stop_loss_percent=float(self._stop_loss_percent)
        )

    async def validate_order(
        self,
        order_request: OrderRequest,
        account_balance: List[Balance]
    ) -> bool:
        """
        Valida si una orden cumple con las reglas de riesgo.

        Args:
            order_request: Solicitud de orden a validar
            account_balance: Balance actual de la cuenta

        Returns:
            True si la orden pasa todas las validaciones
        """
        try:
            # Verificar circuit breaker
            if self._circuit_breaker_active:
                self._logger.log_risk_check(
                    symbol=order_request.symbol,
                    check_type="circuit_breaker",
                    passed=False,
                    reason="Circuit breaker active"
                )
                return False

            # Ejecutar todas las validaciones
            checks = [
                await self._check_balance_sufficient(order_request, account_balance),
                await self._check_position_size_limit(order_request, account_balance),
                await self._check_daily_loss_limit(order_request.symbol),
                await self._check_order_cooldown(order_request.symbol),
                await self._check_market_conditions(order_request),
                await self._check_correlation_risk(order_request)
            ]

            # Todas las verificaciones deben pasar
            all_passed = all(check.passed for check in checks)

            # Log del resultado final
            self._logger.log_risk_check(
                symbol=order_request.symbol,
                check_type="full_validation",
                passed=all_passed,
                reason="; ".join([check.reason for check in checks if not check.passed])
            )

            # Registrar event de riesgo si falló
            if not all_passed:
                await self._record_risk_event(
                    event_type="order_rejected",
                    symbol=order_request.symbol,
                    description=f"Order validation failed: {[c.reason for c in checks if not c.passed]}",
                    severity="MEDIUM"
                )

            return all_passed

        except Exception as e:
            self._logger.log_error_with_context(
                error=e,
                context="order_validation",
                symbol=order_request.symbol
            )
            return False

    async def calculate_position_size(
        self,
        symbol: str,
        signal_confidence: float,
        account_balance: Decimal
    ) -> Decimal:
        """
        Calcula el tamaño de posición óptimo basado en riesgo y confianza.

        Args:
            symbol: Símbolo del par
            signal_confidence: Confianza de la señal (0.0-1.0)
            account_balance: Balance disponible

        Returns:
            Tamaño de posición en USDT
        """
        try:
            # Validar inputs
            if signal_confidence < 0.5 or signal_confidence > 1.0:
                raise ValueError(f"Invalid signal confidence: {signal_confidence}")

            if account_balance <= 0:
                return Decimal('0')

            # Calcular tamaño base usando Kelly Criterion modificado
            base_position_percent = self._max_position_percent * Decimal(str(signal_confidence))

            # Ajustar por volatilidad del mercado
            volatility_adjustment = await self._get_volatility_adjustment(symbol)
            adjusted_percent = base_position_percent * volatility_adjustment

            # Aplicar límite máximo
            final_percent = min(adjusted_percent, self._max_position_percent)

            # Calcular cantidad en USDT
            position_size_usdt = account_balance * final_percent

            # Redondear hacia abajo para seguridad
            position_size = position_size_usdt.quantize(
                Decimal('0.00000001'),
                rounding=ROUND_DOWN
            )

            # Verificar monto mínimo
            min_amount = Decimal(str(self._settings.min_order_amount))
            if position_size < min_amount:
                self._logger.warning(
                    "position_size_below_minimum",
                    symbol=symbol,
                    calculated_size=float(position_size),
                    minimum_required=float(min_amount)
                )
                return Decimal('0')

            self._logger.info(
                "position_size_calculated",
                symbol=symbol,
                signal_confidence=signal_confidence,
                account_balance=float(account_balance),
                position_percent=float(final_percent),
                position_size_usdt=float(position_size)
            )

            return position_size

        except Exception as e:
            self._logger.log_error_with_context(
                error=e,
                context="position_size_calculation",
                symbol=symbol
            )
            return Decimal('0')

    async def should_stop_trading(self) -> bool:
        """
        Determina si se debe parar el trading por riesgo.

        Returns:
            True si se debe parar el trading
        """
        try:
            # Verificar circuit breaker
            if self._circuit_breaker_active:
                return True

            # Verificar pérdidas diarias
            total_daily_pnl = sum(self._daily_pnl.values())
            max_loss = Decimal(str(self._settings.max_daily_loss_percent))

            if total_daily_pnl < -max_loss:
                self._logger.warning(
                    "daily_loss_limit_exceeded",
                    total_pnl=float(total_daily_pnl),
                    max_loss_percent=float(max_loss)
                )
                await self._activate_circuit_breaker("Daily loss limit exceeded")
                return True

            # Verificar número de eventos de riesgo recientes
            recent_events = [
                event for event in self._risk_events
                if event.detected_at > datetime.now() - timedelta(hours=1)
            ]

            if len(recent_events) > 5:  # Más de 5 eventos en 1 hora
                self._logger.warning(
                    "too_many_risk_events",
                    events_count=len(recent_events),
                    time_window="1_hour"
                )
                await self._activate_circuit_breaker("Too many risk events")
                return True

            return False

        except Exception as e:
            self._logger.log_error_with_context(
                error=e,
                context="should_stop_trading_check"
            )
            return True  # Parar por seguridad si hay error

    # === MÉTODOS PRIVADOS DE VALIDACIÓN ===

    async def _check_balance_sufficient(
        self,
        order_request: OrderRequest,
        account_balance: List[Balance]
    ) -> RiskCheckResult:
        """Verifica que el balance sea suficiente para la orden."""
        try:
            # Obtener balance del asset base
            base_asset = self._settings.base_asset
            balance = next(
                (b for b in account_balance if b.asset == base_asset),
                None
            )

            if not balance:
                return RiskCheckResult(
                    passed=False,
                    reason=f"No balance found for {base_asset}"
                )

            # Calcular costo estimado de la orden
            estimated_cost = order_request.quantity
            if order_request.price:
                estimated_cost = order_request.quantity * order_request.price

            # Añadir margen para comisiones (0.1%)
            estimated_cost_with_fees = estimated_cost * Decimal('1.001')

            if balance.free < estimated_cost_with_fees:
                return RiskCheckResult(
                    passed=False,
                    reason=f"Insufficient balance: {balance.free} < {estimated_cost_with_fees}"
                )

            return RiskCheckResult(passed=True)

        except Exception as e:
            return RiskCheckResult(
                passed=False,
                reason=f"Balance check error: {str(e)}"
            )

    async def _check_position_size_limit(
        self,
        order_request: OrderRequest,
        account_balance: List[Balance]
    ) -> RiskCheckResult:
        """Verifica que el tamaño de posición no exceda límites."""
        try:
            base_balance = next(
                (b for b in account_balance if b.asset == self._settings.base_asset),
                None
            )

            if not base_balance:
                return RiskCheckResult(
                    passed=False,
                    reason="No base balance available"
                )

            # Calcular porcentaje de la posición
            total_balance = base_balance.total
            order_value = order_request.quantity
            if order_request.price:
                order_value = order_request.quantity * order_request.price

            position_percent = order_value / total_balance if total_balance > 0 else Decimal('1')

            if position_percent > self._max_position_percent:
                return RiskCheckResult(
                    passed=False,
                    reason=f"Position size {position_percent:.4f} exceeds limit {self._max_position_percent:.4f}"
                )

            return RiskCheckResult(passed=True)

        except Exception as e:
            return RiskCheckResult(
                passed=False,
                reason=f"Position size check error: {str(e)}"
            )

    async def _check_daily_loss_limit(self, symbol: str) -> RiskCheckResult:
        """Verifica que no se haya excedido el límite de pérdida diaria."""
        try:
            symbol_pnl = self._daily_pnl.get(symbol, Decimal('0'))

            if symbol_pnl < -self._max_daily_loss:
                return RiskCheckResult(
                    passed=False,
                    reason=f"Daily loss limit exceeded for {symbol}: {symbol_pnl}"
                )

            return RiskCheckResult(passed=True)

        except Exception as e:
            return RiskCheckResult(
                passed=False,
                reason=f"Daily loss check error: {str(e)}"
            )

    async def _check_order_cooldown(self, symbol: str) -> RiskCheckResult:
        """Verifica el cooldown entre órdenes."""
        try:
            last_order_time = self._last_order_times.get(symbol)

            if last_order_time:
                cooldown_period = timedelta(seconds=self._settings.order_cooldown_seconds)
                time_since_last = datetime.now() - last_order_time

                if time_since_last < cooldown_period:
                    return RiskCheckResult(
                        passed=False,
                        reason=f"Order cooldown active: {time_since_last} < {cooldown_period}"
                    )

            return RiskCheckResult(passed=True)

        except Exception as e:
            return RiskCheckResult(
                passed=False,
                reason=f"Cooldown check error: {str(e)}"
            )

    async def _check_market_conditions(self, order_request: OrderRequest) -> RiskCheckResult:
        """Verifica condiciones de mercado para trading seguro."""
        try:
            # Aquí se pueden añadir verificaciones de:
            # - Volatilidad extrema
            # - Spread bid-ask muy amplio
            # - Volumen insuficiente
            # Por ahora, siempre pasa

            return RiskCheckResult(passed=True)

        except Exception as e:
            return RiskCheckResult(
                passed=False,
                reason=f"Market conditions check error: {str(e)}"
            )

    async def _check_correlation_risk(self, order_request: OrderRequest) -> RiskCheckResult:
        """Verifica riesgo de correlación entre posiciones."""
        try:
            # Verificar número de posiciones abiertas
            open_positions = len(self._position_sizes)

            if open_positions >= self._settings.max_open_positions:
                return RiskCheckResult(
                    passed=False,
                    reason=f"Max open positions reached: {open_positions}"
                )

            return RiskCheckResult(passed=True)

        except Exception as e:
            return RiskCheckResult(
                passed=False,
                reason=f"Correlation check error: {str(e)}"
            )

    # === MÉTODOS AUXILIARES ===

    async def _get_volatility_adjustment(self, symbol: str) -> Decimal:
        """Calcula ajuste de volatilidad para sizing de posición."""
        try:
            # Implementación básica - puede mejorarse con datos históricos
            # Por ahora retorna factor conservador
            return Decimal('0.8')

        except Exception:
            return Decimal('0.5')  # Factor muy conservador en caso de error

    async def _activate_circuit_breaker(self, reason: str) -> None:
        """Activa el circuit breaker para parar trading."""
        self._circuit_breaker_active = True

        self._logger.warning(
            "circuit_breaker_activated",
            reason=reason,
            timestamp=datetime.now().isoformat()
        )

        await self._record_risk_event(
            event_type="circuit_breaker_activated",
            description=f"Circuit breaker activated: {reason}",
            severity="CRITICAL"
        )

    async def _record_risk_event(
        self,
        event_type: str,
        description: str,
        severity: str = "MEDIUM",
        symbol: Optional[str] = None
    ) -> None:
        """Registra un evento de riesgo."""
        risk_event = RiskEvent(
            event_type=event_type,
            severity=severity,
            symbol=symbol,
            description=description,
            detected_at=datetime.now()
        )

        self._risk_events.append(risk_event)

        # Mantener solo los últimos 100 eventos en memoria
        if len(self._risk_events) > 100:
            self._risk_events = self._risk_events[-100:]

    # === MÉTODOS PÚBLICOS ADICIONALES ===

    async def update_daily_pnl(self, symbol: str, pnl: Decimal) -> None:
        """Actualiza el PnL diario para un símbolo."""
        self._daily_pnl[symbol] = self._daily_pnl.get(symbol, Decimal('0')) + pnl

        self._logger.info(
            "daily_pnl_updated",
            symbol=symbol,
            pnl_change=float(pnl),
            total_daily_pnl=float(self._daily_pnl[symbol])
        )

    async def reset_daily_pnl(self) -> None:
        """Resetea el PnL diario (llamar al inicio del día)."""
        self._daily_pnl.clear()
        self._logger.info("daily_pnl_reset")

    async def deactivate_circuit_breaker(self) -> None:
        """Desactiva el circuit breaker (uso manual)."""
        self._circuit_breaker_active = False
        self._logger.info("circuit_breaker_deactivated")

    async def get_risk_summary(self) -> Dict[str, Any]:
        """Obtiene un resumen del estado de riesgo."""
        return {
            "circuit_breaker_active": self._circuit_breaker_active,
            "daily_pnl": {symbol: float(pnl) for symbol, pnl in self._daily_pnl.items()},
            "open_positions": len(self._position_sizes),
            "max_positions": self._settings.max_open_positions,
            "recent_risk_events": len([
                e for e in self._risk_events
                if e.detected_at > datetime.now() - timedelta(hours=1)
            ])
        }
