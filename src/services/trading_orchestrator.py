"""
🧪 EDUCATIONAL Trading Orchestrator - Trading Bot Experimental

Este módulo implementa el orquestador educacional que:
- Coordina todos los servicios del trading bot
- Ejecuta el ciclo de trading en modo educacional
- Integra ML, risk management y execution
- Demuestra patrones de arquitectura limpia

⚠️ EXPERIMENTAL: Solo para fines educacionales
"""

import asyncio
from typing import Optional, Dict, Any, List
from decimal import Decimal
from datetime import datetime, timezone
from contextlib import asynccontextmanager

import structlog

from ..interfaces.trading_interfaces import (
    ITradingClient, IMLPredictor, IRiskManager,
    IMarketDataProvider, INotificationService, ITradingStrategy
)
from ..schemas.trading_schemas import (
    MarketDataSchema, TradingSignalSchema, OrderRequestSchema,
    OrderSchema, BalanceSchema
)
from ..core.config import TradingBotSettings
from ..core.logging_config import TradingLogger

logger = structlog.get_logger(__name__)


class EducationalTradingOrchestrator:
    """
    🎓 Orquestador educacional del trading bot

    Características educacionales:
    - Coordina todos los servicios SOLID
    - Ejecuta ciclo de trading educacional
    - Implementa dry-run mode por defecto
    - Demuestra flujo completo de trading algorítmico
    """

    def __init__(
        self,
        settings: TradingBotSettings,
        trading_logger: TradingLogger,
        trading_client: ITradingClient,
        market_data_provider: IMarketDataProvider,
        ml_predictor: IMLPredictor,
        risk_manager: IRiskManager,
        notification_service: Optional[INotificationService] = None,
        trading_strategy: Optional[ITradingStrategy] = None
    ):
        """
        Inicializa el orquestador educacional

        Args:
            settings: Configuración del bot
            trading_logger: Logger estructurado
            trading_client: Cliente de trading (Binance educacional)
            market_data_provider: Proveedor de datos de mercado
            ml_predictor: Predictor ML con modelo TCN
            risk_manager: Gestor de riesgos educacional
            notification_service: Servicio de notificaciones (opcional)
            trading_strategy: Estrategia de trading (opcional)
        """
        self.settings = settings
        self.logger = trading_logger

        # Servicios principales
        self.trading_client = trading_client
        self.market_data_provider = market_data_provider
        self.ml_predictor = ml_predictor
        self.risk_manager = risk_manager
        self.notification_service = notification_service
        self.trading_strategy = trading_strategy

        # Estado del orquestador
        self.is_running = False
        self.is_paused = False
        self.trading_symbols = settings.trading_symbols
        self.trading_interval = settings.trading_interval_seconds

        # Métricas educacionales
        self.total_signals_generated = 0
        self.total_orders_executed = 0
        self.total_orders_rejected = 0
        self.total_profit_loss = Decimal('0.0')

        # Buffer de datos históricos
        self.market_data_history: Dict[str, List[MarketDataSchema]] = {
            symbol: [] for symbol in self.trading_symbols
        }

        self.logger.log_system_event(
            "educational_orchestrator_initialized",
            symbols=self.trading_symbols,
            interval_seconds=self.trading_interval,
            dry_run=settings.dry_run,
            educational_note="Orquestador educacional listo para experimentación"
        )

    async def start_trading(self) -> None:
        """🎓 Inicia el sistema de trading educacional"""
        if self.is_running:
            self.logger.log_system_event(
                "educational_trading_already_running",
                educational_note="Sistema ya está ejecutándose"
            )
            return

        self.is_running = True
        self.is_paused = False

        self.logger.log_system_event(
            "educational_trading_started",
            symbols=self.trading_symbols,
            educational_note="Sistema de trading educacional iniciado"
        )

        try:
            # Verificar conexiones educacionales
            await self._verify_all_services()

            # Obtener balances iniciales
            await self._log_initial_balances()

            # Iniciar ciclo principal de trading
            await self._run_trading_loop()

        except Exception as e:
            self.logger.log_error(
                "educational_trading_startup_failed",
                error=str(e),
                educational_tip="Verificar configuración y conexiones"
            )
            self.is_running = False
            raise

    async def stop_trading(self) -> None:
        """🎓 Detiene el sistema de trading educacional"""
        if not self.is_running:
            return

        self.is_running = False

        self.logger.log_system_event(
            "educational_trading_stopped",
            total_signals=self.total_signals_generated,
            total_orders=self.total_orders_executed,
            total_rejected=self.total_orders_rejected,
            educational_note="Sistema de trading educacional detenido"
        )

        # Cerrar servicios
        await self._close_all_services()

    async def pause_trading(self) -> None:
        """🎓 Pausa el trading educacional"""
        self.is_paused = True
        self.logger.log_system_event(
            "educational_trading_paused",
            educational_note="Trading pausado para análisis"
        )

    async def resume_trading(self) -> None:
        """🎓 Reanuda el trading educacional"""
        self.is_paused = False
        self.logger.log_system_event(
            "educational_trading_resumed",
            educational_note="Trading reanudado después de pausa"
        )

    async def _verify_all_services(self) -> None:
        """Verifica que todos los servicios estén funcionando"""
        # Verificar cliente de trading
        if not self.trading_client.is_connected():
            raise ConnectionError("🎓 Cliente de trading no conectado")

        # Verificar modelo ML
        ml_performance = self.ml_predictor.get_model_performance()
        if not ml_performance.get("model_loaded", False):
            raise ValueError("🎓 Modelo ML no cargado correctamente")

        self.logger.log_system_event(
            "educational_services_verified",
            ml_model_params=ml_performance.get("model_parameters", 0),
            educational_note="Todos los servicios verificados"
        )

    async def _log_initial_balances(self) -> None:
        """Registra balances iniciales para educación"""
        try:
            balances = await self.trading_client.get_balances()
            self.logger.log_balance_check(
                [b.dict() for b in balances],
                educational_note="Balances iniciales de testnet"
            )
        except Exception as e:
            self.logger.log_error(
                "educational_initial_balance_failed",
                error=str(e),
                educational_tip="Error obteniendo balances de testnet"
            )

    async def _run_trading_loop(self) -> None:
        """🎓 Ciclo principal de trading educacional"""
        while self.is_running:
            try:
                # Verificar si está pausado
                if self.is_paused:
                    await asyncio.sleep(1)
                    continue

                # Ejecutar ciclo para cada símbolo
                for symbol in self.trading_symbols:
                    if not self.is_running:
                        break

                    await self._process_trading_cycle(symbol)

                # Esperar antes del próximo ciclo
                await asyncio.sleep(self.trading_interval)

            except Exception as e:
                self.logger.log_error(
                    "educational_trading_loop_error",
                    error=str(e),
                    educational_tip="Error en ciclo de trading"
                )
                # Continuar con el próximo ciclo
                await asyncio.sleep(5)

    async def _process_trading_cycle(self, symbol: str) -> None:
        """
        🎓 Procesa un ciclo completo de trading para un símbolo

        Flujo educacional:
        1. Obtener datos de mercado
        2. Generar predicción ML
        3. Evaluar riesgo
        4. Ejecutar orden (si es aprobada)
        5. Monitorear resultado
        """
        cycle_start = datetime.now(timezone.utc)

        try:
            # 1. Obtener datos de mercado
            market_data = await self._get_market_data(symbol)
            if not market_data:
                return

            # Actualizar historial
            self._update_market_history(symbol, market_data)

            # 2. Generar señal ML
            signal = await self._generate_ml_signal(symbol)
            if not signal or signal.action == "HOLD":
                return

            self.total_signals_generated += 1

            # 3. Evaluar riesgo
            risk_approved = await self._evaluate_risk(signal, market_data)
            if not risk_approved:
                self.total_orders_rejected += 1
                return

            # 4. Ejecutar orden educacional
            order_executed = await self._execute_order(signal, market_data)
            if order_executed:
                self.total_orders_executed += 1

            # 5. Log del ciclo completo
            cycle_duration = (datetime.now(timezone.utc) - cycle_start).total_seconds()
            self.logger.log_trading_cycle(
                symbol=symbol,
                signal_action=signal.action,
                signal_confidence=float(signal.confidence),
                order_executed=order_executed,
                cycle_duration_seconds=cycle_duration,
                educational_note="Ciclo de trading educacional completado"
            )

        except Exception as e:
            self.logger.log_error(
                "educational_trading_cycle_failed",
                symbol=symbol,
                error=str(e),
                educational_tip="Error en ciclo de trading del símbolo"
            )

    async def _get_market_data(self, symbol: str) -> Optional[MarketDataSchema]:
        """Obtiene datos de mercado actuales"""
        try:
            market_data = await self.market_data_provider.get_market_data(symbol)
            return market_data
        except Exception as e:
            self.logger.log_error(
                "educational_market_data_failed",
                symbol=symbol,
                error=str(e),
                educational_tip="Error obteniendo datos de mercado"
            )
            return None

    def _update_market_history(self, symbol: str, market_data: MarketDataSchema) -> None:
        """Actualiza historial de datos de mercado"""
        if symbol not in self.market_data_history:
            self.market_data_history[symbol] = []

        self.market_data_history[symbol].append(market_data)

        # Mantener solo últimos N elementos
        max_history = 100  # Buffer educacional
        if len(self.market_data_history[symbol]) > max_history:
            self.market_data_history[symbol] = self.market_data_history[symbol][-max_history:]

    async def _generate_ml_signal(self, symbol: str) -> Optional[TradingSignalSchema]:
        """Genera señal usando ML predictor"""
        try:
            # Obtener historial de datos
            historical_data = self.market_data_history.get(symbol, [])
            if len(historical_data) < 60:  # Mínimo para TCN
                return None

            # Generar predicción
            signal = await self.ml_predictor.predict(historical_data)

            self.logger.log_ml_prediction(
                signal.dict(),
                educational_note="Señal ML generada para símbolo"
            )

            return signal

        except Exception as e:
            self.logger.log_error(
                "educational_ml_signal_failed",
                symbol=symbol,
                error=str(e),
                educational_tip="Error generando señal ML"
            )
            return None

    async def _evaluate_risk(
        self,
        signal: TradingSignalSchema,
        market_data: MarketDataSchema
    ) -> bool:
        """Evalúa riesgo de la señal de trading"""
        try:
            # Crear orden request para validación
            order_request = OrderRequestSchema(
                symbol=signal.symbol,
                side=signal.action,
                type="MARKET",
                quantity=Decimal('0.001'),  # Cantidad educacional mínima
                price=market_data.price,
                metadata={
                    "signal_confidence": float(signal.confidence),
                    "educational": True
                }
            )

            # Validar con risk manager
            is_approved = await self.risk_manager.validate_order(order_request)

            self.logger.log_risk_assessment(
                signal.dict(),
                risk_approved=is_approved,
                educational_note="Evaluación de riesgo para señal"
            )

            return is_approved

        except Exception as e:
            self.logger.log_error(
                "educational_risk_evaluation_failed",
                error=str(e),
                educational_tip="Error en evaluación de riesgo"
            )
            return False

    async def _execute_order(
        self,
        signal: TradingSignalSchema,
        market_data: MarketDataSchema
    ) -> bool:
        """Ejecuta orden de trading (modo educacional)"""
        try:
            # Crear orden request
            order_request = OrderRequestSchema(
                symbol=signal.symbol,
                side=signal.action,
                type="MARKET",
                quantity=Decimal('0.001'),  # Cantidad educacional
                price=market_data.price,
                metadata={
                    "signal_confidence": float(signal.confidence),
                    "signal_strength": signal.strength,
                    "educational": True,
                    "ml_reasoning": signal.reasoning
                }
            )

            # Ejecutar orden (simulada en dry-run)
            order = await self.trading_client.create_order(order_request)

            self.logger.log_order_completed(
                order.dict(),
                dry_run=self.settings.dry_run,
                educational_note="Orden educacional ejecutada"
            )

            # Notificar si hay servicio disponible
            if self.notification_service:
                await self._send_notification(
                    f"🎓 Orden educacional: {signal.action} {signal.symbol} @ {market_data.price}"
                )

            return True

        except Exception as e:
            self.logger.log_error(
                "educational_order_execution_failed",
                error=str(e),
                signal_action=signal.action,
                symbol=signal.symbol,
                educational_tip="Error ejecutando orden educacional"
            )
            return False

    async def _send_notification(self, message: str) -> None:
        """Envía notificación educacional"""
        try:
            if self.notification_service:
                await self.notification_service.send_notification(
                    message=message,
                    level="INFO",
                    metadata={"educational": True}
                )
        except Exception as e:
            self.logger.log_error(
                "educational_notification_failed",
                error=str(e),
                educational_tip="Error enviando notificación"
            )

    async def _close_all_services(self) -> None:
        """Cierra todos los servicios del orquestador"""
        try:
            # Cerrar servicios principales
            await self.trading_client.close()
            await self.ml_predictor.close()

            if self.notification_service:
                await self.notification_service.close()

            self.logger.log_system_event(
                "educational_all_services_closed",
                educational_note="Todos los servicios cerrados correctamente"
            )

        except Exception as e:
            self.logger.log_error(
                "educational_services_close_failed",
                error=str(e),
                educational_tip="Error cerrando servicios"
            )

    def get_trading_stats(self) -> Dict[str, Any]:
        """🎓 Obtiene estadísticas educacionales del trading"""
        return {
            "is_running": self.is_running,
            "is_paused": self.is_paused,
            "total_signals_generated": self.total_signals_generated,
            "total_orders_executed": self.total_orders_executed,
            "total_orders_rejected": self.total_orders_rejected,
            "success_rate": (
                self.total_orders_executed / max(1, self.total_signals_generated)
            ) * 100,
            "trading_symbols": self.trading_symbols,
            "trading_interval_seconds": self.trading_interval,
            "market_data_history_size": {
                symbol: len(history)
                for symbol, history in self.market_data_history.items()
            },
            "educational_note": "Estadísticas del trading bot experimental"
        }

    @asynccontextmanager
    async def trading_session(self):
        """🎓 Context manager para sesión de trading educacional"""
        try:
            await self.start_trading()
            yield self
        except Exception as e:
            self.logger.log_error(
                "educational_trading_session_failed",
                error=str(e),
                educational_tip="Error en sesión de trading"
            )
            raise
        finally:
            await self.stop_trading()
