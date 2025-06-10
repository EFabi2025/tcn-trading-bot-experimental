"""
Sistema de logging estructurado para el trading bot.

Implementa logging profesional usando structlog con formateo JSON,
rotación de archivos y contexto estructurado para trading.
"""
import sys
import logging
from pathlib import Path
from typing import Any, Dict
import structlog
from structlog.stdlib import LoggerFactory
from structlog import get_logger
from structlog.processors import JSONRenderer, TimeStamper, add_log_level, CallsiteParameterAdder
from structlog.testing import LogCapture

from .config import get_settings


def configure_logging() -> None:
    """
    Configura el sistema de logging estructurado.
    
    Establece procesadores, formateadores y handlers según la configuración
    del sistema, incluyendo logging a archivos con rotación.
    """
    settings = get_settings()
    
    # Crear directorio de logs si no existe
    log_path = Path(settings.log_file_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configurar logging estándar de Python
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.log_level)
    )
    
    # Configurar procesadores de structlog
    processors = [
        # Añadir timestamp ISO
        TimeStamper(fmt="iso"),
        
        # Añadir nivel de log
        add_log_level,
        
        # Añadir información de ubicación del código (solo en debug)
        CallsiteParameterAdder(
            parameters=[
                CallsiteParameterAdder.Filename,
                CallsiteParameterAdder.Linenumber,
                CallsiteParameterAdder.FuncName,
            ]
        ) if settings.debug_mode else lambda _, __, event_dict: event_dict,
        
        # Procesador para stdlib logging
        structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
    ]
    
    # Configurar structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configurar handler para archivo si está habilitado
    if settings.log_to_file:
        setup_file_logging(settings)


def setup_file_logging(settings) -> None:
    """
    Configura logging a archivo con rotación.
    
    Args:
        settings: Configuración del sistema
    """
    from logging.handlers import TimedRotatingFileHandler
    
    # Handler para archivo con rotación
    file_handler = TimedRotatingFileHandler(
        filename=settings.log_file_path,
        when='midnight',
        interval=1,
        backupCount=settings.log_retention,
        encoding='utf-8'
    )
    
    # Formateador JSON para archivo
    file_formatter = structlog.stdlib.ProcessorFormatter(
        processor=JSONRenderer(),
    )
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(getattr(logging, settings.log_level))
    
    # Añadir handler al root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)


class TradingLogger:
    """
    Logger específico para trading con contexto estructurado.
    
    Proporciona métodos especializados para logging de eventos
    de trading con información estructurada relevante.
    """
    
    def __init__(self, name: str):
        """
        Inicializa el logger.
        
        Args:
            name: Nombre del logger (generalmente __name__)
        """
        self.logger = get_logger(name)
        self._context: Dict[str, Any] = {}
    
    def bind(self, **kwargs) -> 'TradingLogger':
        """
        Crea un nuevo logger con contexto adicional.
        
        Args:
            **kwargs: Contexto adicional para el logger
            
        Returns:
            Nuevo logger con contexto vinculado
        """
        new_logger = TradingLogger(self.logger._context.get('logger', ''))
        new_logger.logger = self.logger.bind(**kwargs)
        new_logger._context = {**self._context, **kwargs}
        return new_logger
    
    def info(self, msg: str, **kwargs) -> None:
        """Log de información."""
        self.logger.info(msg, **kwargs)
    
    def debug(self, msg: str, **kwargs) -> None:
        """Log de debug."""
        self.logger.debug(msg, **kwargs)
    
    def warning(self, msg: str, **kwargs) -> None:
        """Log de advertencia."""
        self.logger.warning(msg, **kwargs)
    
    def error(self, msg: str, **kwargs) -> None:
        """Log de error."""
        self.logger.error(msg, **kwargs)
    
    def critical(self, msg: str, **kwargs) -> None:
        """Log crítico."""
        self.logger.critical(msg, **kwargs)
    
    # === MÉTODOS ESPECIALIZADOS PARA TRADING ===
    
    def log_signal(self, symbol: str, action: str, confidence: float, 
                   predicted_price: float, **kwargs) -> None:
        """
        Log de señal de trading generada.
        
        Args:
            symbol: Símbolo del par
            action: Acción (BUY/SELL)
            confidence: Nivel de confianza
            predicted_price: Precio predicho
            **kwargs: Contexto adicional
        """
        self.info(
            "signal_generated",
            event_type="signal",
            symbol=symbol,
            action=action,
            confidence=confidence,
            predicted_price=predicted_price,
            **kwargs
        )
    
    def log_order_created(self, order_id: str, symbol: str, side: str,
                         quantity: float, price: float, **kwargs) -> None:
        """
        Log de orden creada.
        
        Args:
            order_id: ID de la orden
            symbol: Símbolo del par
            side: Lado de la orden (BUY/SELL)
            quantity: Cantidad
            price: Precio
            **kwargs: Contexto adicional
        """
        self.info(
            "order_created",
            event_type="order",
            order_id=order_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            **kwargs
        )
    
    def log_order_filled(self, order_id: str, symbol: str, side: str,
                        filled_quantity: float, avg_price: float,
                        commission: float, **kwargs) -> None:
        """
        Log de orden ejecutada.
        
        Args:
            order_id: ID de la orden
            symbol: Símbolo del par
            side: Lado de la orden
            filled_quantity: Cantidad ejecutada
            avg_price: Precio promedio
            commission: Comisión
            **kwargs: Contexto adicional
        """
        self.info(
            "order_filled",
            event_type="execution",
            order_id=order_id,
            symbol=symbol,
            side=side,
            filled_quantity=filled_quantity,
            avg_price=avg_price,
            commission=commission,
            **kwargs
        )
    
    def log_balance_update(self, asset: str, free: float, locked: float,
                          previous_free: float = None, **kwargs) -> None:
        """
        Log de actualización de balance.
        
        Args:
            asset: Asset actualizado
            free: Balance libre
            locked: Balance bloqueado
            previous_free: Balance libre anterior
            **kwargs: Contexto adicional
        """
        log_data = {
            "event_type": "balance",
            "asset": asset,
            "free": free,
            "locked": locked,
            "total": free + locked,
        }
        
        if previous_free is not None:
            log_data["change"] = free - previous_free
        
        self.info("balance_updated", **log_data, **kwargs)
    
    def log_risk_check(self, symbol: str, check_type: str, passed: bool,
                      reason: str = None, **kwargs) -> None:
        """
        Log de verificación de riesgo.
        
        Args:
            symbol: Símbolo verificado
            check_type: Tipo de verificación
            passed: Si pasó la verificación
            reason: Razón si no pasó
            **kwargs: Contexto adicional
        """
        level = "info" if passed else "warning"
        
        log_data = {
            "event_type": "risk_check",
            "symbol": symbol,
            "check_type": check_type,
            "passed": passed,
        }
        
        if reason:
            log_data["reason"] = reason
        
        getattr(self, level)("risk_check_performed", **log_data, **kwargs)
    
    def log_model_prediction(self, symbol: str, confidence: float,
                           prediction: str, features_count: int,
                           processing_time_ms: float, **kwargs) -> None:
        """
        Log de predicción del modelo ML.
        
        Args:
            symbol: Símbolo analizado
            confidence: Confianza de la predicción
            prediction: Predicción generada
            features_count: Número de features utilizadas
            processing_time_ms: Tiempo de procesamiento
            **kwargs: Contexto adicional
        """
        self.info(
            "model_prediction",
            event_type="ml_prediction",
            symbol=symbol,
            confidence=confidence,
            prediction=prediction,
            features_count=features_count,
            processing_time_ms=processing_time_ms,
            **kwargs
        )
    
    def log_error_with_context(self, error: Exception, context: str,
                              symbol: str = None, **kwargs) -> None:
        """
        Log de error con contexto completo.
        
        Args:
            error: Excepción ocurrida
            context: Contexto donde ocurrió el error
            symbol: Símbolo relacionado (opcional)
            **kwargs: Contexto adicional
        """
        log_data = {
            "event_type": "error",
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context,
        }
        
        if symbol:
            log_data["symbol"] = symbol
        
        self.error("error_occurred", **log_data, **kwargs)
    
    def log_performance_metric(self, metric_name: str, value: float,
                              symbol: str = None, timeframe: str = None,
                              **kwargs) -> None:
        """
        Log de métricas de performance.
        
        Args:
            metric_name: Nombre de la métrica
            value: Valor de la métrica
            symbol: Símbolo relacionado (opcional)
            timeframe: Timeframe de la métrica (opcional)
            **kwargs: Contexto adicional
        """
        log_data = {
            "event_type": "performance",
            "metric_name": metric_name,
            "value": value,
        }
        
        if symbol:
            log_data["symbol"] = symbol
        if timeframe:
            log_data["timeframe"] = timeframe
        
        self.info("performance_metric", **log_data, **kwargs)


def get_trading_logger(name: str) -> TradingLogger:
    """
    Factory function para obtener un logger de trading.
    
    Args:
        name: Nombre del logger (generalmente __name__)
        
    Returns:
        Instancia de TradingLogger configurada
    """
    return TradingLogger(name)


# Configurar logging al importar el módulo
configure_logging() 