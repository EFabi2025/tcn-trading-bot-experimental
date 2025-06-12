"""
Modelos SQLAlchemy para persistencia de datos de trading.

Define las tablas de base de datos para órdenes, señales, balances
y métricas de performance con integridad referencial.
"""
from sqlalchemy import Column, Integer, String, DateTime, Decimal, Boolean, Text, ForeignKey, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
from decimal import Decimal as PyDecimal
from typing import Optional

# Base para todos los modelos
Base = declarative_base()


class Order(Base):
    """
    Modelo para órdenes de trading ejecutadas.
    
    Almacena información completa de cada orden incluyendo
    estado, precios y comisiones.
    """
    __tablename__ = 'orders'
    
    # Campos principales
    id = Column(String(50), primary_key=True, comment="ID único de la orden")
    symbol = Column(String(20), nullable=False, index=True, comment="Par de trading")
    side = Column(String(10), nullable=False, comment="BUY o SELL")
    order_type = Column(String(20), nullable=False, default="MARKET", comment="Tipo de orden")
    status = Column(String(20), nullable=False, comment="Estado de la orden")
    
    # Cantidades y precios - SIEMPRE Decimal para valores monetarios
    quantity = Column(Decimal(18, 8), nullable=False, comment="Cantidad solicitada")
    price = Column(Decimal(18, 8), nullable=True, comment="Precio de la orden")
    filled_quantity = Column(Decimal(18, 8), nullable=False, default=0, comment="Cantidad ejecutada")
    avg_fill_price = Column(Decimal(18, 8), nullable=True, comment="Precio promedio de ejecución")
    commission = Column(Decimal(18, 8), nullable=False, default=0, comment="Comisión pagada")
    commission_asset = Column(String(10), nullable=True, comment="Asset de la comisión")
    
    # Timestamps
    created_at = Column(DateTime, nullable=False, default=func.now(), comment="Timestamp de creación")
    updated_at = Column(DateTime, nullable=False, default=func.now(), onupdate=func.now(), comment="Última actualización")
    executed_at = Column(DateTime, nullable=True, comment="Timestamp de ejecución")
    
    # Metadatos
    binance_order_id = Column(String(50), nullable=True, unique=True, comment="ID de orden en Binance")
    client_order_id = Column(String(50), nullable=True, comment="ID de orden del cliente")
    time_in_force = Column(String(10), nullable=True, default="GTC", comment="Time in force")
    
    # Relaciones
    signal_id = Column(String(50), ForeignKey('trading_signals.id'), nullable=True, comment="Señal que originó la orden")
    signal = relationship("TradingSignal", back_populates="orders")
    
    # Índices para optimización
    __table_args__ = (
        Index('idx_orders_symbol_created', 'symbol', 'created_at'),
        Index('idx_orders_status_created', 'status', 'created_at'),
        Index('idx_orders_side_symbol', 'side', 'symbol'),
    )
    
    def __repr__(self) -> str:
        return f"<Order {self.id}: {self.side} {self.quantity} {self.symbol} @ {self.price}>"
    
    @property
    def is_filled(self) -> bool:
        """Verifica si la orden está completamente ejecutada."""
        return self.status == "FILLED"
    
    @property
    def fill_percentage(self) -> float:
        """Calcula el porcentaje de ejecución."""
        if self.quantity == 0:
            return 0.0
        return float(self.filled_quantity / self.quantity * 100)


class TradingSignal(Base):
    """
    Modelo para señales de trading generadas por ML.
    
    Almacena predicciones del modelo con metadata
    y seguimiento de performance.
    """
    __tablename__ = 'trading_signals'
    
    # Campos principales
    id = Column(String(50), primary_key=True, comment="ID único de la señal")
    symbol = Column(String(20), nullable=False, index=True, comment="Par de trading")
    action = Column(String(10), nullable=False, comment="BUY o SELL")
    confidence = Column(Decimal(5, 4), nullable=False, comment="Confianza del modelo (0-1)")
    predicted_price = Column(Decimal(18, 8), nullable=False, comment="Precio predicho")
    
    # Timestamps
    created_at = Column(DateTime, nullable=False, default=func.now(), comment="Timestamp de creación")
    expires_at = Column(DateTime, nullable=True, comment="Timestamp de expiración")
    
    # Metadatos del modelo
    model_version = Column(String(50), nullable=True, comment="Versión del modelo ML")
    features_used = Column(Text, nullable=True, comment="Features utilizadas (JSON)")
    market_data_timestamp = Column(DateTime, nullable=True, comment="Timestamp de los datos usados")
    processing_time_ms = Column(Integer, nullable=True, comment="Tiempo de procesamiento en ms")
    
    # Estado de la señal
    is_active = Column(Boolean, nullable=False, default=True, comment="Si la señal está activa")
    was_executed = Column(Boolean, nullable=False, default=False, comment="Si se ejecutó orden")
    execution_reason = Column(String(100), nullable=True, comment="Razón de ejecución/rechazo")
    
    # Análisis post-ejecución
    actual_price_1m = Column(Decimal(18, 8), nullable=True, comment="Precio real 1min después")
    actual_price_5m = Column(Decimal(18, 8), nullable=True, comment="Precio real 5min después")
    actual_price_15m = Column(Decimal(18, 8), nullable=True, comment="Precio real 15min después")
    
    # Relaciones
    orders = relationship("Order", back_populates="signal")
    
    # Índices
    __table_args__ = (
        Index('idx_signals_symbol_created', 'symbol', 'created_at'),
        Index('idx_signals_action_confidence', 'action', 'confidence'),
        Index('idx_signals_active_created', 'is_active', 'created_at'),
    )
    
    def __repr__(self) -> str:
        return f"<Signal {self.id}: {self.action} {self.symbol} conf={self.confidence}>"
    
    @property
    def prediction_accuracy_1m(self) -> Optional[float]:
        """Calcula la precisión de predicción a 1 minuto."""
        if not self.actual_price_1m:
            return None
        
        predicted = float(self.predicted_price)
        actual = float(self.actual_price_1m)
        return 1.0 - abs(predicted - actual) / actual


class Balance(Base):
    """
    Modelo para snapshots de balance de cuenta.
    
    Rastrea cambios en balances para análisis
    y detección de anomalías.
    """
    __tablename__ = 'balances'
    
    # Campos principales
    id = Column(Integer, primary_key=True, autoincrement=True)
    asset = Column(String(10), nullable=False, index=True, comment="Asset")
    free = Column(Decimal(18, 8), nullable=False, comment="Balance libre")
    locked = Column(Decimal(18, 8), nullable=False, default=0, comment="Balance bloqueado")
    
    # Timestamp
    snapshot_at = Column(DateTime, nullable=False, default=func.now(), index=True, comment="Momento del snapshot")
    
    # Contexto del snapshot
    trigger_event = Column(String(50), nullable=True, comment="Evento que disparó el snapshot")
    order_id = Column(String(50), ForeignKey('orders.id'), nullable=True, comment="Orden relacionada")
    
    # Índices
    __table_args__ = (
        Index('idx_balances_asset_snapshot', 'asset', 'snapshot_at'),
        Index('idx_balances_trigger_snapshot', 'trigger_event', 'snapshot_at'),
    )
    
    def __repr__(self) -> str:
        return f"<Balance {self.asset}: {self.free} free, {self.locked} locked>"
    
    @property
    def total(self) -> PyDecimal:
        """Balance total disponible."""
        return self.free + self.locked


class PerformanceMetric(Base):
    """
    Modelo para métricas de performance del bot.
    
    Almacena KPIs y estadísticas para monitoreo
    y optimización del sistema.
    """
    __tablename__ = 'performance_metrics'
    
    # Campos principales
    id = Column(Integer, primary_key=True, autoincrement=True)
    metric_name = Column(String(100), nullable=False, index=True, comment="Nombre de la métrica")
    metric_value = Column(Decimal(18, 8), nullable=False, comment="Valor de la métrica")
    metric_type = Column(String(50), nullable=False, comment="Tipo de métrica (profit, accuracy, etc)")
    
    # Contexto
    symbol = Column(String(20), nullable=True, index=True, comment="Símbolo relacionado")
    timeframe = Column(String(20), nullable=True, comment="Marco temporal")
    calculation_period_start = Column(DateTime, nullable=True, comment="Inicio del período de cálculo")
    calculation_period_end = Column(DateTime, nullable=True, comment="Fin del período de cálculo")
    
    # Timestamp
    recorded_at = Column(DateTime, nullable=False, default=func.now(), index=True, comment="Timestamp de registro")
    
    # Metadatos
    calculation_method = Column(String(100), nullable=True, comment="Método de cálculo")
    additional_data = Column(Text, nullable=True, comment="Datos adicionales (JSON)")
    
    # Índices
    __table_args__ = (
        Index('idx_metrics_name_recorded', 'metric_name', 'recorded_at'),
        Index('idx_metrics_type_symbol', 'metric_type', 'symbol'),
        Index('idx_metrics_symbol_timeframe', 'symbol', 'timeframe', 'recorded_at'),
    )
    
    def __repr__(self) -> str:
        return f"<Metric {self.metric_name}: {self.metric_value} ({self.metric_type})>"


class RiskEvent(Base):
    """
    Modelo para eventos de riesgo y alertas.
    
    Registra violaciones de límites de riesgo
    y acciones tomadas por el sistema.
    """
    __tablename__ = 'risk_events'
    
    # Campos principales
    id = Column(Integer, primary_key=True, autoincrement=True)
    event_type = Column(String(50), nullable=False, index=True, comment="Tipo de evento de riesgo")
    severity = Column(String(20), nullable=False, comment="Severidad: LOW, MEDIUM, HIGH, CRITICAL")
    symbol = Column(String(20), nullable=True, index=True, comment="Símbolo relacionado")
    
    # Detalles del evento
    description = Column(Text, nullable=False, comment="Descripción del evento")
    trigger_value = Column(Decimal(18, 8), nullable=True, comment="Valor que disparó el evento")
    threshold_value = Column(Decimal(18, 8), nullable=True, comment="Valor umbral configurado")
    
    # Acciones tomadas
    action_taken = Column(String(100), nullable=True, comment="Acción tomada por el sistema")
    orders_affected = Column(Text, nullable=True, comment="IDs de órdenes afectadas (JSON)")
    
    # Estado
    is_resolved = Column(Boolean, nullable=False, default=False, comment="Si el evento está resuelto")
    resolved_at = Column(DateTime, nullable=True, comment="Timestamp de resolución")
    resolution_notes = Column(Text, nullable=True, comment="Notas de resolución")
    
    # Timestamps
    detected_at = Column(DateTime, nullable=False, default=func.now(), comment="Timestamp de detección")
    
    # Índices
    __table_args__ = (
        Index('idx_risk_events_type_detected', 'event_type', 'detected_at'),
        Index('idx_risk_events_severity_detected', 'severity', 'detected_at'),
        Index('idx_risk_events_symbol_detected', 'symbol', 'detected_at'),
        Index('idx_risk_events_resolved', 'is_resolved', 'detected_at'),
    )
    
    def __repr__(self) -> str:
        return f"<RiskEvent {self.event_type}: {self.severity} - {self.description[:50]}>"


class ModelPerformance(Base):
    """
    Modelo para tracking de performance del modelo ML.
    
    Analiza la precisión y efectividad de las predicciones
    del modelo a lo largo del tiempo.
    """
    __tablename__ = 'model_performance'
    
    # Campos principales
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_version = Column(String(50), nullable=False, index=True, comment="Versión del modelo")
    symbol = Column(String(20), nullable=False, index=True, comment="Símbolo analizado")
    
    # Métricas de performance
    accuracy_1m = Column(Decimal(5, 4), nullable=True, comment="Precisión a 1 minuto")
    accuracy_5m = Column(Decimal(5, 4), nullable=True, comment="Precisión a 5 minutos")
    accuracy_15m = Column(Decimal(5, 4), nullable=True, comment="Precisión a 15 minutos")
    
    # Estadísticas de confianza
    avg_confidence = Column(Decimal(5, 4), nullable=False, comment="Confianza promedio")
    min_confidence = Column(Decimal(5, 4), nullable=False, comment="Confianza mínima")
    max_confidence = Column(Decimal(5, 4), nullable=False, comment="Confianza máxima")
    
    # Contadores
    total_predictions = Column(Integer, nullable=False, default=0, comment="Total de predicciones")
    successful_predictions = Column(Integer, nullable=False, default=0, comment="Predicciones exitosas")
    
    # Período de análisis
    analysis_period_start = Column(DateTime, nullable=False, comment="Inicio del período")
    analysis_period_end = Column(DateTime, nullable=False, comment="Fin del período")
    calculated_at = Column(DateTime, nullable=False, default=func.now(), comment="Timestamp de cálculo")
    
    # Índices
    __table_args__ = (
        Index('idx_model_perf_version_symbol', 'model_version', 'symbol'),
        Index('idx_model_perf_calculated', 'calculated_at'),
        Index('idx_model_perf_accuracy', 'accuracy_5m', 'calculated_at'),
    )
    
    def __repr__(self) -> str:
        return f"<ModelPerf {self.model_version} {self.symbol}: {self.accuracy_5m} acc>"
    
    @property
    def success_rate(self) -> float:
        """Calcula la tasa de éxito general."""
        if self.total_predictions == 0:
            return 0.0
        return self.successful_predictions / self.total_predictions 