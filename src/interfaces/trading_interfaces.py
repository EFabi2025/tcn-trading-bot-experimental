"""
Interfaces principales para el sistema de trading.

Define los contratos que deben seguir todos los servicios de trading,
garantizando arquitectura limpia y testeable.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from decimal import Decimal
from datetime import datetime
from dataclasses import dataclass
from enum import Enum


class OrderSide(Enum):
    """Tipo de orden: compra o venta."""
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(Enum):
    """Estado de una orden."""
    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"


@dataclass
class MarketData:
    """Datos de mercado para un símbolo."""
    symbol: str
    price: Decimal
    volume: Decimal
    timestamp: datetime
    bid: Optional[Decimal] = None
    ask: Optional[Decimal] = None
    high_24h: Optional[Decimal] = None
    low_24h: Optional[Decimal] = None


@dataclass
class OrderRequest:
    """Solicitud de orden de trading."""
    symbol: str
    side: OrderSide
    quantity: Decimal
    price: Optional[Decimal] = None  # None para market orders
    order_type: str = "MARKET"


@dataclass
class Order:
    """Orden de trading ejecutada."""
    id: str
    symbol: str
    side: OrderSide
    quantity: Decimal
    price: Decimal
    status: OrderStatus
    timestamp: datetime
    filled_quantity: Decimal = Decimal('0')
    commission: Decimal = Decimal('0')


@dataclass
class Balance:
    """Balance de una criptomoneda."""
    asset: str
    free: Decimal
    locked: Decimal

    @property
    def total(self) -> Decimal:
        """Balance total disponible."""
        return self.free + self.locked


@dataclass
class TradingSignal:
    """Señal de trading generada por el modelo ML."""
    symbol: str
    action: OrderSide
    confidence: float  # 0.0 - 1.0
    predicted_price: Decimal
    timestamp: datetime
    metadata: Dict[str, Any]


class IMarketDataProvider(ABC):
    """Interfaz para proveedores de datos de mercado."""

    @abstractmethod
    async def get_current_price(self, symbol: str) -> Decimal:
        """Obtiene el precio actual de un símbolo."""
        pass

    @abstractmethod
    async def get_market_data(self, symbol: str) -> MarketData:
        """Obtiene datos completos de mercado para un símbolo."""
        pass

    @abstractmethod
    async def get_historical_data(
        self,
        symbol: str,
        interval: str,
        limit: int = 500
    ) -> List[Dict[str, Any]]:
        """Obtiene datos históricos para análisis."""
        pass


class ITradingClient(ABC):
    """Interfaz para cliente de trading."""

    @abstractmethod
    async def get_account_balance(self) -> List[Balance]:
        """Obtiene el balance de la cuenta."""
        pass

    @abstractmethod
    async def get_asset_balance(self, asset: str) -> Balance:
        """Obtiene el balance de un asset específico."""
        pass

    @abstractmethod
    async def create_order(self, order_request: OrderRequest) -> Order:
        """Crea una nueva orden."""
        pass

    @abstractmethod
    async def get_order_status(self, symbol: str, order_id: str) -> Order:
        """Obtiene el estado de una orden."""
        pass

    @abstractmethod
    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancela una orden."""
        pass


class IMLPredictor(ABC):
    """Interfaz para modelos de Machine Learning."""

    @abstractmethod
    async def predict(self, market_data: List[Dict[str, Any]]) -> TradingSignal:
        """Genera una señal de trading basada en datos de mercado."""
        pass

    @abstractmethod
    async def get_model_confidence(self) -> float:
        """Obtiene la confianza actual del modelo."""
        pass

    @abstractmethod
    async def update_model(self, new_data: List[Dict[str, Any]]) -> bool:
        """Actualiza el modelo con nuevos datos."""
        pass


class IRiskManager(ABC):
    """Interfaz para gestión de riesgos."""

    @abstractmethod
    async def validate_order(
        self,
        order_request: OrderRequest,
        account_balance: List[Balance]
    ) -> bool:
        """Valida si una orden cumple con las reglas de riesgo."""
        pass

    @abstractmethod
    async def calculate_position_size(
        self,
        symbol: str,
        signal_confidence: float,
        account_balance: Decimal
    ) -> Decimal:
        """Calcula el tamaño de posición óptimo."""
        pass

    @abstractmethod
    async def should_stop_trading(self) -> bool:
        """Determina si se debe parar el trading por riesgo."""
        pass


class ITradeRepository(ABC):
    """Interfaz para persistencia de trades."""

    @abstractmethod
    async def save_order(self, order: Order) -> bool:
        """Guarda una orden en la base de datos."""
        pass

    @abstractmethod
    async def get_order_by_id(self, order_id: str) -> Optional[Order]:
        """Obtiene una orden por su ID."""
        pass

    @abstractmethod
    async def get_recent_orders(
        self,
        symbol: str,
        limit: int = 100
    ) -> List[Order]:
        """Obtiene las órdenes recientes para un símbolo."""
        pass

    @abstractmethod
    async def save_signal(self, signal: TradingSignal) -> bool:
        """Guarda una señal de trading."""
        pass


class INotificationService(ABC):
    """Interfaz para notificaciones."""

    @abstractmethod
    async def send_trade_notification(self, order: Order) -> bool:
        """Envía notificación de trade ejecutado."""
        pass

    @abstractmethod
    async def send_error_notification(self, error: str, context: Dict[str, Any]) -> bool:
        """Envía notificación de error."""
        pass

    @abstractmethod
    async def send_signal_notification(self, signal: TradingSignal) -> bool:
        """Envía notificación de nueva señal."""
        pass


class ITradingStrategy(ABC):
    """Interfaz para estrategias de trading."""

    @abstractmethod
    async def should_trade(
        self,
        signal: TradingSignal,
        market_data: MarketData,
        account_balance: List[Balance]
    ) -> bool:
        """Determina si se debe ejecutar un trade basado en la estrategia."""
        pass

    @abstractmethod
    async def get_strategy_parameters(self) -> Dict[str, Any]:
        """Obtiene los parámetros actuales de la estrategia."""
        pass

    @abstractmethod
    async def update_strategy_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Actualiza los parámetros de la estrategia."""
        pass
