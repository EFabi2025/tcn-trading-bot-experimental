"""
🧪 EDUCATIONAL Binance Client - Trading Bot Experimental

Este módulo implementa un cliente educacional para Binance que:
- Opera SOLO en testnet para aprendizaje
- Implementa dry-run mode por defecto
- Demuestra patrones de integración con APIs externas
- Incluye manejo robusto de errores para educación

⚠️ EXPERIMENTAL: Solo para fines educacionales
"""

import asyncio
from typing import List, Optional, Dict, Any
from decimal import Decimal, ROUND_DOWN
from datetime import datetime, timezone

import structlog
from binance.client import Client as BinanceRestClient
from binance.exceptions import BinanceAPIException, BinanceOrderException

from ..interfaces.trading_interfaces import ITradingClient, IMarketDataProvider
from ..schemas.trading_schemas import (
    OrderRequestSchema, OrderSchema, BalanceSchema, MarketDataSchema
)
from ..core.config import TradingBotSettings
from ..core.logging_config import TradingLogger

logger = structlog.get_logger(__name__)


class EducationalBinanceClient(ITradingClient, IMarketDataProvider):
    """
    🎓 Cliente educacional de Binance para aprendizaje
    
    Características educacionales:
    - Dry-run mode por defecto (NO ejecuta trades reales)
    - Solo testnet para experimentos seguros
    - Logging educacional detallado
    - Manejo de errores didáctico
    """
    
    def __init__(self, settings: TradingBotSettings, trading_logger: TradingLogger):
        """
        Inicializa el cliente educacional de Binance
        
        Args:
            settings: Configuración del bot (debe tener dry_run=True)
            trading_logger: Logger estructurado para educación
        """
        self.settings = settings
        self.logger = trading_logger
        self._client: Optional[BinanceRestClient] = None
        self._is_testnet = True  # 🚨 SIEMPRE testnet para educación
        
        # 🛡️ Validación de seguridad educacional
        if not settings.dry_run:
            raise ValueError("🚨 EDUCATIONAL: Este cliente requiere dry_run=True")
        
        if not settings.binance_testnet:
            raise ValueError("🚨 EDUCATIONAL: Este cliente requiere testnet=True")
        
        self._initialize_client()
    
    def _initialize_client(self) -> None:
        """Inicializa el cliente de Binance (solo testnet)"""
        try:
            # 🎓 EDUCATIONAL: Configuración solo para testnet
            self._client = BinanceRestClient(
                api_key=self.settings.binance_api_key.get_secret_value(),
                api_secret=self.settings.binance_secret.get_secret_value(),
                testnet=True  # 🚨 FORZADO para educación
            )
            
            # Verificar conectividad educacional
            self._verify_connection()
            
            self.logger.log_system_event(
                "binance_client_initialized",
                testnet=True,
                dry_run=self.settings.dry_run
            )
            
        except Exception as e:
            self.logger.log_error(
                "binance_client_init_failed",
                error=str(e),
                educational_note="Verificar credenciales de testnet"
            )
            raise
    
    def _verify_connection(self) -> None:
        """Verifica conexión educacional con Binance testnet"""
        try:
            # Test básico de conectividad
            server_time = self._client.get_server_time()
            account_status = self._client.get_account_status()
            
            self.logger.log_system_event(
                "binance_connection_verified",
                server_time=server_time,
                account_status=account_status.get('data', 'unknown'),
                educational_note="Conexión testnet establecida"
            )
            
        except BinanceAPIException as e:
            self.logger.log_error(
                "binance_connection_failed",
                error_code=e.code,
                error_message=e.message,
                educational_tip="Verificar API keys de testnet en .env"
            )
            raise
    
    async def create_order(self, order_request: OrderRequestSchema) -> OrderSchema:
        """
        🎓 EDUCATIONAL: Crea una orden (DRY-RUN mode)
        
        En modo educacional, simula la orden sin ejecutarla realmente
        """
        self.logger.log_order_request(
            order_request.dict(),
            dry_run=True,
            educational_note="Modo educacional - NO se ejecuta trade real"
        )
        
        # 🛡️ Validación educacional
        if not self.settings.dry_run:
            raise ValueError("🚨 EDUCATIONAL: Solo dry-run permitido")
        
        try:
            # 🎓 SIMULAR orden para educación
            simulated_order = await self._simulate_order(order_request)
            
            self.logger.log_order_completed(
                simulated_order.dict(),
                dry_run=True,
                educational_note="Orden simulada exitosamente"
            )
            
            return simulated_order
            
        except Exception as e:
            self.logger.log_error(
                "educational_order_simulation_failed",
                error=str(e),
                order_symbol=order_request.symbol,
                educational_tip="Error en simulación educacional"
            )
            raise
    
    async def _simulate_order(self, order_request: OrderRequestSchema) -> OrderSchema:
        """
        🎓 Simula una orden para propósitos educacionales
        
        Genera datos realistas sin ejecutar trade real
        """
        # Obtener precio actual para simulación realista
        current_price = await self._get_current_price(order_request.symbol)
        
        # Simular fill price con pequeño slippage educacional
        slippage_factor = Decimal('1.001') if order_request.side == 'BUY' else Decimal('0.999')
        fill_price = current_price * slippage_factor
        
        # ID simulado para educación
        simulated_order_id = f"EDU_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return OrderSchema(
            id=simulated_order_id,
            symbol=order_request.symbol,
            side=order_request.side,
            type=order_request.type,
            quantity=order_request.quantity,
            price=fill_price,
            status="FILLED_SIMULATED",  # 🎓 Status educacional
            timestamp=datetime.now(timezone.utc),
            filled_quantity=order_request.quantity,
            remaining_quantity=Decimal('0'),
            average_price=fill_price,
            commission=fill_price * order_request.quantity * Decimal('0.001'),  # 0.1% simulado
            commission_asset="USDT"
        )
    
    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """🎓 EDUCATIONAL: Cancela orden (simulado en dry-run)"""
        self.logger.log_system_event(
            "educational_order_cancellation",
            symbol=symbol,
            order_id=order_id,
            dry_run=True,
            educational_note="Cancelación simulada"
        )
        
        # En modo educacional, simular cancelación exitosa
        return True
    
    async def get_order_status(self, symbol: str, order_id: str) -> OrderSchema:
        """🎓 EDUCATIONAL: Obtiene status de orden simulada"""
        # Para educación, simular orden completada
        current_price = await self._get_current_price(symbol)
        
        return OrderSchema(
            id=order_id,
            symbol=symbol,
            side="BUY",  # Ejemplo educacional
            type="MARKET",
            quantity=Decimal('0.001'),
            price=current_price,
            status="FILLED_SIMULATED",
            timestamp=datetime.now(timezone.utc),
            filled_quantity=Decimal('0.001'),
            remaining_quantity=Decimal('0'),
            average_price=current_price,
            commission=current_price * Decimal('0.001') * Decimal('0.001'),
            commission_asset="USDT"
        )
    
    async def get_balances(self) -> List[BalanceSchema]:
        """🎓 EDUCATIONAL: Obtiene balances de testnet"""
        try:
            account_info = self._client.get_account()
            balances = []
            
            for balance in account_info['balances']:
                free_balance = Decimal(balance['free'])
                locked_balance = Decimal(balance['locked'])
                
                # Solo incluir balances con valor para educación
                if free_balance > 0 or locked_balance > 0:
                    balances.append(BalanceSchema(
                        asset=balance['asset'],
                        free=free_balance,
                        locked=locked_balance,
                        total=free_balance + locked_balance
                    ))
            
            self.logger.log_balance_check(
                [b.dict() for b in balances],
                educational_note="Balances de testnet para educación"
            )
            
            return balances
            
        except BinanceAPIException as e:
            self.logger.log_error(
                "educational_balance_fetch_failed",
                error_code=e.code,
                error_message=e.message,
                educational_tip="Verificar permisos de API en testnet"
            )
            raise
    
    async def get_market_data(self, symbol: str) -> MarketDataSchema:
        """🎓 EDUCATIONAL: Obtiene datos de mercado en tiempo real"""
        try:
            # Obtener ticker 24h
            ticker = self._client.get_ticker(symbol=symbol)
            
            # Obtener orderbook
            orderbook = self._client.get_order_book(symbol=symbol, limit=5)
            
            # Obtener trades recientes
            recent_trades = self._client.get_recent_trades(symbol=symbol, limit=10)
            
            market_data = MarketDataSchema(
                symbol=symbol,
                price=Decimal(ticker['lastPrice']),
                bid_price=Decimal(orderbook['bids'][0][0]),
                ask_price=Decimal(orderbook['asks'][0][0]),
                volume=Decimal(ticker['volume']),
                price_change_24h=Decimal(ticker['priceChange']),
                price_change_percent_24h=Decimal(ticker['priceChangePercent']),
                high_24h=Decimal(ticker['highPrice']),
                low_24h=Decimal(ticker['lowPrice']),
                timestamp=datetime.now(timezone.utc),
                bid_volume=Decimal(orderbook['bids'][0][1]),
                ask_volume=Decimal(orderbook['asks'][0][1])
            )
            
            self.logger.log_market_data(
                market_data.dict(),
                educational_note="Datos reales de testnet para educación"
            )
            
            return market_data
            
        except BinanceAPIException as e:
            self.logger.log_error(
                "educational_market_data_failed",
                symbol=symbol,
                error_code=e.code,
                error_message=e.message,
                educational_tip="Verificar símbolo válido en testnet"
            )
            raise
    
    async def _get_current_price(self, symbol: str) -> Decimal:
        """Helper para obtener precio actual"""
        try:
            ticker = self._client.get_symbol_ticker(symbol=symbol)
            return Decimal(ticker['price'])
        except Exception:
            # Fallback educacional
            return Decimal('50000.0')  # Precio BTC ejemplo para educación
    
    async def get_symbol_info(self, symbol: str) -> Dict[str, Any]:
        """🎓 EDUCATIONAL: Obtiene información del símbolo"""
        try:
            exchange_info = self._client.get_exchange_info()
            
            for symbol_info in exchange_info['symbols']:
                if symbol_info['symbol'] == symbol:
                    return {
                        'symbol': symbol_info['symbol'],
                        'status': symbol_info['status'],
                        'baseAsset': symbol_info['baseAsset'],
                        'quoteAsset': symbol_info['quoteAsset'],
                        'minQty': symbol_info['filters'][2]['minQty'],
                        'maxQty': symbol_info['filters'][2]['maxQty'],
                        'stepSize': symbol_info['filters'][2]['stepSize'],
                        'minPrice': symbol_info['filters'][0]['minPrice'],
                        'maxPrice': symbol_info['filters'][0]['maxPrice'],
                        'tickSize': symbol_info['filters'][0]['tickSize'],
                        'educational_note': "Info de testnet para aprendizaje"
                    }
            
            raise ValueError(f"🎓 Símbolo {symbol} no encontrado en testnet")
            
        except BinanceAPIException as e:
            self.logger.log_error(
                "educational_symbol_info_failed",
                symbol=symbol,
                error_code=e.code,
                educational_tip="Verificar símbolo disponible en testnet"
            )
            raise
    
    def is_connected(self) -> bool:
        """🎓 Verifica conexión educacional"""
        try:
            if self._client:
                self._client.ping()
                return True
            return False
        except Exception:
            return False
    
    async def close(self) -> None:
        """🎓 Cierra conexión educacional"""
        self.logger.log_system_event(
            "educational_binance_client_closed",
            educational_note="Cliente educacional desconectado"
        )
        # No hay conexión persistente que cerrar en REST API
        pass 