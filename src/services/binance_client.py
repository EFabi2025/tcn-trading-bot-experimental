"""
🧪 EXPERIMENTAL Binance Client - Trading Bot Research

Este módulo implementa un cliente experimental para Binance que:
- Puede operar en testnet O producción según configuración
- Implementa dry-run mode configurable
- Demuestra patrones de integración con APIs externas
- Incluye manejo robusto de errores para investigación

⚠️ EXPERIMENTAL: Para investigación en trading algorítmico
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


class ExperimentalBinanceClient(ITradingClient, IMarketDataProvider):
    """
    🧪 Cliente experimental de Binance para investigación
    
    Características experimentales:
    - Dry-run mode configurable (para investigación segura)
    - Soporte para testnet Y producción
    - Logging detallado para análisis
    - Manejo de errores para investigación
    """
    
    def __init__(self, settings: TradingBotSettings, trading_logger: TradingLogger):
        """
        Inicializa el cliente experimental de Binance
        
        Args:
            settings: Configuración del bot
            trading_logger: Logger estructurado para investigación
        """
        self.settings = settings
        self.logger = trading_logger
        self._client: Optional[BinanceRestClient] = None
        self._is_testnet = settings.binance_testnet
        
        self._initialize_client()
    
    def _initialize_client(self) -> None:
        """Inicializa el cliente de Binance (testnet o producción)"""
        try:
            # 🧪 EXPERIMENTAL: Configuración según settings
            self._client = BinanceRestClient(
                api_key=self.settings.binance_api_key.get_secret_value(),
                api_secret=self.settings.binance_secret.get_secret_value(),
                testnet=self._is_testnet
            )
            
            # Verificar conectividad
            self._verify_connection()
            
            self.logger.log_system_event(
                "binance_client_initialized",
                testnet=self._is_testnet,
                dry_run=self.settings.dry_run,
                environment=self.settings.environment,
                experimental_note="Cliente Binance inicializado para investigación"
            )
            
        except Exception as e:
            self.logger.log_error(
                "binance_client_init_failed",
                error=str(e),
                testnet=self._is_testnet,
                research_note="Verificar credenciales de API"
            )
            raise
    
    def _verify_connection(self) -> None:
        """Verifica conexión experimental con Binance"""
        try:
            # Test básico de conectividad
            server_time = self._client.get_server_time()
            account_status = self._client.get_account_status()
            
            self.logger.log_system_event(
                "binance_connection_verified",
                server_time=server_time,
                account_status=account_status.get('data', 'unknown'),
                testnet=self._is_testnet,
                research_note="Conexión establecida para investigación"
            )
            
        except BinanceAPIException as e:
            self.logger.log_error(
                "binance_connection_failed",
                error_code=e.code,
                error_message=e.message,
                testnet=self._is_testnet,
                research_tip="Verificar API keys y permisos"
            )
            raise
    
    async def create_order(self, order_request: OrderRequestSchema) -> OrderSchema:
        """
        🧪 EXPERIMENTAL: Crea una orden (real o simulada según configuración)
        
        En modo dry_run=True: simula la orden
        En modo dry_run=False: ejecuta orden real en Binance
        """
        self.logger.log_order_request(
            order_request.dict(),
            dry_run=self.settings.dry_run,
            testnet=self._is_testnet,
            research_note="Procesando orden experimental"
        )
        
        try:
            if self.settings.dry_run:
                # 🧪 SIMULAR orden para investigación segura
                simulated_order = await self._simulate_order(order_request)
                
                self.logger.log_order_completed(
                    simulated_order.dict(),
                    dry_run=True,
                    research_note="Orden simulada para investigación"
                )
                
                return simulated_order
            else:
                # 🚨 EJECUTAR orden REAL en Binance
                real_order = await self._execute_real_order(order_request)
                
                self.logger.log_order_completed(
                    real_order.dict(),
                    dry_run=False,
                    testnet=self._is_testnet,
                    research_note="Orden REAL ejecutada en Binance"
                )
                
                return real_order
            
        except Exception as e:
            self.logger.log_error(
                "experimental_order_failed",
                error=str(e),
                order_symbol=order_request.symbol,
                dry_run=self.settings.dry_run,
                research_tip="Error en ejecución de orden experimental"
            )
            raise
    
    async def _execute_real_order(self, order_request: OrderRequestSchema) -> OrderSchema:
        """
        🚨 Ejecuta orden REAL en Binance
        
        CUIDADO: Esta función ejecuta trades reales con dinero real
        """
        try:
            # Preparar parámetros para Binance API
            order_params = {
                'symbol': order_request.symbol,
                'side': order_request.side.upper(),
                'type': order_request.type.upper(),
                'quantity': float(order_request.quantity),
            }
            
            # Añadir precio si es orden LIMIT
            if order_request.type.upper() == 'LIMIT':
                order_params['price'] = float(order_request.price)
                order_params['timeInForce'] = 'GTC'  # Good Till Cancelled
            
            # 🚨 EJECUTAR ORDEN REAL
            result = self._client.create_order(**order_params)
            
            # Convertir respuesta de Binance a nuestro schema
            order = OrderSchema(
                id=result['orderId'],
                symbol=result['symbol'],
                side=result['side'],
                type=result['type'],
                quantity=Decimal(result['origQty']),
                price=Decimal(result.get('price', order_request.price)),
                status=result['status'],
                timestamp=datetime.fromtimestamp(result['transactTime'] / 1000, timezone.utc),
                filled_quantity=Decimal(result.get('executedQty', '0')),
                remaining_quantity=Decimal(result['origQty']) - Decimal(result.get('executedQty', '0')),
                average_price=Decimal(result.get('cummulativeQuoteQty', '0')) / Decimal(result.get('executedQty', '1')) if float(result.get('executedQty', '0')) > 0 else Decimal('0'),
                commission=Decimal('0'),  # Se obtiene de otro endpoint
                commission_asset="USDT"
            )
            
            return order
            
        except BinanceAPIException as e:
            self.logger.log_error(
                "real_order_execution_failed",
                error_code=e.code,
                error_message=e.message,
                order_symbol=order_request.symbol,
                research_note="Error en orden real de Binance"
            )
            raise
        except Exception as e:
            self.logger.log_error(
                "real_order_unexpected_error",
                error=str(e),
                order_symbol=order_request.symbol,
                research_note="Error inesperado en orden real"
            )
            raise
    
    async def _simulate_order(self, order_request: OrderRequestSchema) -> OrderSchema:
        """
        🧪 Simula una orden para investigación segura
        
        Genera datos realistas basados en precio actual de mercado
        """
        # Obtener precio actual para simulación realista
        current_price = await self._get_current_price(order_request.symbol)
        
        # Simular fill price con pequeño slippage
        slippage_factor = Decimal('1.001') if order_request.side == 'BUY' else Decimal('0.999')
        fill_price = current_price * slippage_factor
        
        # ID simulado para investigación
        simulated_order_id = f"SIM_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return OrderSchema(
            id=simulated_order_id,
            symbol=order_request.symbol,
            side=order_request.side,
            type=order_request.type,
            quantity=order_request.quantity,
            price=fill_price,
            status="FILLED_SIMULATED",  # 🧪 Status experimental
            timestamp=datetime.now(timezone.utc),
            filled_quantity=order_request.quantity,
            remaining_quantity=Decimal('0'),
            average_price=fill_price,
            commission=fill_price * order_request.quantity * Decimal('0.001'),  # 0.1% simulado
            commission_asset="USDT"
        )
    
    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """🧪 Cancela orden (real o simulada según configuración)"""
        self.logger.log_system_event(
            "experimental_order_cancellation",
            symbol=symbol,
            order_id=order_id,
            dry_run=self.settings.dry_run,
            research_note="Cancelando orden experimental"
        )
        
        if self.settings.dry_run:
            # En modo simulado, siempre exitoso
            return True
        else:
            try:
                # 🚨 CANCELAR orden REAL en Binance
                result = self._client.cancel_order(symbol=symbol, orderId=order_id)
                return result['status'] == 'CANCELED'
            except BinanceAPIException as e:
                self.logger.log_error(
                    "real_order_cancellation_failed",
                    error_code=e.code,
                    error_message=e.message,
                    symbol=symbol,
                    order_id=order_id
                )
                return False
    
    async def get_order_status(self, symbol: str, order_id: str) -> OrderSchema:
        """🧪 Obtiene status de orden (real o simulada)"""
        if self.settings.dry_run:
            # Para simulación, retornar orden completada
            current_price = await self._get_current_price(symbol)
            
            return OrderSchema(
                id=order_id,
                symbol=symbol,
                side="BUY",  # Ejemplo
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
        else:
            try:
                # 🚨 CONSULTAR orden REAL en Binance
                result = self._client.get_order(symbol=symbol, orderId=order_id)
                
                return OrderSchema(
                    id=result['orderId'],
                    symbol=result['symbol'],
                    side=result['side'],
                    type=result['type'],
                    quantity=Decimal(result['origQty']),
                    price=Decimal(result.get('price', '0')),
                    status=result['status'],
                    timestamp=datetime.fromtimestamp(result['time'] / 1000, timezone.utc),
                    filled_quantity=Decimal(result.get('executedQty', '0')),
                    remaining_quantity=Decimal(result['origQty']) - Decimal(result.get('executedQty', '0')),
                    average_price=Decimal(result.get('cummulativeQuoteQty', '0')) / Decimal(result.get('executedQty', '1')) if float(result.get('executedQty', '0')) > 0 else Decimal('0'),
                    commission=Decimal('0'),
                    commission_asset="USDT"
                )
            except BinanceAPIException as e:
                self.logger.log_error(
                    "real_order_status_failed",
                    error_code=e.code,
                    error_message=e.message,
                    symbol=symbol,
                    order_id=order_id
                )
                raise
    
    async def get_balances(self) -> List[BalanceSchema]:
        """🧪 Obtiene balances reales de Binance"""
        try:
            account_info = self._client.get_account()
            balances = []
            
            for balance in account_info['balances']:
                free_balance = Decimal(balance['free'])
                locked_balance = Decimal(balance['locked'])
                
                # Solo incluir balances con valor
                if free_balance > 0 or locked_balance > 0:
                    balances.append(BalanceSchema(
                        asset=balance['asset'],
                        free=free_balance,
                        locked=locked_balance,
                        total=free_balance + locked_balance
                    ))
            
            self.logger.log_balance_check(
                [b.dict() for b in balances],
                testnet=self._is_testnet,
                research_note="Balances obtenidos para investigación"
            )
            
            return balances
            
        except BinanceAPIException as e:
            self.logger.log_error(
                "experimental_balance_fetch_failed",
                error_code=e.code,
                error_message=e.message,
                testnet=self._is_testnet,
                research_tip="Verificar permisos de API"
            )
            raise
    
    async def get_market_data(self, symbol: str) -> MarketDataSchema:
        """🧪 Obtiene datos de mercado en tiempo real"""
        try:
            # Obtener ticker 24h
            ticker = self._client.get_ticker(symbol=symbol)
            
            # Obtener orderbook
            orderbook = self._client.get_order_book(symbol=symbol, limit=5)
            
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
                testnet=self._is_testnet,
                research_note="Datos de mercado para investigación"
            )
            
            return market_data
            
        except BinanceAPIException as e:
            self.logger.log_error(
                "experimental_market_data_failed",
                symbol=symbol,
                error_code=e.code,
                error_message=e.message,
                testnet=self._is_testnet,
                research_tip="Verificar símbolo válido"
            )
            raise
    
    async def _get_current_price(self, symbol: str) -> Decimal:
        """Helper para obtener precio actual"""
        try:
            ticker = self._client.get_symbol_ticker(symbol=symbol)
            return Decimal(ticker['price'])
        except Exception:
            # Fallback para investigación
            return Decimal('50000.0')  # Precio BTC ejemplo
    
    async def get_symbol_info(self, symbol: str) -> Dict[str, Any]:
        """🧪 Obtiene información del símbolo"""
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
                        'research_note': "Info obtenida para investigación"
                    }
            
            raise ValueError(f"🧪 Símbolo {symbol} no encontrado")
            
        except BinanceAPIException as e:
            self.logger.log_error(
                "experimental_symbol_info_failed",
                symbol=symbol,
                error_code=e.code,
                research_tip="Verificar símbolo disponible"
            )
            raise
    
    def is_connected(self) -> bool:
        """🧪 Verifica conexión"""
        try:
            if self._client:
                self._client.ping()
                return True
            return False
        except Exception:
            return False
    
    async def close(self) -> None:
        """🧪 Cierra conexión"""
        self.logger.log_system_event(
            "experimental_binance_client_closed",
            research_note="Cliente experimental desconectado"
        )
        # No hay conexión persistente que cerrar en REST API
        pass 