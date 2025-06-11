#!/usr/bin/env python3
"""
💼 PROFESSIONAL PORTFOLIO MANAGER
Sistema avanzado para gestión de portafolio con datos reales de Binance
Replica y mejora el formato del bot TCN anterior
VERSIÓN CORREGIDA: Múltiples posiciones por par con precios de entrada reales
"""

import asyncio
import aiohttp
import time
import hmac
import hashlib
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Position:
    """📈 Posición individual en el portafolio"""
    symbol: str
    side: str  # BUY o SELL
    size: float  # Cantidad del activo
    entry_price: float
    current_price: float
    market_value: float  # Valor actual en USDT
    unrealized_pnl_usd: float
    unrealized_pnl_percent: float
    entry_time: datetime
    duration_minutes: int
    order_id: Optional[str] = None  # ID de la orden original
    batch_id: Optional[str] = None  # Para agrupar órdenes relacionadas
    
    # ✅ NUEVO: Sistema de Trailing Stop Profesional
    trailing_stop_active: bool = False
    trailing_stop_price: Optional[float] = None
    trailing_stop_percent: float = 2.0  # Default 2%
    highest_price_since_entry: Optional[float] = None  # Para tracking del máximo
    lowest_price_since_entry: Optional[float] = None   # Para shorts
    trailing_activation_threshold: float = 1.0  # Activar trailing después de +1% ganancia
    last_trailing_update: Optional[datetime] = None
    trailing_movements: int = 0  # Contador de movimientos del trailing
    
    # Stop Loss y Take Profit tradicionales
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    stop_loss_percent: float = 3.0  # Default 3%
    take_profit_percent: float = 6.0  # Default 6%

@dataclass
class Asset:
    """🪙 Activo individual en el portafolio"""
    symbol: str
    free: float
    locked: float
    total: float
    usd_value: float
    percentage_of_portfolio: float

@dataclass
class TradeOrder:
    """📋 Orden de trading individual"""
    order_id: str
    symbol: str
    side: str  # BUY/SELL
    quantity: float
    price: float
    executed_qty: float
    cumulative_quote_qty: float
    time: datetime
    status: str

@dataclass
class PortfolioSnapshot:
    """📊 Snapshot completo del portafolio"""
    timestamp: datetime
    total_balance_usd: float
    free_usdt: float
    total_unrealized_pnl: float
    total_unrealized_pnl_percent: float
    active_positions: List[Position]
    all_assets: List[Asset]
    position_count: int
    max_positions: int
    total_trades_today: int

class ProfessionalPortfolioManager:
    """💼 Gestor Profesional de Portafolio"""
    
    def __init__(self, api_key: str, secret_key: str, base_url: str = "https://testnet.binance.vision"):
        """🚀 Inicializar Portfolio Manager"""
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = base_url
        
        # Cache de precios para evitar llamadas innecesarias
        self.price_cache = {}
        self.last_price_update = None
        
        # ✅ NUEVO: Cache de órdenes para tracking de posiciones
        self.orders_cache = {}
        self.last_orders_update = None
        
        # Configuración
        self.max_positions = 5
        self.min_position_value = 10.0  # USD mínimo por posición
        
        # ✅ NUEVO: Configuración para historial de órdenes
        self.days_to_lookback = 30  # Días hacia atrás para buscar órdenes
        
        # Métricas
        self.api_calls_count = 0
        self.last_snapshot_time = None
        
    def _generate_signature(self, params: str) -> str:
        """🔐 Generar firma HMAC SHA256 para Binance"""
        return hmac.new(
            self.secret_key.encode('utf-8'),
            params.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    async def _make_authenticated_request(self, endpoint: str, params: Dict = None) -> Dict:
        """🔗 Realizar petición autenticada a Binance"""
        if params is None:
            params = {}
        
        # Añadir timestamp
        params['timestamp'] = int(time.time() * 1000)
        
        # Crear query string
        query_string = '&'.join([f"{key}={value}" for key, value in params.items()])
        
        # Generar firma
        signature = self._generate_signature(query_string)
        query_string += f"&signature={signature}"
        
        # Headers
        headers = {
            'X-MBX-APIKEY': self.api_key
        }
        
        # Realizar petición
        url = f"{self.base_url}/api/v3/{endpoint}?{query_string}"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as response:
                self.api_calls_count += 1
                
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise Exception(f"Error API Binance: {response.status} - {error_text}")
    
    async def get_current_price(self, symbol: str) -> float:
        """💲 Obtener precio actual de un símbolo"""
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/api/v3/ticker/price"
                params = {'symbol': symbol}
                async with session.get(url, params=params) as response:
                    self.api_calls_count += 1
                    if response.status == 200:
                        data = await response.json()
                        price = float(data['price'])
                        self.price_cache[symbol] = price
                        return price
        except Exception as e:
            print(f"❌ Error obteniendo precio {symbol}: {e}")
        return 0.0
    
    async def update_all_prices(self, symbols: List[str]) -> Dict[str, float]:
        """💲 Actualizar precios de múltiples símbolos en paralelo"""
        tasks = [self.get_current_price(symbol) for symbol in symbols]
        prices = await asyncio.gather(*tasks)
        
        price_dict = {}
        for symbol, price in zip(symbols, prices):
            if price > 0:
                price_dict[symbol] = price
        
        self.last_price_update = datetime.now()
        return price_dict
    
    async def get_account_balances(self) -> Dict[str, Dict]:
        """💰 Obtener balances de la cuenta"""
        try:
            data = await self._make_authenticated_request("account")
            
            balances = {}
            for balance in data.get('balances', []):
                asset = balance['asset']
                free = float(balance['free'])
                locked = float(balance['locked'])
                total = free + locked
                
                if total > 0:  # Solo activos con balance > 0
                    balances[asset] = {
                        'free': free,
                        'locked': locked,
                        'total': total
                    }
            
            return balances
            
        except Exception as e:
            print(f"❌ Error obteniendo balances: {e}")
            return {}
    
    async def get_order_history(self, symbol: str = None, days_back: int = None) -> List[TradeOrder]:
        """📋 Obtener historial de órdenes ejecutadas"""
        try:
            if days_back is None:
                days_back = self.days_to_lookback
            
            # Calcular timestamp de inicio (días hacia atrás)
            start_time = int((datetime.now() - timedelta(days=days_back)).timestamp() * 1000)
            
            orders = []
            
            if symbol:
                # Obtener órdenes para un símbolo específico
                params = {
                    'symbol': symbol,
                    'startTime': start_time,
                    'limit': 1000  # Máximo por request
                }
                
                data = await self._make_authenticated_request("allOrders", params)
                
                for order in data:
                    if order['status'] == 'FILLED':  # Solo órdenes ejecutadas
                        trade_order = TradeOrder(
                            order_id=str(order['orderId']),
                            symbol=order['symbol'],
                            side=order['side'],
                            quantity=float(order['origQty']),
                            price=float(order['price']) if order['price'] != '0.00000000' else float(order['cummulativeQuoteQty']) / float(order['executedQty']),
                            executed_qty=float(order['executedQty']),
                            cumulative_quote_qty=float(order['cummulativeQuoteQty']),
                            time=datetime.fromtimestamp(order['time'] / 1000),
                            status=order['status']
                        )
                        orders.append(trade_order)
            else:
                # Obtener órdenes para todos los símbolos activos
                balances = await self.get_account_balances()
                
                for asset in balances.keys():
                    if asset != 'USDT':
                        try:
                            symbol_orders = await self.get_order_history(f"{asset}USDT", days_back)
                            orders.extend(symbol_orders)
                        except Exception as e:
                            print(f"⚠️ Error obteniendo órdenes para {asset}USDT: {e}")
                            continue
            
            return sorted(orders, key=lambda x: x.time, reverse=True)
            
        except Exception as e:
            print(f"❌ Error obteniendo historial de órdenes: {e}")
            return []
    
    def group_orders_into_positions(self, orders: List[TradeOrder], current_balances: Dict[str, Dict]) -> List[Position]:
        """🔄 Agrupar órdenes en posiciones individuales usando FIFO"""
        try:
            positions = []
            
            # Agrupar órdenes por símbolo
            orders_by_symbol = {}
            for order in orders:
                if order.symbol not in orders_by_symbol:
                    orders_by_symbol[order.symbol] = []
                orders_by_symbol[order.symbol].append(order)
            
            # Procesar cada símbolo
            for symbol, symbol_orders in orders_by_symbol.items():
                # Ordenar órdenes por tiempo (más antiguas primero)
                symbol_orders.sort(key=lambda x: x.time)
                
                # Obtener balance actual del activo
                asset = symbol.replace('USDT', '')
                current_balance = current_balances.get(asset, {}).get('total', 0.0)
                
                if current_balance <= 0:
                    continue  # No hay balance actual, skip
                
                # Algoritmo FIFO para determinar posiciones actuales
                remaining_balance = current_balance
                buy_orders = [order for order in symbol_orders if order.side == 'BUY']
                sell_orders = [order for order in symbol_orders if order.side == 'SELL']
                
                # Primero, restar todas las ventas del balance inicial acumulado
                total_bought = sum(order.executed_qty for order in buy_orders)
                total_sold = sum(order.executed_qty for order in sell_orders)
                
                # Si el balance actual es menor que el total comprado menos vendido,
                # significa que algunas posiciones fueron cerradas
                
                # Crear posiciones basadas en órdenes de compra que aún están "abiertas"
                current_position_qty = remaining_balance
                
                # Procesar órdenes de compra desde la más reciente (LIFO para mostrar mejor info)
                for buy_order in reversed(buy_orders):
                    if current_position_qty <= 0:
                        break
                    
                    # Determinar cuánta cantidad de esta orden aún está en posición
                    qty_from_this_order = min(buy_order.executed_qty, current_position_qty)
                    
                    if qty_from_this_order > 0:
                        # Crear posición para esta parte
                        current_price = self.price_cache.get(symbol, buy_order.price)
                        market_value = qty_from_this_order * current_price
                        
                        # Calcular PnL
                        entry_value = qty_from_this_order * buy_order.price
                        pnl_usd = market_value - entry_value
                        pnl_percent = (pnl_usd / entry_value) * 100 if entry_value > 0 else 0
                        
                        # Calcular duración
                        duration_minutes = int((datetime.now() - buy_order.time).total_seconds() / 60)
                        
                        new_position = Position(
                            symbol=symbol,
                            side='BUY',
                            size=qty_from_this_order,
                            entry_price=buy_order.price,
                            current_price=current_price,
                            market_value=market_value,
                            unrealized_pnl_usd=pnl_usd,
                            unrealized_pnl_percent=pnl_percent,
                            entry_time=buy_order.time,
                            duration_minutes=duration_minutes,
                            order_id=f"{len(orders)}ord_{buy_order.order_id}",  # ID único para la posición
                            batch_id=buy_order.order_id
                        )
                        
                        # ✅ NUEVO: Inicializar stops para nueva posición
                        new_position = self.initialize_position_stops(new_position)
                        positions.append(new_position)
                        current_position_qty -= qty_from_this_order
            
            return positions
            
        except Exception as e:
            print(f"❌ Error agrupando órdenes en posiciones: {e}")
            return []

    async def get_portfolio_snapshot(self) -> PortfolioSnapshot:
        """📊 Obtener snapshot completo del portafolio"""
        try:
            print("📊 Obteniendo snapshot del portafolio...")
            
            # 1. Obtener balances
            balances = await self.get_account_balances()
            if not balances:
                raise Exception("No se pudieron obtener balances")
            
            # 2. Identificar símbolos para obtener precios
            symbols_needed = []
            for asset in balances.keys():
                if asset != 'USDT':
                    symbols_needed.append(f"{asset}USDT")
            
            # 3. Obtener precios actuales
            if symbols_needed:
                prices = await self.update_all_prices(symbols_needed)
            else:
                prices = {}
            
            # 4. ✅ NUEVO: Obtener historial de órdenes para posiciones reales
            print("📋 Obteniendo historial de órdenes...")
            all_orders = await self.get_order_history(days_back=self.days_to_lookback)
            print(f"   📄 Encontradas {len(all_orders)} órdenes ejecutadas")
            
            # 5. ✅ NUEVO: Agrupar órdenes en posiciones individuales
            print("🔄 Agrupando órdenes en posiciones...")
            individual_positions = self.group_orders_into_positions(all_orders, balances)
            print(f"   📈 Identificadas {len(individual_positions)} posiciones individuales")
            
            # 6. Calcular valor de cada activo
            all_assets = []
            total_portfolio_value = 0.0
            free_usdt = balances.get('USDT', {}).get('free', 0.0)
            
            for asset, balance_info in balances.items():
                if balance_info['total'] > 0:
                    if asset == 'USDT':
                        usd_value = balance_info['total']
                    else:
                        symbol = f"{asset}USDT"
                        price = prices.get(symbol, 0.0)
                        usd_value = balance_info['total'] * price if price > 0 else 0.0
                    
                    total_portfolio_value += usd_value
                    
                    asset_obj = Asset(
                        symbol=asset,
                        free=balance_info['free'],
                        locked=balance_info['locked'],
                        total=balance_info['total'],
                        usd_value=usd_value,
                        percentage_of_portfolio=0.0  # Se calculará después
                    )
                    all_assets.append(asset_obj)
            
            # 7. Calcular porcentajes
            for asset in all_assets:
                asset.percentage_of_portfolio = (asset.usd_value / total_portfolio_value * 100) if total_portfolio_value > 0 else 0.0
            
            # 8. ✅ NUEVO: Filtrar posiciones por valor mínimo
            active_positions = [pos for pos in individual_positions 
                             if pos.market_value >= self.min_position_value]
            
            # 9. Calcular PnL total
            total_unrealized_pnl = sum(pos.unrealized_pnl_usd for pos in active_positions)
            
            # 10. Crear snapshot
            snapshot = PortfolioSnapshot(
                timestamp=datetime.now(),
                total_balance_usd=total_portfolio_value,
                free_usdt=free_usdt,
                total_unrealized_pnl=total_unrealized_pnl,
                total_unrealized_pnl_percent=(total_unrealized_pnl / total_portfolio_value * 100) if total_portfolio_value > 0 else 0.0,
                active_positions=active_positions,
                all_assets=all_assets,
                position_count=len(active_positions),
                max_positions=self.max_positions,
                total_trades_today=len([o for o in all_orders if o.time.date() == datetime.now().date()])
            )
            
            self.last_snapshot_time = datetime.now()
            print(f"✅ Snapshot obtenido: {len(all_assets)} activos, {len(active_positions)} posiciones individuales")
            
            return snapshot
            
        except Exception as e:
            print(f"❌ Error obteniendo snapshot: {e}")
            raise
    
    def format_tcn_style_report(self, snapshot: PortfolioSnapshot) -> str:
        """🎨 Formatear reporte estilo TCN para Discord"""
        try:
            now = snapshot.timestamp
            
            # Header
            report = f"**🚀 TCN SIGNALS - {now.strftime('%H:%M:%S')}**\n"
            report += f"📊 **Recomendaciones del Modelo Profesional**\n\n"
            
            # Posiciones activas - ✅ MEJORADO: Mostrar posiciones individuales
            if snapshot.active_positions:
                report += f"**📈 POSICIONES ACTIVAS ({len(snapshot.active_positions)})**\n"
                
                # Agrupar por símbolo para mejor visualización
                positions_by_symbol = {}
                for pos in snapshot.active_positions:
                    if pos.symbol not in positions_by_symbol:
                        positions_by_symbol[pos.symbol] = []
                    positions_by_symbol[pos.symbol].append(pos)
                
                for symbol, positions in positions_by_symbol.items():
                    if len(positions) == 1:
                        # Una sola posición para este símbolo
                        pos = positions[0]
                        pnl_sign = "+" if pos.unrealized_pnl_usd >= 0 else ""
                        pnl_color = "🟢" if pos.unrealized_pnl_usd >= 0 else "🔴"
                        
                        report += f"**{pos.symbol}: {pos.side}**\n"
                        report += f"└ ${pos.entry_price:,.2f} → ${pos.current_price:,.2f} "
                        report += f"({pnl_sign}{pos.unrealized_pnl_percent:.2f}% = ${pnl_sign}{pos.unrealized_pnl_usd:.2f}) {pnl_color}\n"
                        report += f"   💰 Cantidad: {pos.size:.6f} | 🕐 {pos.duration_minutes}min"
                        
                        # ✅ NUEVO: Mostrar estado del trailing stop
                        if hasattr(pos, 'trailing_stop_active') and pos.trailing_stop_active:
                            report += f" | 📈 Trail: ${pos.trailing_stop_price:.2f}"
                        
                        report += "\n\n"
                    else:
                        # Múltiples posiciones para este símbolo
                        report += f"**{symbol}: MÚLTIPLES POSICIONES ({len(positions)})**\n"
                        
                        total_pnl = sum(p.unrealized_pnl_usd for p in positions)
                        total_pnl_sign = "+" if total_pnl >= 0 else ""
                        total_pnl_color = "🟢" if total_pnl >= 0 else "🔴"
                        
                        for i, pos in enumerate(positions, 1):
                            pnl_sign = "+" if pos.unrealized_pnl_usd >= 0 else ""
                            pnl_color = "🟢" if pos.unrealized_pnl_usd >= 0 else "🔴"
                            
                            report += f"├─ **Pos #{i}:** ${pos.entry_price:,.2f} → ${pos.current_price:,.2f} "
                            report += f"({pnl_sign}{pos.unrealized_pnl_percent:.2f}% = ${pnl_sign}{pos.unrealized_pnl_usd:.2f}) {pnl_color}\n"
                            report += f"│  💰 {pos.size:.6f} | 🕐 {pos.duration_minutes}min"
                            
                            # ✅ NUEVO: Estado trailing stop por posición
                            if hasattr(pos, 'trailing_stop_active') and pos.trailing_stop_active:
                                report += f" | 📈 Trail: ${pos.trailing_stop_price:.2f}"
                            
                            report += "\n"
                        
                        report += f"└─ **TOTAL:** ${total_pnl_sign}{total_pnl:.2f} {total_pnl_color}\n\n"
            else:
                report += "**📈 POSICIONES ACTIVAS (0)**\n"
                report += "└ Sin posiciones activas\n\n"
            
            # Resumen rápido
            report += "**⚡ RESUMEN RÁPIDO**\n"
            report += f"💰 **USDT Libre:** ${snapshot.free_usdt:,.2f}\n"
            
            pnl_sign = "+" if snapshot.total_unrealized_pnl >= 0 else ""
            pnl_emoji = "📈" if snapshot.total_unrealized_pnl >= 0 else "📉"
            report += f"{pnl_emoji} **P&L No Realizado:** ${pnl_sign}{snapshot.total_unrealized_pnl:.2f}\n"
            
            report += f"🎯 **Posiciones:** {snapshot.position_count}/{snapshot.max_positions}\n"
            report += f"📊 **Trades Totales:** {snapshot.total_trades_today}\n\n"
            
            # Detalle del portafolio
            report += "**💼 DETALLE DEL PORTAFOLIO**\n"
            
            # Ordenar activos por valor USD (mayor a menor)
            sorted_assets = sorted(snapshot.all_assets, key=lambda x: x.usd_value, reverse=True)
            
            for asset in sorted_assets:
                if asset.usd_value >= 0.01:  # Solo mostrar activos con valor > $0.01
                    if asset.symbol == 'USDT':
                        report += f"💵 **{asset.symbol}:** ${asset.total:,.2f}\n"
                    else:
                        report += f"🪙 **{asset.symbol}:** {asset.total:.6f} (${asset.usd_value:,.2f})\n"
            
            report += f"\n💎 **VALOR TOTAL: ${snapshot.total_balance_usd:,.2f}**\n"
            
            # Footer
            report += f"\n🔄 *Actualización cada 5 min • {now.strftime('%d/%m/%y, %H:%M')}*"
            
            return report
            
        except Exception as e:
            print(f"❌ Error formateando reporte: {e}")
            return f"❌ Error generando reporte: {e}"
    
    def format_compact_report(self, snapshot: PortfolioSnapshot) -> str:
        """📱 Formatear reporte compacto para notificaciones"""
        try:
            total_value = snapshot.total_balance_usd
            pnl = snapshot.total_unrealized_pnl
            positions = len(snapshot.active_positions)
            
            pnl_emoji = "📈" if pnl >= 0 else "📉"
            pnl_sign = "+" if pnl >= 0 else ""
            
            return (f"💼 Portfolio: ${total_value:,.2f} | "
                   f"{pnl_emoji} PnL: ${pnl_sign}{pnl:.2f} | "
                   f"🎯 Pos: {positions}/{snapshot.max_positions}")
            
        except Exception as e:
            return f"❌ Error: {e}"

    # ✅ NUEVO: Sistema de Trailing Stop Profesional
    
    def initialize_position_stops(self, position: Position) -> Position:
        """🛡️ Inicializar stops tradicionales y trailing para nueva posición"""
        try:
            # Configurar Stop Loss y Take Profit tradicionales
            if position.side == 'BUY':
                position.stop_loss_price = position.entry_price * (1 - position.stop_loss_percent / 100)
                position.take_profit_price = position.entry_price * (1 + position.take_profit_percent / 100)
                position.highest_price_since_entry = position.entry_price
                position.lowest_price_since_entry = None
            else:  # SELL (para futuros)
                position.stop_loss_price = position.entry_price * (1 + position.stop_loss_percent / 100)
                position.take_profit_price = position.entry_price * (1 - position.take_profit_percent / 100)
                position.lowest_price_since_entry = position.entry_price
                position.highest_price_since_entry = None
            
            # Trailing stop inicialmente inactivo
            position.trailing_stop_active = False
            position.trailing_stop_price = None
            position.last_trailing_update = datetime.now()
            position.trailing_movements = 0
            
            print(f"🛡️ Stops inicializados para {position.symbol} Pos #{position.order_id}:")
            print(f"   📍 Entrada: ${position.entry_price:.4f}")
            print(f"   🛑 Stop Loss: ${position.stop_loss_price:.4f} (-{position.stop_loss_percent}%)")
            print(f"   🎯 Take Profit: ${position.take_profit_price:.4f} (+{position.take_profit_percent}%)")
            print(f"   📈 Trailing: INACTIVO (activar en +{position.trailing_activation_threshold}%)")
            
            return position
            
        except Exception as e:
            print(f"❌ Error inicializando stops para {position.symbol}: {e}")
            return position
    
    def update_trailing_stop_professional(self, position: Position, current_price: float) -> Tuple[Position, bool, str]:
        """📈 Sistema profesional de Trailing Stop por posición individual"""
        try:
            stop_triggered = False
            trigger_reason = ""
            
            if position.side == 'BUY':
                # ✅ LONG POSITION LOGIC
                
                # 1. Actualizar precio máximo histórico
                if position.highest_price_since_entry is None or current_price > position.highest_price_since_entry:
                    position.highest_price_since_entry = current_price
                
                # 2. Calcular ganancia actual
                current_pnl_percent = ((current_price - position.entry_price) / position.entry_price) * 100
                
                # 3. Verificar si debe activarse el trailing stop
                if not position.trailing_stop_active and current_pnl_percent >= position.trailing_activation_threshold:
                    position.trailing_stop_active = True
                    position.trailing_stop_price = current_price * (1 - position.trailing_stop_percent / 100)
                    position.last_trailing_update = datetime.now()
                    
                    print(f"📈 TRAILING ACTIVADO {position.symbol} Pos #{position.order_id}:")
                    print(f"   🎯 Ganancia: +{current_pnl_percent:.2f}% (umbral: +{position.trailing_activation_threshold}%)")
                    print(f"   📈 Trailing Stop inicial: ${position.trailing_stop_price:.4f}")
                
                # 4. Actualizar trailing stop si está activo
                elif position.trailing_stop_active:
                    new_trailing_price = position.highest_price_since_entry * (1 - position.trailing_stop_percent / 100)
                    
                    # Solo mover trailing stop hacia arriba (más favorable)
                    if new_trailing_price > position.trailing_stop_price:
                        old_price = position.trailing_stop_price
                        position.trailing_stop_price = new_trailing_price
                        position.last_trailing_update = datetime.now()
                        position.trailing_movements += 1
                        
                        profit_protection = ((position.trailing_stop_price - position.entry_price) / position.entry_price) * 100
                        
                        print(f"📈 TRAILING MOVIDO {position.symbol} Pos #{position.order_id}:")
                        print(f"   🔄 ${old_price:.4f} → ${new_trailing_price:.4f}")
                        print(f"   🏔️ Máximo: ${position.highest_price_since_entry:.4f}")
                        print(f"   🛡️ Protegiendo: +{profit_protection:.2f}% ganancia")
                        print(f"   📊 Movimiento #{position.trailing_movements}")
                
                # 5. Verificar si se debe cerrar por trailing stop
                if position.trailing_stop_active and current_price <= position.trailing_stop_price:
                    stop_triggered = True
                    trigger_reason = "TRAILING_STOP"
                    
                    final_pnl = ((position.trailing_stop_price - position.entry_price) / position.entry_price) * 100
                    max_profit = ((position.highest_price_since_entry - position.entry_price) / position.entry_price) * 100
                    
                    print(f"🛑 TRAILING STOP EJECUTADO {position.symbol} Pos #{position.order_id}:")
                    print(f"   📉 Precio: ${current_price:.4f} <= Trailing: ${position.trailing_stop_price:.4f}")
                    print(f"   💰 PnL Final: +{final_pnl:.2f}%")
                    print(f"   🏔️ Máximo alcanzado: +{max_profit:.2f}%")
                    print(f"   📈 Movimientos trailing: {position.trailing_movements}")
                
                # 6. Verificar stop loss tradicional (solo si trailing no está activo o es menor)
                elif current_price <= position.stop_loss_price:
                    if not position.trailing_stop_active or position.stop_loss_price > position.trailing_stop_price:
                        stop_triggered = True
                        trigger_reason = "STOP_LOSS"
                        
                        loss_pnl = ((position.stop_loss_price - position.entry_price) / position.entry_price) * 100
                        print(f"🛑 STOP LOSS TRADICIONAL {position.symbol} Pos #{position.order_id}: {loss_pnl:.2f}%")
            
            else:
                # ✅ SHORT POSITION LOGIC (para futuros)
                
                # Actualizar precio mínimo histórico
                if position.lowest_price_since_entry is None or current_price < position.lowest_price_since_entry:
                    position.lowest_price_since_entry = current_price
                
                # Calcular ganancia para short
                current_pnl_percent = ((position.entry_price - current_price) / position.entry_price) * 100
                
                # Activar trailing si está en ganancia
                if not position.trailing_stop_active and current_pnl_percent >= position.trailing_activation_threshold:
                    position.trailing_stop_active = True
                    position.trailing_stop_price = current_price * (1 + position.trailing_stop_percent / 100)
                    position.last_trailing_update = datetime.now()
                
                # Actualizar trailing (solo hacia abajo para shorts)
                elif position.trailing_stop_active:
                    new_trailing_price = position.lowest_price_since_entry * (1 + position.trailing_stop_percent / 100)
                    
                    if new_trailing_price < position.trailing_stop_price:
                        position.trailing_stop_price = new_trailing_price
                        position.last_trailing_update = datetime.now()
                        position.trailing_movements += 1
                
                # Verificar cierre por trailing
                if position.trailing_stop_active and current_price >= position.trailing_stop_price:
                    stop_triggered = True
                    trigger_reason = "TRAILING_STOP"
            
            return position, stop_triggered, trigger_reason
            
        except Exception as e:
            print(f"❌ Error en trailing stop para {position.symbol}: {e}")
            return position, False, ""
    
    def get_atr_based_trailing_distance(self, symbol: str, periods: int = 14) -> float:
        """📊 Calcular distancia de trailing basada en ATR (Average True Range)"""
        try:
            # Esta sería una implementación más avanzada usando ATR
            # Por ahora, usar porcentajes adaptativos según el activo
            
            atr_multipliers = {
                'BTC': 1.5,    # Menos volátil, trailing más cercano
                'ETH': 2.0,    # Volatilidad media
                'BNB': 2.5,    # Más volátil, trailing más amplio
                'ADA': 3.0,    # Altcoin más volátil
                'default': 2.0
            }
            
            # Extraer el asset del símbolo
            asset = symbol.replace('USDT', '').replace('BUSD', '')
            multiplier = atr_multipliers.get(asset, atr_multipliers['default'])
            
            # Retornar porcentaje adaptativo
            base_percent = 2.0
            adaptive_percent = base_percent * multiplier
            
            return min(adaptive_percent, 5.0)  # Máximo 5%
            
        except Exception as e:
            print(f"❌ Error calculando ATR para {symbol}: {e}")
            return 2.0  # Default fallback
    
    def generate_trailing_stop_report(self, positions: List[Position]) -> str:
        """📊 Generar reporte detallado de trailing stops"""
        try:
            if not positions:
                return "📈 No hay posiciones con trailing stop activo"
            
            report = "📈 **TRAILING STOPS ACTIVOS**\n"
            
            active_trailing = [pos for pos in positions if hasattr(pos, 'trailing_stop_active') and pos.trailing_stop_active]
            
            if not active_trailing:
                return "📈 No hay trailing stops activos"
            
            for pos in active_trailing:
                current_protection = 0.0
                if pos.trailing_stop_price and pos.entry_price:
                    if pos.side == 'BUY':
                        current_protection = ((pos.trailing_stop_price - pos.entry_price) / pos.entry_price) * 100
                    else:
                        current_protection = ((pos.entry_price - pos.trailing_stop_price) / pos.entry_price) * 100
                
                max_profit = 0.0
                if pos.side == 'BUY' and pos.highest_price_since_entry:
                    max_profit = ((pos.highest_price_since_entry - pos.entry_price) / pos.entry_price) * 100
                elif pos.side == 'SELL' and pos.lowest_price_since_entry:
                    max_profit = ((pos.entry_price - pos.lowest_price_since_entry) / pos.entry_price) * 100
                
                report += f"\n🎯 **{pos.symbol} Pos #{pos.order_id}**\n"
                report += f"├─ Entrada: ${pos.entry_price:.4f}\n"
                report += f"├─ Actual: ${pos.current_price:.4f}\n"
                report += f"├─ Trailing: ${pos.trailing_stop_price:.4f}\n"
                report += f"├─ Protegiendo: +{current_protection:.2f}%\n"
                report += f"├─ Máximo: +{max_profit:.2f}%\n"
                report += f"└─ Movimientos: {pos.trailing_movements}\n"
            
            return report
            
        except Exception as e:
            print(f"❌ Error generando reporte trailing: {e}")
            return "❌ Error en reporte trailing stops"

async def test_portfolio_manager():
    """🧪 Probar Portfolio Manager"""
    print("🧪 TESTING PORTFOLIO MANAGER")
    print("=" * 50)
    
    try:
        # Configuración
        api_key = os.getenv('BINANCE_API_KEY')
        secret_key = os.getenv('BINANCE_SECRET_KEY')
        base_url = os.getenv('BINANCE_BASE_URL', 'https://testnet.binance.vision')
        
        if not api_key or not secret_key:
            print("❌ Faltan credenciales de Binance")
            return
        
        # Crear manager
        portfolio_manager = ProfessionalPortfolioManager(api_key, secret_key, base_url)
        
        # Obtener snapshot
        print("📊 Obteniendo snapshot del portafolio...")
        snapshot = await portfolio_manager.get_portfolio_snapshot()
        
        # Generar reporte TCN
        print("\n" + "="*60)
        tcn_report = portfolio_manager.format_tcn_style_report(snapshot)
        print(tcn_report)
        print("="*60)
        
        # Reporte compacto
        compact_report = portfolio_manager.format_compact_report(snapshot)
        print(f"\n📱 Compacto: {compact_report}")
        
        print(f"\n✅ Test completado - {portfolio_manager.api_calls_count} API calls realizadas")
        
    except Exception as e:
        print(f"❌ Error en test: {e}")

if __name__ == "__main__":
    asyncio.run(test_portfolio_manager()) 