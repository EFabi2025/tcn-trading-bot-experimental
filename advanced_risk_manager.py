#!/usr/bin/env python3
"""
🛡️ ADVANCED RISK MANAGER
Sistema de gestión de riesgo profesional para trading algorítmico
"""

import asyncio
import time
import hmac
import hashlib
import math
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_DOWN
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import aiohttp
import json
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Position:
    """📊 Representación de una posición activa"""
    symbol: str
    side: str  # 'BUY' or 'SELL'
    quantity: float
    entry_price: float
    current_price: float
    entry_time: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing_stop: Optional[float] = None
    pnl_percent: float = 0.0
    pnl_usd: float = 0.0

@dataclass
class RiskLimits:
    """⚠️ Límites de riesgo configurables - VALORES OBTENIDOS DE .ENV"""
    max_position_size_percent: float
    max_total_exposure_percent: float
    max_daily_loss_percent: float
    max_drawdown_percent: float
    min_confidence_threshold: float
    stop_loss_percent: float
    take_profit_percent: float
    trailing_stop_percent: float
    max_concurrent_positions: int
    correlation_limit: float
    min_position_value_usdt: float  # Se obtiene dinámicamente de Binance API

class AdvancedRiskManager:
    """🛡️ Gestor avanzado de riesgo para trading"""

    def __init__(self, binance_config):
        self.config = binance_config
        self.limits = self._load_risk_limits_from_env()

        # Balance se obtiene dinámicamente de Binance API
        self.current_balance = 0.0  # Se actualizará desde get_account_balance()
        self.start_balance = 0.0    # Se establecerá en initialize()
        self.peak_balance = 0.0     # Se actualizará dinámicamente

        # Estado del sistema
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        self.active_positions = {}

        # Circuit breaker
        self.circuit_breaker_active = False
        self.circuit_breaker_until = None

        # Estadísticas
        self.stats = {
            'trades': 0,
            'pnl': 0.0,
            'wins': 0,
            'losses': 0,
            'largest_win': 0.0,
            'largest_loss': 0.0
        }

    def _load_risk_limits_from_env(self) -> RiskLimits:
        """🔧 Cargar límites de riesgo desde variables de entorno"""
        import os

        return RiskLimits(
            max_position_size_percent=float(os.getenv('MAX_POSITION_SIZE_PERCENT', '15.0')),
            max_total_exposure_percent=float(os.getenv('MAX_TOTAL_EXPOSURE_PERCENT', '40.0')),
            max_daily_loss_percent=float(os.getenv('MAX_DAILY_LOSS_PERCENT', '10.0')),
            max_drawdown_percent=float(os.getenv('MAX_DRAWDOWN_PERCENT', '15.0')),
            min_confidence_threshold=float(os.getenv('MIN_CONFIDENCE_THRESHOLD', '0.70')),
            stop_loss_percent=float(os.getenv('STOP_LOSS_PERCENT', '3.0')),
            take_profit_percent=float(os.getenv('TAKE_PROFIT_PERCENT', '6.0')),
            trailing_stop_percent=float(os.getenv('TRAILING_STOP_PERCENT', '2.0')),
            max_concurrent_positions=int(os.getenv('MAX_CONCURRENT_POSITIONS', '2')),
            correlation_limit=float(os.getenv('CORRELATION_LIMIT', '0.7')),
            min_position_value_usdt=11.0  # Se actualizará dinámicamente desde Binance
        )

    async def initialize(self):
        """🚀 Inicializar el risk manager"""
        print("🛡️ Inicializando Advanced Risk Manager...")

        # 💰 OBTENER BALANCE REAL DE BINANCE API
        print("💰 Obteniendo balance real de Binance...")
        real_balance = await self.get_account_balance()

        if real_balance > 0:
            self.current_balance = real_balance
            self.start_balance = real_balance
            self.peak_balance = real_balance
            print(f"✅ Balance obtenido de Binance: ${real_balance:.2f}")
        else:
            print("❌ Error obteniendo balance de Binance, usando valores por defecto")
            self.current_balance = 100.0  # Fallback mínimo
            self.start_balance = 100.0
            self.peak_balance = 100.0

        # Verificar balance mínimo para trading
        if self.current_balance < 50.0:
            print("⚠️ ADVERTENCIA: Balance muy bajo para trading seguro")

        print(f"💰 Balance para trading: ${self.current_balance:.2f}")
        print(f"💡 Con {self.current_balance:.0f} USDT puedes hacer:")

        # Calcular posiciones posibles
        max_position_value = self.current_balance * (self.limits.max_position_size_percent / 100)
        min_binance = self.limits.min_position_value_usdt

        if max_position_value >= min_binance:
            positions_possible = int(self.current_balance * (self.limits.max_total_exposure_percent / 100) / min_binance)
            print(f"   📊 Valor máximo por posición: ${max_position_value:.2f}")
            print(f"   🔢 Posiciones posibles: {min(positions_possible, self.limits.max_concurrent_positions)}")
        else:
            print(f"   ⚠️ PROBLEMA: Posición máxima (${max_position_value:.2f}) < Mínimo Binance (${min_binance:.2f})")
            print(f"   💡 Ajustando límites automáticamente...")
            # Ajustar automáticamente para cumplir mínimo de Binance
            self.limits.max_position_size_percent = (min_binance / self.current_balance) * 100 + 1.0
            print(f"   🔧 Nuevo % máximo por posición: {self.limits.max_position_size_percent:.1f}%")

        print(f"⚠️ Límites de riesgo configurados:")
        print(f"   📊 Max posición: {self.limits.max_position_size_percent:.1f}% (${self.current_balance * self.limits.max_position_size_percent/100:.2f})")
        print(f"   🚨 Max pérdida diaria: {self.limits.max_daily_loss_percent}%")
        print(f"   🛑 Stop Loss: {self.limits.stop_loss_percent}%")
        print(f"   🎯 Take Profit: {self.limits.take_profit_percent}%")
        print(f"   💵 Mínimo Binance: ${self.limits.min_position_value_usdt} USDT")

    async def get_account_balance(self) -> float:
        """💰 Obtener balance USDT actual"""
        try:
            timestamp = int(time.time() * 1000)
            query_string = f"timestamp={timestamp}"
            signature = hmac.new(
                self.config.secret_key.encode('utf-8'),
                query_string.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()

            url = f"{self.config.base_url}/api/v3/account"
            headers = {'X-MBX-APIKEY': self.config.api_key}
            params = {'timestamp': timestamp, 'signature': signature}

            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        for balance in data['balances']:
                            if balance['asset'] == 'USDT':
                                real_balance = float(balance['free'])
                                self.current_balance = real_balance
                                print(f"💰 Balance real actualizado: ${real_balance:.2f}")
                                return real_balance

        except Exception as e:
            print(f"❌ Error obteniendo balance: {e}")

        # Mantener balance configurado si hay error
        return self.current_balance

    def calculate_position_size(self, symbol: str, confidence: float, price: float) -> float:
        """📊 Calcular tamaño de posición considerando mínimo de Binance"""

        # Kelly Criterion básico: f = (bp - q) / b
        base_size_percent = self.limits.max_position_size_percent

        # Ajustar según confianza (más confianza = más posición)
        confidence_multiplier = min(confidence / 0.5, 2.0)  # Max 2x

        # Ajustar según volatilidad histórica (simplificado)
        volatility_adjustment = 0.8  # Reducir en mercados volátiles

        # Calcular tamaño final
        final_size_percent = base_size_percent * confidence_multiplier * volatility_adjustment

        # Limitar al máximo configurado
        final_size_percent = min(final_size_percent, self.limits.max_position_size_percent)

        # Calcular cantidad en USD
        position_value_usd = self.current_balance * (final_size_percent / 100)

        # ⚠️ VALIDACIÓN CRÍTICA: Verificar mínimo de Binance
        if position_value_usd < self.limits.min_position_value_usdt:
            print(f"⚠️ Posición ${position_value_usd:.2f} < mínimo Binance ${self.limits.min_position_value_usdt}")

            # Si el balance lo permite, usar el mínimo de Binance
            if self.current_balance >= self.limits.min_position_value_usdt * 1.2:  # 20% de margen
                position_value_usd = self.limits.min_position_value_usdt
                final_size_percent = (position_value_usd / self.current_balance) * 100
                print(f"🔧 Ajustado al mínimo Binance: ${position_value_usd:.2f} ({final_size_percent:.1f}%)")
            else:
                print(f"❌ Balance insuficiente para mínimo Binance")
                return 0.0

        quantity = position_value_usd / price

        print(f"📊 Position sizing para {symbol}:")
        print(f"   🎯 Confianza: {confidence:.1%}")
        print(f"   📈 Size %: {final_size_percent:.2f}%")
        print(f"   💵 Valor USD: ${position_value_usd:.2f}")
        print(f"   🔢 Cantidad: {quantity:.6f}")
        print(f"   ✅ Cumple mínimo Binance: {'Sí' if position_value_usd >= self.limits.min_position_value_usdt else 'No'}")

        return quantity

    def set_stop_loss_take_profit(self, position: Position) -> Position:
        """🛑 Configurar Stop Loss y Take Profit automáticos"""

        if position.side == 'BUY':
            # Para posición LONG
            position.stop_loss = position.entry_price * (1 - self.limits.stop_loss_percent / 100)
            position.take_profit = position.entry_price * (1 + self.limits.take_profit_percent / 100)
            position.trailing_stop = position.entry_price * (1 - self.limits.trailing_stop_percent / 100)
        else:
            # Para posición SHORT
            position.stop_loss = position.entry_price * (1 + self.limits.stop_loss_percent / 100)
            position.take_profit = position.entry_price * (1 - self.limits.take_profit_percent / 100)
            position.trailing_stop = position.entry_price * (1 + self.limits.trailing_stop_percent / 100)

        print(f"🛑 Stop Loss/Take Profit configurado para {position.symbol}:")
        print(f"   📍 Entrada: ${position.entry_price:.4f}")
        print(f"   🛑 Stop Loss: ${position.stop_loss:.4f}")
        print(f"   🎯 Take Profit: ${position.take_profit:.4f}")
        print(f"   📈 Trailing: ${position.trailing_stop:.4f}")

        return position

    def update_trailing_stop(self, position: Position, current_price: float) -> Position:
        """📈 Actualizar Trailing Stop según precio actual"""

        if position.side == 'BUY' and current_price > position.entry_price:
            # Solo actualizar si estamos en ganancia
            new_trailing = current_price * (1 - self.limits.trailing_stop_percent / 100)
            if new_trailing > position.trailing_stop:
                old_trailing = position.trailing_stop
                position.trailing_stop = new_trailing
                print(f"📈 Trailing Stop actualizado {position.symbol}: ${old_trailing:.4f} → ${new_trailing:.4f}")

        elif position.side == 'SELL' and current_price < position.entry_price:
            # Para SHORT, trailing stop baja
            new_trailing = current_price * (1 + self.limits.trailing_stop_percent / 100)
            if new_trailing < position.trailing_stop:
                old_trailing = position.trailing_stop
                position.trailing_stop = new_trailing
                print(f"📈 Trailing Stop actualizado {position.symbol}: ${old_trailing:.4f} → ${new_trailing:.4f}")

        return position

    async def check_risk_limits_before_trade(self, symbol: str, signal: str, confidence: float) -> Tuple[bool, str]:
        """🛡️ Verificar límites de riesgo antes de abrir trade"""

        # ⚠️ VALIDACIÓN CRÍTICA: Binance Spot no permite ventas en corto
        if signal == 'SELL':
            return False, "🚫 SELL prohibido en Binance Spot - no tienes el activo"

        # Verificar si circuit breaker está activo
        if self.circuit_breaker_active:
            remaining_time = self.circuit_breaker_until - datetime.now()
            return False, f"🔥 Circuit breaker activo por {remaining_time.seconds//60} minutos"

        # Verificar confianza mínima
        if confidence < self.limits.min_confidence_threshold:
            return False, f"📉 Confianza muy baja: {confidence:.1%} < {self.limits.min_confidence_threshold:.1%}"

        # Verificar pérdida diaria máxima
        daily_loss_percent = (abs(self.daily_pnl) / self.start_balance) * 100 if self.daily_pnl < 0 else 0
        if daily_loss_percent >= self.limits.max_daily_loss_percent:
            await self.activate_circuit_breaker("Pérdida diaria máxima alcanzada", 60)
            return False, f"🚨 Pérdida diaria máxima alcanzada: {daily_loss_percent:.1f}%"

        # Verificar máximo número de posiciones
        if len(self.active_positions) >= self.limits.max_concurrent_positions:
            return False, f"📊 Máximo de posiciones alcanzado: {len(self.active_positions)}/{self.limits.max_concurrent_positions}"

        # Verificar exposición total
        current_exposure = sum(pos.quantity * pos.current_price for pos in self.active_positions.values())
        exposure_percent = (current_exposure / self.current_balance) * 100

        if exposure_percent >= self.limits.max_total_exposure_percent:
            return False, f"💼 Exposición máxima alcanzada: {exposure_percent:.1f}%"

        # Verificar correlación con posiciones existentes
        if not await self.check_correlation_risk(symbol):
            return False, f"🔗 Riesgo de correlación alto con {symbol}"

        # ✅ VALIDACIÓN SPOT: Solo permitir BUY si hay suficiente USDT
        if signal == 'BUY':
            # Calcular valor mínimo requerido
            min_value_needed = self.limits.min_position_value_usdt

            if self.current_balance < min_value_needed:
                return False, f"💵 Balance insuficiente: ${self.current_balance:.2f} < ${min_value_needed:.2f}"

        return True, "✅ Todos los límites de riesgo aprobados"

    async def check_correlation_risk(self, symbol: str) -> bool:
        """🔗 Verificar riesgo de correlación entre pares"""

        # ✅ LÓGICA INTELIGENTE: Solo verificar si ya tenemos posiciones activas
        if len(self.active_positions) == 0:
            return True  # Sin posiciones activas = sin riesgo de correlación

        # ✅ CORRELACIONES REALES (no incluir el mismo par)
        correlations = {
            'BTCUSDT': ['ETHUSDT'],  # BTC correlacionado con ETH
            'ETHUSDT': ['BTCUSDT'],  # ETH correlacionado con BTC
            'BNBUSDT': []  # BNB independiente por ahora
        }

        # ✅ LÓGICA MEJORADA: Solo rechazar si hay MÚLTIPLES posiciones correlacionadas
        if symbol in correlations:
            correlated_count = 0
            for correlated_symbol in correlations[symbol]:
                if correlated_symbol in self.active_positions:
                    correlated_count += 1

            # Solo rechazar si ya tenemos 2+ posiciones correlacionadas
            if correlated_count >= 2:
                return False  # Demasiado riesgo de correlación

        return True  # ✅ Permitir el trade

    async def activate_circuit_breaker(self, reason: str, duration_minutes: int):
        """🚨 Activar circuit breaker del sistema"""
        self.circuit_breaker_active = True
        self.circuit_breaker_until = datetime.now() + timedelta(minutes=duration_minutes)

        print(f"🚨 CIRCUIT BREAKER ACTIVADO")
        print(f"   📝 Razón: {reason}")
        print(f"   ⏰ Duración: {duration_minutes} minutos")
        print(f"   🔚 Hasta: {self.circuit_breaker_until.strftime('%H:%M:%S')}")

        # Cerrar todas las posiciones activas (emergencia)
        await self.emergency_close_all_positions()

    async def emergency_close_all_positions(self):
        """🚨 Cerrar todas las posiciones en emergencia"""
        print("🚨 CERRANDO TODAS LAS POSICIONES - EMERGENCIA")

        for symbol, position in list(self.active_positions.items()):
            try:
                # Simular cierre de posición
                await self.close_position(symbol, "EMERGENCY_STOP")
                print(f"✅ Posición {symbol} cerrada por emergencia")
            except Exception as e:
                print(f"❌ Error cerrando {symbol}: {e}")

    async def monitor_positions(self):
        """👁️ Monitorear posiciones activas para Stop Loss/Take Profit"""

        for symbol, position in list(self.active_positions.items()):
            try:
                # Obtener precio actual
                current_price = await self.get_current_price(symbol)
                position.current_price = current_price

                # Calcular PnL actual
                if position.side == 'BUY':
                    position.pnl_percent = ((current_price - position.entry_price) / position.entry_price) * 100
                else:
                    position.pnl_percent = ((position.entry_price - current_price) / position.entry_price) * 100

                position.pnl_usd = (position.pnl_percent / 100) * (position.quantity * position.entry_price)

                # Actualizar trailing stop
                position = self.update_trailing_stop(position, current_price)

                # Verificar Stop Loss
                should_close = False
                close_reason = ""

                if position.side == 'BUY':
                    if current_price <= position.stop_loss:
                        should_close = True
                        close_reason = "STOP_LOSS"
                    elif current_price <= position.trailing_stop:
                        should_close = True
                        close_reason = "TRAILING_STOP"
                    elif current_price >= position.take_profit:
                        should_close = True
                        close_reason = "TAKE_PROFIT"
                else:  # SELL
                    if current_price >= position.stop_loss:
                        should_close = True
                        close_reason = "STOP_LOSS"
                    elif current_price >= position.trailing_stop:
                        should_close = True
                        close_reason = "TRAILING_STOP"
                    elif current_price <= position.take_profit:
                        should_close = True
                        close_reason = "TAKE_PROFIT"

                if should_close:
                    await self.close_position(symbol, close_reason)

            except Exception as e:
                print(f"❌ Error monitoreando {symbol}: {e}")

    async def get_current_price(self, symbol: str) -> float:
        """💲 Obtener precio actual de un símbolo"""
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.config.base_url}/api/v3/ticker/price"
                params = {'symbol': symbol}
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return float(data['price'])
        except:
            pass
        return 0.0

    async def open_position(self, symbol: str, signal: str, confidence: float, price: float) -> Optional[Position]:
        """📈 Abrir nueva posición con gestión de riesgo"""

        # Verificar límites de riesgo
        can_trade, reason = await self.check_risk_limits_before_trade(symbol, signal, confidence)
        if not can_trade:
            print(f"❌ Trade rechazado: {reason}")
            return None

        # Calcular tamaño de posición
        quantity = self.calculate_position_size(symbol, confidence, price)

        # ⚠️ VALIDACIÓN CRÍTICA: Si quantity = 0, cancelar operación
        if quantity <= 0.0:
            print(f"❌ Operación cancelada: Balance insuficiente para {symbol}")
            print(f"   📊 Cantidad calculada: {quantity}")
            print(f"   💰 Balance disponible: ${self.current_balance:.2f}")
            print(f"   💎 Mínimo requerido: ${self.limits.min_position_value_usdt:.2f}")
            return None

        # 🚀 EJECUTAR ORDEN REAL EN BINANCE
        try:
            order_result = await self._execute_real_binance_order(symbol, signal, quantity, price)
            if not order_result:
                print(f"❌ Error ejecutando orden real en Binance para {symbol}")
                return None

            # Crear posición con datos reales de Binance
            position = Position(
                symbol=symbol,
                side=signal,
                quantity=float(order_result['executedQty']),
                entry_price=float(order_result['fills'][0]['price']),  # Precio real de ejecución
                current_price=float(order_result['fills'][0]['price']),
                entry_time=datetime.now()
            )

            # Configurar Stop Loss y Take Profit
            position = self.set_stop_loss_take_profit(position)

            # Guardar posición activa
            self.active_positions[symbol] = position

            print(f"✅ ORDEN REAL EJECUTADA: {signal} {symbol}")
            print(f"   🆔 Order ID: {order_result['orderId']}")
            print(f"   💵 Cantidad ejecutada: {position.quantity:.6f}")
            print(f"   💲 Precio real: ${position.entry_price:.4f}")
            print(f"   🎯 Confianza: {confidence:.1%}")

            return position

        except Exception as e:
            print(f"❌ ERROR CRÍTICO ejecutando orden real: {e}")
            return None

    async def _execute_real_binance_order(self, symbol: str, side: str, quantity: float, price: float) -> Optional[Dict]:
        """🔥 EJECUTAR ORDEN REAL EN BINANCE"""

        try:
            # 🔧 OBTENER FILTROS DEL SÍMBOLO
            filters = await self._get_symbol_filters(symbol)

            if 'LOT_SIZE' in filters:
                # Ajustar cantidad según filtros LOT_SIZE
                original_qty = quantity
                quantity = self._adjust_quantity_to_lot_size(quantity, filters['LOT_SIZE'])

                print(f"🔧 Cantidad ajustada para {symbol}:")
                print(f"   📊 Original: {original_qty:.8f}")
                print(f"   ✅ Ajustada: {quantity:.8f}")
                print(f"   📏 Step Size: {filters['LOT_SIZE']['stepSize']}")
                print(f"   📉 Min Qty: {filters['LOT_SIZE']['minQty']}")

            # Verificar MIN_NOTIONAL si existe
            if 'MIN_NOTIONAL' in filters:
                notional_value = quantity * price
                min_notional = filters['MIN_NOTIONAL']['minNotional']

                if notional_value < min_notional:
                    print(f"❌ Valor notional {notional_value:.2f} < mínimo {min_notional:.2f}")
                    return None

            # Preparar parámetros de orden
            timestamp = int(time.time() * 1000)

            params = {
                'symbol': symbol,
                'side': side,  # 'BUY' or 'SELL'
                'type': 'MARKET',  # Orden de mercado para ejecución inmediata
                'quantity': f"{quantity:.8f}".rstrip('0').rstrip('.'),  # Formato optimizado
                'timestamp': timestamp
            }

            # Crear signature para autenticación
            query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
            signature = hmac.new(
                self.config.secret_key.encode('utf-8'),
                query_string.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()

            params['signature'] = signature

            # Headers de autenticación
            headers = {
                'X-MBX-APIKEY': self.config.api_key,
                'Content-Type': 'application/x-www-form-urlencoded'
            }

            print(f"📡 Enviando orden: {side} {params['quantity']} {symbol}")

            # Ejecutar orden POST /api/v3/order
            async with aiohttp.ClientSession() as session:
                url = f"{self.config.base_url}/api/v3/order"

                async with session.post(url, data=params, headers=headers) as response:
                    if response.status == 200:
                        result = await response.json()
                        print(f"🎉 ORDEN EJECUTADA EXITOSAMENTE: {result['orderId']}")
                        return result
                    else:
                        error_text = await response.text()
                        print(f"❌ Error Binance API: {response.status} - {error_text}")
                        return None

        except Exception as e:
            print(f"❌ ERROR ejecutando orden Binance: {e}")
            return None

    async def close_position(self, symbol: str, reason: str) -> Optional[Dict]:
        """📉 Cerrar posición activa"""

        if symbol not in self.active_positions:
            return None

        position = self.active_positions[symbol]
        current_price = await self.get_current_price(symbol)

        # 🚀 EJECUTAR ORDEN REAL DE CIERRE EN BINANCE
        try:
            # Para cerrar posición BUY, necesitamos hacer SELL
            close_side = 'SELL' if position.side == 'BUY' else 'BUY'

            close_order = await self._execute_real_binance_order(
                symbol, close_side, position.quantity, current_price
            )

            if not close_order:
                print(f"❌ ERROR: No se pudo cerrar posición real en Binance para {symbol}")
                return None

            # Usar precio real de cierre
            real_close_price = float(close_order['fills'][0]['price'])
            real_quantity = float(close_order['executedQty'])

            print(f"✅ POSICIÓN CERRADA EN BINANCE:")
            print(f"   🆔 Close Order ID: {close_order['orderId']}")
            print(f"   💲 Precio real de cierre: ${real_close_price:.4f}")

        except Exception as e:
            print(f"❌ ERROR cerrando posición real: {e}")
            # Usar precio de mercado como fallback
            real_close_price = current_price
            real_quantity = position.quantity

        # Calcular PnL final con precio real
        if position.side == 'BUY':
            pnl_percent = ((real_close_price - position.entry_price) / position.entry_price) * 100
        else:
            pnl_percent = ((position.entry_price - real_close_price) / position.entry_price) * 100

        pnl_usd = (pnl_percent / 100) * (real_quantity * position.entry_price)

        # Actualizar estadísticas
        self.daily_pnl += pnl_usd
        self.total_pnl += pnl_usd
        self.current_balance += pnl_usd

        if pnl_usd > 0:
            self.daily_stats['wins'] += 1
            self.daily_stats['largest_win'] = max(self.daily_stats['largest_win'], pnl_usd)
        else:
            self.daily_stats['losses'] += 1
            self.daily_stats['largest_loss'] = min(self.daily_stats['largest_loss'], pnl_usd)

        # Actualizar peak balance
        if self.current_balance > self.peak_balance:
            self.peak_balance = self.current_balance

        # Eliminar de posiciones activas
        del self.active_positions[symbol]

        # Log del cierre
        emoji = "🟢" if pnl_usd > 0 else "🔴"
        print(f"{emoji} Posición cerrada: {symbol}")
        print(f"   📝 Razón: {reason}")
        print(f"   💰 PnL: {pnl_percent:+.2f}% (${pnl_usd:+.2f})")
        print(f"   ⏰ Duración: {datetime.now() - position.entry_time}")

        return {
            'symbol': symbol,
            'side': position.side,
            'entry_price': position.entry_price,
            'exit_price': real_close_price,
            'quantity': real_quantity,
            'pnl_percent': pnl_percent,
            'pnl_usd': pnl_usd,
            'duration': datetime.now() - position.entry_time,
            'reason': reason
        }

    def get_risk_report(self) -> Dict:
        """📊 Generar reporte de riesgo actual"""

        total_exposure = sum(pos.quantity * pos.current_price for pos in self.active_positions.values())
        exposure_percent = (total_exposure / self.current_balance) * 100 if self.current_balance > 0 else 0

        daily_return = (self.daily_pnl / self.start_balance) * 100 if self.start_balance > 0 else 0
        total_return = ((self.current_balance - self.start_balance) / self.start_balance) * 100 if self.start_balance > 0 else 0

        current_drawdown = ((self.peak_balance - self.current_balance) / self.peak_balance) * 100 if self.peak_balance > 0 else 0

        return {
            'timestamp': datetime.now().isoformat(),
            'balance': {
                'current': self.current_balance,
                'start': self.start_balance,
                'peak': self.peak_balance
            },
            'pnl': {
                'daily_usd': self.daily_pnl,
                'daily_percent': daily_return,
                'total_usd': self.total_pnl,
                'total_percent': total_return
            },
            'risk_metrics': {
                'exposure_usd': total_exposure,
                'exposure_percent': exposure_percent,
                'current_drawdown': current_drawdown,
                'active_positions': len(self.active_positions),
                'circuit_breaker_active': self.circuit_breaker_active
            },
            'daily_stats': self.daily_stats,
            'positions': [
                {
                    'symbol': pos.symbol,
                    'side': pos.side,
                    'quantity': pos.quantity,
                    'entry_price': pos.entry_price,
                    'current_price': pos.current_price,
                    'pnl_percent': pos.pnl_percent,
                    'stop_loss': pos.stop_loss,
                    'take_profit': pos.take_profit
                }
                for pos in self.active_positions.values()
            ]
        }

    async def update_balance(self, new_balance: float):
        """💰 Actualizar balance actual"""
        self.current_balance = new_balance
        print(f"💰 Balance actualizado: ${self.current_balance:.2f}")

    async def get_current_balance(self) -> float:
        """💰 Obtener balance actual"""
        return self.current_balance

    async def _get_symbol_filters(self, symbol: str) -> Dict:
        """📋 Obtener filtros específicos del símbolo desde Binance"""
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.config.base_url}/api/v3/exchangeInfo"
                params = {'symbol': symbol}

                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        symbol_info = data['symbols'][0]

                        filters = {}
                        for filter_info in symbol_info['filters']:
                            if filter_info['filterType'] == 'LOT_SIZE':
                                filters['LOT_SIZE'] = {
                                    'minQty': float(filter_info['minQty']),
                                    'maxQty': float(filter_info['maxQty']),
                                    'stepSize': float(filter_info['stepSize'])
                                }
                            elif filter_info['filterType'] == 'MIN_NOTIONAL':
                                filters['MIN_NOTIONAL'] = {
                                    'minNotional': float(filter_info['minNotional'])
                                }

                        return filters
        except Exception as e:
            print(f"❌ Error obteniendo filtros para {symbol}: {e}")

        return {}

    def _adjust_quantity_to_lot_size(self, quantity: float, lot_size_filter: Dict) -> float:
        """🔧 Ajustar cantidad según filtros LOT_SIZE"""
        min_qty = lot_size_filter['minQty']
        step_size = lot_size_filter['stepSize']

        # Ajustar a step_size más cercano
        adjusted_qty = math.floor(quantity / step_size) * step_size

        # Asegurar que cumple el mínimo
        if adjusted_qty < min_qty:
            adjusted_qty = min_qty

        # Redondear según precisión del step_size
        if step_size >= 1:
            decimals = 0
        elif step_size >= 0.1:
            decimals = 1
        elif step_size >= 0.01:
            decimals = 2
        elif step_size >= 0.001:
            decimals = 3
        elif step_size >= 0.0001:
            decimals = 4
        elif step_size >= 0.00001:
            decimals = 5
        elif step_size >= 0.000001:
            decimals = 6
        else:
            decimals = 8

        return round(adjusted_qty, decimals)
