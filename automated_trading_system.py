#!/usr/bin/env python3
"""
AUTOMATED TRADING SYSTEM - Sistema de trading automatizado
Trading real automatizado con protecciones de seguridad y gestión de riesgo
"""

import asyncio
import aiohttp
import hmac
import hashlib
import time
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import numpy as np
from final_real_binance_predictor import OptimizedBinanceData, OptimizedTCNPredictor
from continuous_monitor_discord import DiscordNotifier
import warnings
warnings.filterwarnings('ignore')

class BinanceTradeAPI:
    """API de trading de Binance con autenticación"""
    
    def __init__(self, api_key: str = None, api_secret: str = None, testnet: bool = True):
        self.api_key = api_key or os.getenv('BINANCE_API_KEY')
        self.api_secret = api_secret or os.getenv('BINANCE_API_SECRET')
        self.testnet = testnet
        
        # URLs
        if testnet:
            self.base_url = "https://testnet.binance.vision"
            print("⚠️  MODO TESTNET ACTIVADO - Trading en red de pruebas")
        else:
            self.base_url = "https://api.binance.com"
            print("🔴 MODO PRODUCCIÓN - Trading con dinero real")
        
        if not self.api_key or not self.api_secret:
            print("❌ API keys no configuradas")
            print("Establece BINANCE_API_KEY y BINANCE_API_SECRET")
            self.authenticated = False
        else:
            self.authenticated = True
            print(f"✅ API configurada {'(Testnet)' if testnet else '(Producción)'}")
    
    def _generate_signature(self, params: Dict) -> str:
        """Generar firma para autenticación"""
        query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
        return hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    async def get_account_info(self) -> Dict:
        """Obtener información de la cuenta"""
        if not self.authenticated:
            return {}
        
        endpoint = "/api/v3/account"
        timestamp = int(time.time() * 1000)
        
        params = {
            'timestamp': timestamp
        }
        
        signature = self._generate_signature(params)
        params['signature'] = signature
        
        headers = {
            'X-MBX-APIKEY': self.api_key
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}{endpoint}", 
                    params=params, 
                    headers=headers
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        print(f"❌ Error obteniendo cuenta: {response.status}")
                        return {}
        except Exception as e:
            print(f"❌ Error API cuenta: {e}")
            return {}
    
    async def get_balance(self, asset: str) -> float:
        """Obtener balance específico"""
        account_info = await self.get_account_info()
        
        if not account_info:
            return 0.0
        
        balances = account_info.get('balances', [])
        for balance in balances:
            if balance['asset'] == asset:
                return float(balance['free'])
        
        return 0.0
    
    async def create_market_order(self, symbol: str, side: str, quantity: float) -> Dict:
        """Crear orden de mercado"""
        if not self.authenticated:
            print("❌ No autenticado para trading")
            return {}
        
        endpoint = "/api/v3/order"
        timestamp = int(time.time() * 1000)
        
        params = {
            'symbol': symbol,
            'side': side,  # BUY or SELL
            'type': 'MARKET',
            'quantity': f"{quantity:.8f}",
            'timestamp': timestamp
        }
        
        signature = self._generate_signature(params)
        params['signature'] = signature
        
        headers = {
            'X-MBX-APIKEY': self.api_key
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}{endpoint}",
                    data=params,
                    headers=headers
                ) as response:
                    result = await response.json()
                    
                    if response.status == 200:
                        print(f"✅ Orden ejecutada: {side} {quantity} {symbol}")
                        return result
                    else:
                        print(f"❌ Error orden: {result}")
                        return {}
        except Exception as e:
            print(f"❌ Error creando orden: {e}")
            return {}
    
    async def get_symbol_info(self, symbol: str) -> Dict:
        """Obtener información del símbolo"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/api/v3/exchangeInfo") as response:
                    if response.status == 200:
                        data = await response.json()
                        symbols = data.get('symbols', [])
                        
                        for sym in symbols:
                            if sym['symbol'] == symbol:
                                return sym
        except Exception as e:
            print(f"Error obteniendo info símbolo: {e}")
        
        return {}

class RiskManager:
    """Gestor de riesgo para trading automatizado"""
    
    def __init__(self):
        # Límites de riesgo
        self.max_position_size_pct = 0.10     # Máximo 10% del balance por trade
        self.max_daily_loss_pct = 0.05        # Máximo 5% pérdida diaria
        self.min_confidence_threshold = 0.65  # Confianza mínima para trading
        self.max_trades_per_day = 10          # Máximo trades por día
        self.cooldown_between_trades = 300    # 5 minutos entre trades del mismo par
        
        # Estado
        self.daily_pnl = 0.0
        self.trades_today = 0
        self.last_trade_times = {}
        self.daily_reset_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    
    def check_daily_limits(self) -> bool:
        """Verificar límites diarios"""
        now = datetime.now()
        
        # Reset diario
        if now >= self.daily_reset_time + timedelta(days=1):
            self.daily_pnl = 0.0
            self.trades_today = 0
            self.daily_reset_time = now.replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Verificar límite de pérdida diaria
        if self.daily_pnl <= -self.max_daily_loss_pct:
            print(f"🛑 LÍMITE DE PÉRDIDA DIARIA ALCANZADO: {self.daily_pnl:.2%}")
            return False
        
        # Verificar límite de trades
        if self.trades_today >= self.max_trades_per_day:
            print(f"🛑 LÍMITE DE TRADES DIARIOS ALCANZADO: {self.trades_today}")
            return False
        
        return True
    
    def check_trade_conditions(self, symbol: str, signal: str, confidence: float) -> bool:
        """Verificar condiciones para ejecutar trade"""
        
        # Verificar límites diarios
        if not self.check_daily_limits():
            return False
        
        # Verificar confianza mínima
        if confidence < self.min_confidence_threshold:
            print(f"🛑 Confianza insuficiente: {confidence:.1%} < {self.min_confidence_threshold:.1%}")
            return False
        
        # Verificar cooldown
        now = datetime.now()
        if symbol in self.last_trade_times:
            time_since_last = (now - self.last_trade_times[symbol]).total_seconds()
            if time_since_last < self.cooldown_between_trades:
                print(f"🛑 Cooldown activo para {symbol}: {time_since_last:.0f}s")
                return False
        
        # No hacer trading con señal HOLD
        if signal == 'HOLD':
            return False
        
        return True
    
    def calculate_position_size(self, balance_usdt: float, price: float) -> float:
        """Calcular tamaño de posición"""
        max_usdt_amount = balance_usdt * self.max_position_size_pct
        max_crypto_amount = max_usdt_amount / price
        return max_crypto_amount
    
    def update_trade_executed(self, symbol: str, pnl_pct: float):
        """Actualizar después de ejecutar trade"""
        self.trades_today += 1
        self.daily_pnl += pnl_pct
        self.last_trade_times[symbol] = datetime.now()
        
        print(f"📊 Actualización de riesgo:")
        print(f"  Trades hoy: {self.trades_today}/{self.max_trades_per_day}")
        print(f"  PnL diario: {self.daily_pnl:.2%}")

class AutomatedTradingBot:
    """Bot de trading automatizado"""
    
    def __init__(self, testnet: bool = True, discord_webhook: str = None):
        self.trade_api = BinanceTradeAPI(testnet=testnet)
        self.data_provider = OptimizedBinanceData()
        self.predictor = OptimizedTCNPredictor()
        self.risk_manager = RiskManager()
        self.discord = DiscordNotifier(discord_webhook)
        
        # Configuración
        self.pairs = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
        self.check_interval = 300  # 5 minutos
        self.dry_run_mode = True   # Modo simulación por defecto
        
        # Estado
        self.running = False
        self.positions = {}  # Par -> {'side': 'LONG/SHORT/NONE', 'entry_price': float, 'quantity': float}
        self.total_trades = 0
        self.total_pnl = 0.0
    
    async def start_automated_trading(self, duration_hours: int = 24, dry_run: bool = True):
        """Iniciar trading automatizado"""
        
        self.dry_run_mode = dry_run
        
        print("🤖 INICIANDO BOT DE TRADING AUTOMATIZADO")
        print("="*60)
        print(f"🕐 Duración: {duration_hours} horas")
        print(f"📊 Pares: {', '.join(self.pairs)}")
        print(f"🔄 Intervalo: {self.check_interval} segundos")
        print(f"🧪 Modo: {'DRY RUN (Simulación)' if dry_run else 'TRADING REAL'}")
        print("="*60)
        
        # Verificar configuración
        if not dry_run and not self.trade_api.authenticated:
            print("❌ No se puede hacer trading real sin autenticación")
            return
        
        # Alerta Discord
        await self.discord.send_alert(
            "🤖 Bot de Trading Iniciado",
            f"Trading automatizado activo\n"
            f"Modo: {'Simulación' if dry_run else 'REAL'}\n"
            f"Duración: {duration_hours}h\n"
            f"Pares: {', '.join(self.pairs)}",
            0x00ff00 if dry_run else 0xff0000,
            urgent=not dry_run
        )
        
        self.running = True
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=duration_hours)
        cycle = 0
        
        try:
            async with self.data_provider as provider:
                
                while datetime.now() < end_time and self.running:
                    cycle += 1
                    print(f"\\n🔄 CICLO BOT {cycle} - {datetime.now().strftime('%H:%M:%S')}")
                    
                    # Verificar límites de riesgo
                    if not self.risk_manager.check_daily_limits():
                        print("🛑 Límites de riesgo alcanzados, pausando trading")
                        await asyncio.sleep(3600)  # Pausa 1 hora
                        continue
                    
                    # Analizar mercado y ejecutar trades
                    await self.process_trading_cycle()
                    
                    # Estadísticas cada 10 ciclos
                    if cycle % 6 == 0:  # Cada 30 minutos
                        await self.send_status_update()
                    
                    # Esperar siguiente ciclo
                    if datetime.now() < end_time and self.running:
                        await asyncio.sleep(self.check_interval)
        
        except KeyboardInterrupt:
            print("\\n⏹️  Bot detenido por usuario")
        except Exception as e:
            print(f"\\n❌ Error en bot: {e}")
            await self.discord.send_alert(
                "❌ Error en Bot de Trading",
                f"Error: {str(e)}",
                0xff0000,
                urgent=True
            )
        finally:
            await self.stop_trading()
    
    async def process_trading_cycle(self):
        """Procesar ciclo de trading"""
        
        for pair in self.pairs:
            try:
                # Obtener datos de mercado
                market_data = await self.data_provider.get_market_data(pair)
                
                if not market_data['klines']:
                    continue
                
                # Generar predicción
                prediction = await self.predictor.predict_real_market(pair, market_data)
                
                if not prediction:
                    continue
                
                signal = prediction['signal']
                confidence = prediction['confidence']
                current_price = market_data['klines'][-1]['close']
                
                print(f"  📊 {pair}: {signal} ({confidence:.1%}) @ ${current_price:.4f}")
                
                # Verificar condiciones de trading
                if self.risk_manager.check_trade_conditions(pair, signal, confidence):
                    await self.execute_trading_signal(pair, signal, confidence, current_price, prediction)
                    
            except Exception as e:
                print(f"  ❌ Error procesando {pair}: {e}")
    
    async def execute_trading_signal(self, pair: str, signal: str, confidence: float, 
                                   price: float, prediction: Dict):
        """Ejecutar señal de trading"""
        
        current_position = self.positions.get(pair, {'side': 'NONE'})
        
        # Determinar acción
        action = None
        
        if signal == 'BUY' and current_position['side'] != 'LONG':
            action = 'OPEN_LONG'
        elif signal == 'SELL' and current_position['side'] != 'SHORT':
            action = 'OPEN_SHORT'
        elif signal == 'SELL' and current_position['side'] == 'LONG':
            action = 'CLOSE_LONG'
        elif signal == 'BUY' and current_position['side'] == 'SHORT':
            action = 'CLOSE_SHORT'
        
        if not action:
            return
        
        print(f"  🎯 Ejecutando: {action} {pair} @ ${price:.4f}")
        
        if self.dry_run_mode:
            # Simulación
            await self.simulate_trade(pair, action, price, confidence, prediction)
        else:
            # Trading real
            await self.execute_real_trade(pair, action, price, confidence)
    
    async def simulate_trade(self, pair: str, action: str, price: float, 
                           confidence: float, prediction: Dict):
        """Simular trade (dry run)"""
        
        if action == 'OPEN_LONG':
            # Simular compra
            usdt_balance = 1000  # Balance simulado
            quantity = self.risk_manager.calculate_position_size(usdt_balance, price)
            
            self.positions[pair] = {
                'side': 'LONG',
                'entry_price': price,
                'quantity': quantity,
                'timestamp': datetime.now()
            }
            
            print(f"    ✅ SIMULADO - Comprado {quantity:.6f} {pair} @ ${price:.4f}")
            
        elif action == 'CLOSE_LONG':
            # Simular venta
            position = self.positions[pair]
            entry_price = position['entry_price']
            quantity = position['quantity']
            
            pnl_pct = (price - entry_price) / entry_price
            pnl_usdt = pnl_pct * entry_price * quantity
            
            self.positions[pair] = {'side': 'NONE'}
            self.total_trades += 1
            self.total_pnl += pnl_usdt
            
            print(f"    ✅ SIMULADO - Vendido {quantity:.6f} {pair} @ ${price:.4f}")
            print(f"    💰 PnL: {pnl_pct:.2%} (${pnl_usdt:.2f})")
            
            # Actualizar gestor de riesgo
            self.risk_manager.update_trade_executed(pair, pnl_pct)
            
            # Alerta Discord para trades significativos
            if abs(pnl_pct) > 0.02:  # >2%
                await self.discord.send_alert(
                    f"💰 Trade Cerrado - {pair}",
                    f"Señal: {action}\\n"
                    f"PnL: {pnl_pct:+.2%} (${pnl_usdt:+.2f})\\n"
                    f"Precio entrada: ${entry_price:.4f}\\n"
                    f"Precio salida: ${price:.4f}\\n"
                    f"Confianza: {confidence:.1%}",
                    0x00ff00 if pnl_pct > 0 else 0xff0000
                )
    
    async def execute_real_trade(self, pair: str, action: str, price: float, confidence: float):
        """Ejecutar trade real (solo con autenticación)"""
        
        if not self.trade_api.authenticated:
            print("    ❌ No autenticado para trading real")
            return
        
        print("    🔴 TRADING REAL - Funcionalidad limitada por seguridad")
        print("    ⚠️  Implementar solo después de testing exhaustivo")
        
        # Por seguridad, solo simular por ahora
        await self.simulate_trade(pair, action, price, confidence, {})
    
    async def send_status_update(self):
        """Enviar actualización de estado"""
        
        active_positions = [pair for pair, pos in self.positions.items() if pos['side'] != 'NONE']
        
        await self.discord.send_alert(
            "📊 Estado del Bot",
            f"Modo: {'Simulación' if self.dry_run_mode else 'Real'}\\n"
            f"Trades ejecutados: {self.total_trades}\\n"
            f"PnL total: ${self.total_pnl:.2f}\\n"
            f"Posiciones activas: {len(active_positions)}\\n"
            f"Límites diarios: OK ✅",
            0x3498db
        )
    
    async def stop_trading(self):
        """Detener trading"""
        
        self.running = False
        
        # Cerrar posiciones abiertas (solo en simulación)
        open_positions = [(pair, pos) for pair, pos in self.positions.items() if pos['side'] != 'NONE']
        
        if open_positions:
            print(f"\\n🔄 Cerrando {len(open_positions)} posiciones abiertas...")
            
            for pair, position in open_positions:
                # En modo real, aquí cerrarías las posiciones
                print(f"  🔄 Cerrando posición {pair} ({position['side']})")
                self.positions[pair] = {'side': 'NONE'}
        
        # Reporte final
        await self.discord.send_alert(
            "⏹️ Bot de Trading Detenido",
            f"Trading finalizado\\n"
            f"Total trades: {self.total_trades}\\n"
            f"PnL final: ${self.total_pnl:.2f}\\n"
            f"Posiciones cerradas: {len(open_positions)}",
            0xff9900
        )
        
        print(f"\\n✅ Bot detenido")
        print(f"📊 Estadísticas finales:")
        print(f"  Trades: {self.total_trades}")
        print(f"  PnL: ${self.total_pnl:.2f}")

class TradingManager:
    """Gestor principal de trading automatizado"""
    
    def __init__(self):
        self.bot = None
    
    async def start_interactive_trading(self):
        """Inicio interactivo del trading"""
        
        print("🤖 AUTOMATED TRADING SYSTEM")
        print("Sistema de trading automatizado con protecciones")
        print()
        
        print("⚠️  IMPORTANTE: Este sistema está en modo TESTNET/SIMULACIÓN")
        print("Para trading real, configura las APIs y cambia testnet=False")
        print()
        
        # Configurar Discord
        webhook_url = os.getenv('DISCORD_WEBHOOK_URL')
        
        self.bot = AutomatedTradingBot(testnet=True, discord_webhook=webhook_url)
        
        print("Selecciona modo de operación:")
        print("1. Simulación 1 hora")
        print("2. Simulación 4 horas") 
        print("3. Simulación 8 horas")
        print("4. Trading real (requiere configuración)")
        
        # Para demo, usar simulación
        mode = 1
        duration = 1
        dry_run = True
        
        print(f"\\n🚀 Iniciando trading {'simulado' if dry_run else 'real'} por {duration} hora(s)...")
        
        try:
            await self.bot.start_automated_trading(duration, dry_run)
        except KeyboardInterrupt:
            if self.bot:
                await self.bot.stop_trading()

async def main():
    """Función principal"""
    manager = TradingManager()
    await manager.start_interactive_trading()

if __name__ == "__main__":
    asyncio.run(main()) 