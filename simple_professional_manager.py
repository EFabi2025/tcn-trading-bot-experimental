#!/usr/bin/env python3
"""
🚀 SIMPLE PROFESSIONAL TRADING MANAGER
Sistema de trading básico sin ML para testing inicial
Integrado con Professional Portfolio Manager para reportes TCN
"""

import asyncio
import aiohttp
import time
import hmac
import hashlib
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from decimal import Decimal
import pandas as pd
from dotenv import load_dotenv

# Importar nuestros módulos de risk y database
from advanced_risk_manager import AdvancedRiskManager, Position, RiskLimits
from trading_database import TradingDatabase

# Importar el módulo de Smart Discord Notifier
from smart_discord_notifier import SmartDiscordNotifier

# ✅ NUEVO: Importar Professional Portfolio Manager
from professional_portfolio_manager import ProfessionalPortfolioManager

# ✅ NUEVO: Importar Portfolio Diversification Manager
from portfolio_diversification_manager import PortfolioDiversificationManager, PortfolioPosition

load_dotenv()

@dataclass
class BinanceConfig:
    """⚙️ Configuración de Binance"""
    api_key: str
    secret_key: str
    base_url: str
    environment: str

@dataclass
class AccountInfo:
    """💰 Información de cuenta de Binance"""
    usdt_balance: float
    total_balance_usd: float
    positions: Dict[str, Dict]
    balances: Dict[str, float]

class TradingManagerStatus:
    """📊 Estados del Trading Manager"""
    STOPPED = "STOPPED"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    ERROR = "ERROR"
    EMERGENCY_STOP = "EMERGENCY_STOP"

class SimpleProfessionalTradingManager:
    """🚀 Trading Manager Profesional Simplificado"""

    def __init__(self):
        """🚀 Inicializar Trading Manager"""
        print("🚀 Simple Professional Trading Manager inicializado")

        # Configuración básica
        self.config = self._load_config()

        # ✅ CORREGIDO: Solo pares con modelos TCN disponibles
        # Excluir temporalmente ADAUSDT, DOTUSDT, SOLUSDT hasta entrenar modelos
        self.symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]

        # ⚠️ PARES PENDIENTES (sin modelos): ["ADAUSDT", "DOTUSDT", "SOLUSDT"]
        self.excluded_symbols = ["ADAUSDT", "DOTUSDT", "SOLUSDT"]

        print(f"📊 Pares activos: {self.symbols}")
        print(f"⏸️ Pares excluidos (sin modelos): {self.excluded_symbols}")

        self.check_interval = 60  # 1 minuto

        # Estado del sistema
        self.status = TradingManagerStatus.STOPPED
        self.database = None
        self.risk_manager = None
        self.client = None

        # ✅ NUEVO: Professional Portfolio Manager
        self.portfolio_manager = None

        # ✅ NUEVO: Portfolio Diversification Manager
        self.diversification_manager = PortfolioDiversificationManager()

        # Balance y trading - ✅ CORREGIDO: Inicializar en 0, obtener de Binance
        self.current_balance = 0.0  # Se actualizará desde Binance
        self.session_pnl = 0.0
        self.trade_count = 0
        # ✅ CORREGIDO: Clave por order_id para múltiples posiciones por símbolo
        self.active_positions: Dict[str, Position] = {}
        self.account_info = None

        # ✅ NUEVO: Portfolio tracking
        self.last_portfolio_snapshot = None
        self.last_tcn_report_time = None

        # Smart Discord Notifier
        self.discord_notifier = SmartDiscordNotifier()

        # Configurar filtros conservadores para evitar spam
        self.discord_notifier.configure_filters(
            min_trade_value_usd=12.0,          # Solo trades > $12
            min_pnl_percent_notify=2.0,        # Solo PnL > 2%
            max_notifications_per_hour=8,      # Max 8/hora
            max_notifications_per_day=40,      # Max 40/día
            suppress_similar_minutes=10,       # 10 min entre similares
            only_profitable_trades=False,      # Notificar pérdidas también
            emergency_only_mode=False          # Todas las prioridades
        )

        # Control de tiempo
        self.last_check_time = None
        self.last_balance_update = None

        # Configuración de trading
        self.monitoring_interval = 30  # segundos

        # Control de pausa/resume
        self.pause_trading = False
        self.pause_reason = None

        self.start_time = None
        self.last_heartbeat = None
        self.emergency_mode = False

        # Precios en tiempo real
        self.current_prices = {}

        # 🔧 CORREGIDO: Métricas unificadas con todas las claves necesarias
        self.metrics = {
            'uptime_seconds': 0,
            'total_checks': 0,
            'successful_checks': 0,
            'api_calls_count': 0,
            'error_count': 0,
            'last_error': None,
            'balance_updates': 0,
            'last_balance_update': None,
            'portfolio_snapshots': 0,
            'tcn_reports_sent': 0,
            'active_positions': 0,
            'session_pnl': 0.0,
            'total_trades': 0,
            'profitable_trades': 0
        }

    def _load_config(self) -> BinanceConfig:
        """⚙️ Cargar configuración desde variables de entorno"""
        return BinanceConfig(
            api_key=os.getenv('BINANCE_API_KEY'),
            secret_key=os.getenv('BINANCE_SECRET_KEY'),
            base_url=os.getenv('BINANCE_BASE_URL', 'https://testnet.binance.vision'),
            environment=os.getenv('ENVIRONMENT', 'testnet')
        )

    def _generate_signature(self, params: str) -> str:
        """🔐 Generar firma para API de Binance"""
        return hmac.new(
            self.config.secret_key.encode('utf-8'),
            params.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

    async def get_account_info(self) -> AccountInfo:
        """💰 Obtener información completa de la cuenta de Binance"""
        try:
            params = {
                'timestamp': int(time.time() * 1000),
                'recvWindow': 10000
            }
            query_string = '&'.join([f"{key}={value}" for key, value in params.items()])
            signature = self._generate_signature(query_string)

            headers = {
                'X-MBX-APIKEY': self.config.api_key
            }

            url = f"{self.config.base_url}/api/v3/account"
            full_url = f"{url}?{query_string}&signature={signature}"

            async with aiohttp.ClientSession() as session:
                async with session.get(full_url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()

                        # Procesar balances
                        balances = {}
                        usdt_balance = 0.0

                        for balance in data.get('balances', []):
                            asset = balance['asset']
                            free = float(balance['free'])
                            locked = float(balance['locked'])
                            total = free + locked

                            if total > 0:
                                balances[asset] = {
                                    'free': free,
                                    'locked': locked,
                                    'total': total
                                }

                                if asset == 'USDT':
                                    usdt_balance = total

                        # Calcular valor total en USD (aproximado)
                        total_balance_usd = usdt_balance  # Base USDT

                        # Obtener precios para otros activos
                        for asset, balance_info in balances.items():
                            if asset != 'USDT' and balance_info['total'] > 0:
                                try:
                                    # Intentar obtener precio en USDT
                                    price_symbol = f"{asset}USDT"
                                    price = await self.get_current_price(price_symbol)
                                    if price > 0:
                                        total_balance_usd += balance_info['total'] * price
                                except:
                                    pass  # Si no se puede obtener precio, ignorar

                        self.metrics['api_calls_count'] += 1
                        self.metrics['balance_updates'] += 1
                        self.metrics['last_balance_update'] = datetime.now().isoformat()

                        return AccountInfo(
                            usdt_balance=usdt_balance,
                            total_balance_usd=total_balance_usd,
                            positions={},  # Implementar si necesitas posiciones específicas
                            balances=balances
                        )

                    else:
                        error_text = await response.text()
                        raise Exception(f"Error API Binance: {response.status} - {error_text}")

        except Exception as e:
            print(f"❌ Error obteniendo info de cuenta: {e}")
            self.metrics['error_count'] += 1
            self.metrics['last_error'] = str(e)
            return None

    async def update_balance_from_binance(self):
        """🔄 Actualizar balance desde Binance"""
        try:
            account_info = await self.get_account_info()
            if account_info:
                old_balance = self.current_balance
                self.current_balance = account_info.usdt_balance
                self.account_info = account_info
                self.last_balance_update = datetime.now()

                # Solo mostrar cambio si es significativo
                if abs(old_balance - self.current_balance) > 0.01:
                    print(f"💰 Balance actualizado: ${old_balance:.2f} → ${self.current_balance:.2f}")

                return True
        except Exception as e:
            print(f"❌ Error actualizando balance: {e}")
            return False

        return False

    async def initialize(self):
        """🚀 Inicializar todos los componentes del sistema"""
        print("🚀 Iniciando Simple Professional Trading Manager...")
        self.status = TradingManagerStatus.STARTING

        try:
            # 1. Inicializar base de datos
            await self._initialize_database()

            # 2. Obtener balance inicial de Binance - ✅ NUEVO
            print("💰 Obteniendo balance de Binance...")
            await self.update_balance_from_binance()
            if self.current_balance == 0:
                print("⚠️ No se pudo obtener balance de Binance, usando valor por defecto")
                self.current_balance = 100.0  # Fallback mínimo si falla API

            # 3. ✅ NUEVO: Inicializar Professional Portfolio Manager
            print("💼 Inicializando Professional Portfolio Manager...")
            self.portfolio_manager = ProfessionalPortfolioManager(
                api_key=self.config.api_key,
                secret_key=self.config.secret_key,
                base_url=self.config.base_url
            )
            print("✅ Portfolio Manager inicializado")

            # 4. Inicializar Risk Manager
            await self._initialize_risk_manager()

            # ✅ 5. SINCRONIZAR POSICIONES EXISTENTES AL ARRANCAR
            await self._sync_positions_on_startup()

            # 6. Verificar conectividad
            await self._verify_connectivity()

            # 7. Configurar monitoreo
            await self._setup_monitoring()

            self.start_time = time.time()
            self.last_heartbeat = datetime.now()
            self.status = TradingManagerStatus.RUNNING

            # Log inicial
            await self.database.log_event('INFO', 'SYSTEM', 'Simple Trading Manager inicializado correctamente')

            print("✅ Simple Professional Trading Manager iniciado correctamente")

        except Exception as e:
            self.status = TradingManagerStatus.ERROR
            print(f"❌ Error inicializando Trading Manager: {e}")
            if self.database:
                await self.database.log_event('ERROR', 'SYSTEM', f'Error inicializando: {e}')
            raise

    async def _sync_positions_on_startup(self):
        """🔄 Sincronizar estado de posiciones activas al arrancar."""
        print("🔄 Sincronizando posiciones existentes al inicio...")
        try:
            # Obtener el estado real del portafolio desde el exchange
            snapshot = await self.portfolio_manager.get_portfolio_snapshot()
            if not snapshot or not snapshot.active_positions:
                print("   ✅ No se encontraron posiciones activas en el exchange.")
                return

            print(f"   🔍 Encontradas {len(snapshot.active_positions)} posiciones en el exchange. Sincronizando con DB...")

            # Reconstruir el estado interno de self.active_positions
            synced_count = 0
            for portfolio_pos in snapshot.active_positions:
                order_id = portfolio_pos.order_id
                if not order_id:
                    print(f"      ⚠️ Advertencia: Posición para {portfolio_pos.symbol} no tiene ID de orden. Se omite.")
                    continue

                # Intento 1: Buscar por ID de orden (para trades nuevos y ya vinculados)
                db_trade = await self.database.get_trade_by_order_id(order_id)

                # Intento 2 (Fallback): Si no se encuentra, es un trade antiguo. Intentar vincularlo.
                if not db_trade:
                    print(f"      🔧 Intentando vincular trade antiguo para {portfolio_pos.symbol} con ID de orden {order_id}...")
                    unlinked_trade = await self.database.get_last_unlinked_buy_trade(portfolio_pos.symbol)
                    if unlinked_trade:
                        # Vincularlo en la DB para futuras ejecuciones
                        await self.database.update_trade_order_id(unlinked_trade['id'], order_id)
                        # Usar este trade para la sesión actual
                        db_trade = unlinked_trade
                    else:
                        print(f"      ❌ No se encontró un trade de compra sin vincular para {portfolio_pos.symbol}.")


                if db_trade:
                    # Reconstruir el objeto Position
                    reconstructed_pos = Position(
                        symbol=db_trade['symbol'],
                        side='BUY',
                        quantity=float(db_trade['quantity']),
                        entry_price=float(db_trade['entry_price']),
                        entry_time=pd.to_datetime(db_trade['entry_time']),
                        stop_loss=float(db_trade['stop_loss']) if db_trade['stop_loss'] else None,
                        take_profit=float(db_trade['take_profit']) if db_trade['take_profit'] else None,
                        trade_id=db_trade['id'], # ID interno de la DB
                        current_price=portfolio_pos.current_price,
                        pnl_percent=portfolio_pos.unrealized_pnl_percent,
                        pnl_usd=portfolio_pos.unrealized_pnl_usd,
                    )

                    self.active_positions[order_id] = reconstructed_pos
                    synced_count += 1
                    print(f"      ✅ Sincronizada posición para {reconstructed_pos.symbol} con ID de orden {order_id}.")
                else:
                    print(f"      ⚠️ Advertencia: Posición con ID de orden {order_id} existe en el exchange pero no se encontró un trade correspondiente en la DB.")

            print(f"   👍 Sincronización completa. {synced_count} posiciones activas cargadas en el estado del bot.")

        except Exception as e:
            print(f"❌ Error fatal durante la sincronización de posiciones: {e}")
            # Decidimos no continuar si no podemos sincronizar el estado, para evitar operaciones incorrectas.
            raise e

    def _get_positions_for_symbol(self, symbol: str) -> List[Position]:
        """Helper: Obtiene todas las posiciones activas para un símbolo específico."""
        return [pos for pos in self.active_positions.values() if pos.symbol == symbol]

    async def _initialize_database(self):
        """🗄️ Inicializar sistema de base de datos"""
        print("🗄️ Inicializando base de datos...")
        self.database = TradingDatabase()

        # Limpiar datos antiguos si es necesario
        await self.database.cleanup_old_data(days_to_keep=90)

        print("✅ Base de datos lista")

    async def _initialize_risk_manager(self):
        """🛡️ Inicializar Risk Manager"""
        print("🛡️ Inicializando Risk Manager...")
        self.risk_manager = AdvancedRiskManager(self.config)
        await self.risk_manager.initialize()

        print("✅ Risk Manager configurado")

    async def _verify_connectivity(self):
        """🔗 Verificar conectividad con APIs"""
        print("🔗 Verificando conectividad...")

        # Test Binance API simple
        try:
            price = await self.get_current_price('BTCUSDT')
            if price > 0:
                print(f"✅ Conectividad Binance OK - BTC: ${price:.2f}")
            else:
                raise Exception("No se pudo obtener precio de test")
        except Exception as e:
            raise Exception(f"Error conectividad Binance: {e}")

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
        except Exception as e:
            print(f"❌ Error obteniendo precio {symbol}: {e}")
        return 0.0

    async def _setup_monitoring(self):
        """🛠️ Configurar tareas de monitoreo del sistema"""
        print("🛠️ Configurando sistema de monitoreo...")

        try:
            # Monitoreo de latido (heartbeat)
            asyncio.create_task(self._heartbeat_monitor())
            print("   ✅ Heartbeat monitor iniciado")

            # ✅ NUEVO: Monitoreo de posiciones con trailing stops cada 30 segundos
            async def _position_monitor_loop():
                """Loop de monitoreo de posiciones con trailing stops"""
                while self.status == TradingManagerStatus.RUNNING:
                    try:
                        await self._position_monitor()
                        await asyncio.sleep(30)  # Cada 30 segundos
                    except Exception as e:
                        print(f"❌ Error en monitor de posiciones: {e}")
                        await asyncio.sleep(60)

            asyncio.create_task(_position_monitor_loop())
            print("   ✅ Position monitor con trailing stops iniciado")

            # Recolector de métricas
            asyncio.create_task(self._metrics_collector())
            print("   ✅ Metrics collector iniciado")

        except Exception as e:
            print(f"❌ Error configurando monitoreo: {e}")
            raise

    async def run(self):
        """🎯 Ejecutar loop principal de trading"""
        print("🎯 Iniciando loop principal de trading...")

        while self.status == TradingManagerStatus.RUNNING:
            try:
                loop_start_time = datetime.now()

                # Verificar si está pausado
                if self.pause_trading:
                    await self._handle_pause_state()
                    await asyncio.sleep(10)
                    continue

                # ✅ NUEVO: Generar reporte TCN cada 5 minutos
                await self._generate_tcn_report_if_needed()

                # ✅ MEJORADO: Mostrar información profesional en tiempo real
                await self._display_professional_info()

                # 1. Actualizar balance cada 5 minutos
                time_since_balance_update = None
                if self.last_balance_update:
                    time_since_balance_update = (datetime.now() - self.last_balance_update).total_seconds()

                if not self.last_balance_update or time_since_balance_update > 300:  # 5 minutos
                    print("🔄 Actualizando balance desde Binance...")
                    await self.update_balance_from_binance()

                # 2. Obtener precios actuales
                prices = await self._get_current_prices()
                self.current_prices = prices

                # ✅ NUEVO: Actualizar PnL de posiciones existentes
                await self._update_positions_pnl(prices)

                # 3. Generar señales usando modelo TCN REAL
                signals = await self._generate_tcn_signals(prices)

                # 4. Procesar cada señal
                for symbol, signal_data in signals.items():
                    await self._process_signal(symbol, signal_data)

                # 5. Actualizar métricas
                await self._update_metrics()

                # 6. Guardar estado en DB
                await self._save_periodic_metrics()

                # ✅ NUEVO: Mostrar resumen cada ciclo
                loop_duration = (datetime.now() - loop_start_time).total_seconds()
                print(f"⏱️ Ciclo completado en {loop_duration:.1f}s")

                # 7. Esperar siguiente ciclo
                await asyncio.sleep(self.check_interval)

            except Exception as e:
                await self._handle_error(e)
                await asyncio.sleep(30)  # Pausa en caso de error

    async def _generate_tcn_report_if_needed(self):
        """📊 Generar reporte TCN cada 5 minutos"""
        try:
            now = datetime.now()

            # Verificar si es hora de generar reporte (cada 5 minutos)
            should_generate = False

            if self.last_tcn_report_time is None:
                should_generate = True
            else:
                time_since_last = (now - self.last_tcn_report_time).total_seconds()
                if time_since_last >= 300:  # 5 minutos
                    should_generate = True

            if should_generate:
                print("📊 Generando reporte TCN profesional...")

                # Obtener snapshot del portafolio
                snapshot = await self.portfolio_manager.get_portfolio_snapshot()
                self.last_portfolio_snapshot = snapshot
                self.metrics['portfolio_snapshots'] += 1

                # Generar reporte TCN
                tcn_report = self.portfolio_manager.format_tcn_style_report(snapshot)

                # ✅ NUEVO: Agregar reporte de modelos TCN
                tcn_models_report = await self._generate_tcn_models_section()

                # ✅ NUEVO: Agregar reporte de diversificación
                diversification_report = await self._generate_diversification_section(snapshot)

                # Combinar reportes
                full_report = tcn_report + tcn_models_report + diversification_report

                # Mostrar en consola
                print("\n" + "="*80)
                print("🎯 REPORTE TCN PROFESSIONAL")
                print("="*80)
                print(full_report)
                print("="*80)

                # Enviar a Discord si está configurado
                if hasattr(self, 'discord_notifier'):
                    await self._send_tcn_discord_notification(full_report)
                    self.metrics['tcn_reports_sent'] += 1

                self.last_tcn_report_time = now

        except Exception as e:
            print(f"❌ Error generando reporte TCN: {e}")

    async def _generate_tcn_models_section(self) -> str:
        """🤖 Generar sección de estado de modelos TCN"""
        try:
            models_section = f"""

🤖 **ESTADO DE MODELOS TCN**
"""

            # ✅ CORREGIDO: Inicializar predictor TCN si no existe (igual que en _generate_tcn_signals)
            if not hasattr(self, 'tcn_predictor'):
                try:
                    from tcn_definitivo_predictor import TCNDefinitivoPredictor
                    self.tcn_predictor = TCNDefinitivoPredictor()
                    print("🎯 Predictor TCN DEFINITIVO inicializado para reporte Discord")
                except Exception as e:
                    models_section += f"❌ **Error inicializando predictor**: {str(e)[:50]}...\n"
                    return models_section

            # Obtener precios actuales para las predicciones (reutilizar si ya los tenemos)
            current_prices = {}
            for symbol in self.symbols:
                try:
                    price = await self.get_current_price(symbol)
                    if price > 0:
                        current_prices[symbol] = price
                except Exception as e:
                    print(f"⚠️ Error obteniendo precio {symbol} para Discord: {e}")

            if not current_prices:
                models_section += "⚠️ **Sin datos de precios para análisis**\n"
                return models_section

            # Generar predicciones para cada símbolo
            for symbol in self.symbols:
                try:
                    if symbol not in current_prices:
                        models_section += f"❌ **{symbol}**: Sin precio disponible\n"
                        continue

                    # Obtener predicción del modelo
                    prediction = None
                    if hasattr(self.tcn_predictor, 'predict_symbol'):
                        prediction = self.tcn_predictor.predict_symbol(symbol)

                    if prediction:
                        signal = prediction['signal']
                        confidence = prediction['confidence']
                        probabilities = prediction.get('probabilities', {})

                        # Emoji según la señal
                        signal_emoji = {
                            'BUY': '🟢',
                            'SELL': '🔴',
                            'HOLD': '🟡'
                        }.get(signal, '⚪')

                        # Formato de confianza con color
                        conf_status = "🔥" if confidence >= 0.80 else "✅" if confidence >= 0.70 else "⚠️"

                        models_section += f"{signal_emoji} **{symbol}**: {signal} ({conf_status} {confidence:.1%})\n"

                        # Mostrar distribución de probabilidades si están disponibles
                        if probabilities:
                            buy_prob = probabilities.get('BUY', 0)
                            hold_prob = probabilities.get('HOLD', 0)
                            sell_prob = probabilities.get('SELL', 0)
                            models_section += f"   📊 BUY:{buy_prob:.1%} | HOLD:{hold_prob:.1%} | SELL:{sell_prob:.1%}\n"

                        # Precio actual
                        current_price = current_prices[symbol]
                        models_section += f"   💰 Precio: ${current_price:,.4f}\n"

                    else:
                        models_section += f"❌ **{symbol}**: Error en predicción\n"

                except Exception as e:
                    models_section += f"❌ **{symbol}**: Error ({str(e)[:30]}...)\n"
                    continue

            # Agregar timestamp del análisis
            models_section += f"\n⏰ Análisis: {datetime.now().strftime('%H:%M:%S')}\n"

            return models_section

        except Exception as e:
            print(f"⚠️ Error generando sección de modelos TCN: {e}")
            return f"\n🤖 **MODELOS TCN:** Error al generar análisis ({str(e)[:30]}...)\n"

    async def _generate_diversification_section(self, snapshot) -> str:
        """🎯 Generar sección de diversificación para el reporte"""
        try:
            # Convertir posiciones a formato PortfolioPosition
            current_positions = []
            for pos in snapshot.active_positions:
                portfolio_pos = PortfolioPosition(
                    symbol=pos.symbol,
                    quantity=pos.size,  # ✅ CORREGIDO: usar 'size' en lugar de 'quantity'
                    entry_price=pos.entry_price,
                    current_price=pos.current_price,
                    value_usd=pos.market_value,  # ✅ CORREGIDO: usar 'market_value' en lugar de 'value_usd'
                    percentage=(pos.market_value / snapshot.total_balance_usd * 100) if snapshot.total_balance_usd > 0 else 0,
                    category=self.diversification_manager.diversification_config['SYMBOL_CATEGORIES'].get(pos.symbol, 'UNKNOWN'),
                    age_minutes=int((datetime.now() - pos.entry_time).total_seconds() / 60),
                    pnl_percent=pos.unrealized_pnl_percent  # ✅ CORREGIDO: usar 'unrealized_pnl_percent'
                )
                current_positions.append(portfolio_pos)

            # Generar análisis de diversificación
            analysis = await self.diversification_manager.analyze_portfolio_diversification(current_positions)

            # Crear sección del reporte
            diversification_section = f"""

🎯 **ANÁLISIS DE DIVERSIFICACIÓN**
📊 **Score:** {analysis.diversification_score:.1f}/100
"""

            # Concentraciones por símbolo
            if analysis.symbol_concentrations:
                diversification_section += "\n**📈 CONCENTRACIÓN POR SÍMBOLO:**\n"
                for symbol, conc in sorted(analysis.symbol_concentrations.items(), key=lambda x: x[1], reverse=True):
                    status = "🔴" if conc > 40 else "🟡" if conc > 35 else "🟢"
                    diversification_section += f"{status} {symbol}: {conc:.1f}%\n"

            # Concentraciones por categoría
            if analysis.category_concentrations:
                diversification_section += "\n**🏷️ CONCENTRACIÓN POR CATEGORÍA:**\n"
                for category, conc in sorted(analysis.category_concentrations.items(), key=lambda x: x[1], reverse=True):
                    status = "🔴" if conc > 60 else "🟢"
                    diversification_section += f"{status} {category}: {conc:.1f}%\n"

            # Alertas importantes
            if analysis.over_concentrated_symbols or analysis.over_concentrated_categories:
                diversification_section += "\n**⚠️ ALERTAS:**\n"
                for symbol in analysis.over_concentrated_symbols:
                    conc = analysis.symbol_concentrations[symbol]
                    diversification_section += f"🚨 {symbol} sobre-concentrado: {conc:.1f}%\n"

                for category in analysis.over_concentrated_categories:
                    conc = analysis.category_concentrations[category]
                    diversification_section += f"🚨 Categoría {category} sobre-concentrada: {conc:.1f}%\n"

            # Recomendaciones principales (máximo 3)
            if analysis.recommendations and len(analysis.recommendations) > 0:
                diversification_section += "\n**💡 RECOMENDACIONES:**\n"
                for rec in analysis.recommendations[:3]:
                    diversification_section += f"• {rec}\n"

            return diversification_section

        except Exception as e:
            print(f"⚠️ Error generando sección de diversificación: {e}")
            return "\n🎯 **DIVERSIFICACIÓN:** Error al generar análisis\n"

    async def _display_professional_info(self):
        """📺 Mostrar información profesional mejorada"""
        try:
            uptime_minutes = (time.time() - self.start_time) / 60

            # Obtener snapshot actualizado del portafolio
            if self.portfolio_manager:
                try:
                    current_snapshot = await self.portfolio_manager.get_portfolio_snapshot()
                    self.last_portfolio_snapshot = current_snapshot
                except Exception as e:
                    print(f"⚠️ Error obteniendo snapshot: {e}")
                    current_snapshot = self.last_portfolio_snapshot
            else:
                current_snapshot = None

            print("🔥" * 80)
            print(f"🕐 {datetime.now().strftime('%H:%M:%S')} | ⏱️ Uptime: {uptime_minutes:.1f}min | 🎯 Trading Manager Professional")

            if current_snapshot:
                print(f"💼 PORTAFOLIO: ${current_snapshot.total_balance_usd:.2f} USDT")
                print(f"💰 USDT Libre: ${current_snapshot.free_usdt:.2f}")

                pnl_sign = "+" if current_snapshot.total_unrealized_pnl >= 0 else ""
                pnl_emoji = "📈" if current_snapshot.total_unrealized_pnl >= 0 else "📉"
                print(f"{pnl_emoji} PnL No Realizado: ${pnl_sign}{current_snapshot.total_unrealized_pnl:.2f}")

                print(f"🎯 Posiciones Activas: {current_snapshot.position_count}/{current_snapshot.max_positions}")

                # ✅ MEJORADO: Mostrar posiciones con información de múltiples entradas
                if current_snapshot.active_positions:
                    print("📈 POSICIONES:")

                    # Agrupar posiciones por símbolo
                    positions_by_symbol = {}
                    for pos in current_snapshot.active_positions:
                        if pos.symbol not in positions_by_symbol:
                            positions_by_symbol[pos.symbol] = []
                        positions_by_symbol[pos.symbol].append(pos)

                    for symbol, positions in positions_by_symbol.items():
                        if len(positions) == 1:
                            # Una sola posición
                            pos = positions[0]
                            pnl_sign = "+" if pos.unrealized_pnl_usd >= 0 else ""
                            pnl_color = "🟢" if pos.unrealized_pnl_usd >= 0 else "🔴"

                            print(f"   {pnl_color} {symbol}: ${pos.entry_price:,.4f} → ${pos.current_price:,.4f} ({pnl_sign}{pos.unrealized_pnl_percent:.2f}% = ${pnl_sign}{pos.unrealized_pnl_usd:.2f})")
                        else:
                            # Múltiples posiciones - mostrar resumen + total
                            total_pnl = sum(p.unrealized_pnl_usd for p in positions)
                            total_value = sum(p.market_value for p in positions)
                            pnl_sign = "+" if total_pnl >= 0 else ""
                            pnl_color = "🟢" if total_pnl >= 0 else "🔴"

                            print(f"   {pnl_color} {symbol} ({len(positions)} pos): ${total_value:.2f} (${pnl_sign}{total_pnl:.2f})")

                            # Mostrar detalle de cada posición individual
                            for i, pos in enumerate(positions, 1):
                                pos_pnl_sign = "+" if pos.unrealized_pnl_usd >= 0 else ""
                                duration_str = f"{pos.duration_minutes}min" if pos.duration_minutes < 60 else f"{pos.duration_minutes//60}h"
                                print(f"      #{i}: {pos.size:.6f} @ ${pos.entry_price:,.2f} ({pos_pnl_sign}{pos.unrealized_pnl_percent:.1f}%) {duration_str}")

                # Mostrar principales activos
                print("🪙 ACTIVOS PRINCIPALES:")
                main_assets = [asset for asset in current_snapshot.all_assets
                             if asset.usd_value >= 1.0 and asset.symbol != 'USDT'][:5]

                for asset in main_assets:
                    print(f"   🪙 {asset.symbol}: {asset.total:.6f} (${asset.usd_value:.2f})")

                if current_snapshot.free_usdt > 0:
                    print(f"   💵 USDT: ${current_snapshot.free_usdt:.2f}")
            else:
                print(f"💼 PORTAFOLIO: ${self.current_balance:.2f} USDT")
                print(f"💰 USDT Libre: ${self.current_balance:.2f}")
                print(f"📈 PnL No Realizado: $+0.00")
                print(f"🎯 Posiciones Activas: 0/5")
                print("📈 POSICIONES: Ninguna")

            # Mostrar métricas
            print(f"📊 MÉTRICAS: API calls: {self.metrics.get('api_calls_count', 0)} | Errores: {self.metrics.get('error_count', 0)} | Reportes TCN: {self.metrics.get('tcn_reports_sent', 0)}")

            print("🔥" * 80)

        except Exception as e:
            print(f"❌ Error en display: {e}")

    async def _send_tcn_discord_notification(self, tcn_report: str):
        """💬 Enviar reporte TCN a Discord"""
        try:
            if not tcn_report or len(tcn_report.strip()) == 0:
                print("⚠️ Reporte TCN vacío, saltando Discord")
                return

            from smart_discord_notifier import NotificationPriority

            # Enviar reporte completo con prioridad alta
            result = await self.discord_notifier.send_system_notification(
                tcn_report,
                NotificationPriority.HIGH
            )

            if result and hasattr(result, 'status_code'):
                if result.status_code == 204:
                    print("✅ Discord: Reporte TCN enviado (204 OK)")
                elif result.status_code == 200:
                    print("✅ Discord: Reporte TCN enviado (200 OK)")
                else:
                    print(f"⚠️ Discord: Status {result.status_code}")

        except Exception as e:
            print(f"❌ Discord error: {e}")

    async def _update_positions_pnl(self, prices: Dict[str, float]):
        """📈 Actualizar PnL de todas las posiciones activas"""
        for symbol, position in self.active_positions.items():
            if symbol in prices:
                current_price = prices[symbol]
                position.current_price = current_price

                # Calcular PnL actualizado
                if position.side == 'BUY':
                    position.pnl_percent = ((current_price - position.entry_price) / position.entry_price) * 100
                else:
                    position.pnl_percent = ((position.entry_price - current_price) / position.entry_price) * 100

                position.pnl_usd = (position.pnl_percent / 100) * (position.quantity * position.entry_price)

    async def _get_current_prices(self) -> Dict[str, float]:
        """💰 Obtener precios actuales de todos los símbolos"""
        prices = {}

        print("🔄 Obteniendo precios actuales...")

        for symbol in self.symbols:
            try:
                price = await self.get_current_price(symbol)
                if price > 0:
                    prices[symbol] = price
                    self.metrics['successful_checks'] += 1
                    print(f"   ✅ {symbol}: ${price:.4f}")
                else:
                    print(f"   ❌ {symbol}: Sin precio")

                self.metrics['total_checks'] += 1

            except Exception as e:
                print(f"   ❌ Error obteniendo precio {symbol}: {e}")
                self.metrics['error_count'] += 1
                await self.database.log_event('ERROR', 'MARKET_DATA', f'Error precio {symbol}: {e}', symbol)

        self.last_check_time = datetime.now()
        self.metrics['api_calls_count'] += len(self.symbols)

        return prices

    async def _generate_tcn_signals(self, prices: Dict[str, float]) -> Dict:
        """🧠 Generar señales usando modelo TCN REAL - SOLO BUY para Binance Spot"""
        signals = {}

        # ✅ CORREGIDO: Inicializar predictor TCN DEFINITIVO PRIMERO, independientemente del balance
        if not hasattr(self, 'tcn_predictor'):
            try:
                from tcn_definitivo_predictor import TCNDefinitivoPredictor
                self.tcn_predictor = TCNDefinitivoPredictor()
                print("🎯 Predictor TCN DEFINITIVO inicializado correctamente")
                print(f"   📊 Modelos cargados: {len(self.tcn_predictor.models)}")
                print(f"   🎯 Símbolos: {list(self.tcn_predictor.models.keys())}")
            except Exception as e:
                print(f"❌ Error inicializando predictor TCN definitivo: {e}")
                # Fallback al predictor de emergencia
                try:
                    from emergency_tcn_predictor import EmergencyTCNPredictor
                    self.tcn_predictor = EmergencyTCNPredictor()
                    print("🚨 Usando predictor TCN de emergencia como fallback")
                except Exception as e2:
                    print(f"❌ Error con fallback: {e2}")
                    return signals

        # Verificar si tenemos USDT suficiente para operar (DESPUÉS de inicializar predictor)
        if self.current_balance < self.risk_manager.limits.min_position_value_usdt:
            print(f"⚠️ Balance insuficiente para nuevas posiciones: ${self.current_balance:.2f} < ${self.risk_manager.limits.min_position_value_usdt:.2f}")
            print("🎯 Generando predicciones TCN para análisis (sin ejecutar trades)...")
            # Continuar para generar predicciones de análisis

        # Generar señales para cada símbolo usando TCN DEFINITIVO
        try:
            for symbol, current_price in prices.items():
                try:
                    print(f"🔍 Analizando {symbol} con modelo TCN DEFINITIVO...")

                    # Generar predicción TCN DEFINITIVA
                    if hasattr(self.tcn_predictor, 'predict_symbol'):
                        # Usar predictor definitivo - no necesita datos externos
                        prediction = self.tcn_predictor.predict_symbol(symbol)
                    else:
                        # Fallback al método de emergencia con datos de mercado
                        from emergency_tcn_predictor import AdvancedBinanceData
                        async with AdvancedBinanceData() as binance_data:
                            market_data = await binance_data.get_comprehensive_data(symbol)
                            if not market_data or not market_data.get('klines_1m'):
                                print(f"  ❌ Sin datos suficientes para {symbol}")
                                continue
                            prediction = await self.tcn_predictor.predict_enhanced(symbol, market_data)

                    if not prediction:
                        print(f"  ❌ No se pudo generar predicción para {symbol}")
                        continue

                    signal = prediction['signal']
                    confidence = prediction['confidence']

                    print(f"  🎯 TCN Señal: {signal} | Confianza: {confidence:.1%}")
                    if 'probabilities' in prediction:
                        probs = prediction['probabilities']
                        print(f"  📊 Probabilidades: BUY:{probs.get('BUY', 0):.3f} | HOLD:{probs.get('HOLD', 0):.3f} | SELL:{probs.get('SELL', 0):.3f}")

                    # ✅ FILTROS CRÍTICOS - REORDENADOS PARA MÁXIMA CLARIDAD

                    # 1. Verificar confianza mínima ANTES que nada.
                    min_confidence = float(os.getenv('MIN_CONFIDENCE_THRESHOLD', '0.70'))
                    if confidence < min_confidence:
                        print(f"  ❌ Confianza insuficiente: {confidence:.1%} < {min_confidence:.1%}")
                        continue

                    # 2. Filtrar señales de HOLD.
                    if signal == 'HOLD':
                        print(f"  ⏸️ Señal HOLD ignorada - Mantener estado actual en {symbol}")
                        continue

                    # 3. Procesar señales BUY y SELL según las posiciones existentes.
                    existing_positions = self._get_positions_for_symbol(symbol)
                    has_position = len(existing_positions) > 0

                    if signal == 'BUY':
                        if has_position:
                            print(f"  ⏸️ Señal BUY ignorada - Ya existe(n) {len(existing_positions)} posición(es) en {symbol}")
                            continue

                        balance_sufficient = self.current_balance >= self.risk_manager.limits.min_position_value_usdt
                        if not balance_sufficient:
                            print(f"  📊 SEÑAL BUY GENERADA (solo análisis): {symbol} {signal} ({confidence:.1%}) - Balance insuficiente para trade")
                            continue

                    elif signal == 'SELL':
                        if not has_position:
                            print(f"  ⏸️ Señal SELL ignorada - No hay posición que vender en {symbol}")
                            continue
                        else:
                            # La señal es de venta y tenemos posición. ¡Es una señal válida para procesar!
                            print(f"  🔥 SEÑAL SELL VÁLIDA - Se cerrarán {len(existing_positions)} posición(es) en {symbol}")

                    # ✅ SEÑAL VÁLIDA - Si hemos llegado hasta aquí, la señal es buena.
                        signals[symbol] = {
                            'signal': signal,
                            'confidence': confidence,
                            'current_price': current_price,
                            'timestamp': datetime.now(),
                            'reason': 'TCN_MODEL_PREDICTION',
                            'available_usdt': self.current_balance,
                            'probabilities': prediction.get('probabilities', {}),
                        'balance_sufficient': self.current_balance >= self.risk_manager.limits.min_position_value_usdt
                        }
                    print(f"  ✅ SEÑAL AÑADIDA A LA COLA: {symbol} {signal} ({confidence:.1%})")

                except Exception as e:
                    print(f"  ❌ Error procesando {symbol}: {e}")
                    continue

        except Exception as e:
            print(f"❌ Error generando señales TCN: {e}")
            return signals

        if signals:
            print(f"🎯 Total señales TCN generadas: {len(signals)}")
        else:
            print("📊 No se generaron señales TCN válidas en este ciclo")

        return signals

    async def _process_signal(self, symbol: str, signal_data: Dict):
        """⚡ Procesar una señal individual"""

        signal = signal_data['signal']
        confidence = signal_data['confidence']
        current_price = signal_data['current_price']
        balance_sufficient = signal_data.get('balance_sufficient', True)

        # Skip si es HOLD
        if signal == 'HOLD':
            return

        # Verificar si el balance es suficiente para nuevas posiciones BUY
        if signal == 'BUY' and not balance_sufficient:
            print(f"  💰 Señal BUY {symbol} no ejecutada - Balance insuficiente")
            return

        # Verificar si ya tenemos posición en este símbolo
        if self._get_positions_for_symbol(symbol):
            await self._manage_existing_position(symbol, signal_data)
        else:
            await self._consider_new_position(symbol, signal_data)

    async def _consider_new_position(self, symbol: str, signal_data: Dict):
        """📈 Considerar nueva posición con diversificación"""

        signal = signal_data['signal']
        confidence = signal_data['confidence']
        current_price = signal_data['current_price']

        # 🔧 CORRECCIÓN: SELL no debe crear nuevas posiciones
        if signal == 'SELL':
            print(f"  ⚠️ Señal SELL ignorada - No hay posición existente que vender en {symbol}")
            return

        # ✅ NUEVO: Verificar diversificación del portafolio ANTES de risk management
        await self._check_portfolio_diversification_before_trade(symbol, signal_data)

        # Verificar límites de riesgo
        can_trade, reason = await self.risk_manager.check_risk_limits_before_trade(
            symbol, signal, confidence
        )

        if not can_trade:
            print(f"❌ Trade rechazado {symbol}: {reason}")
            await self.database.log_event('WARNING', 'RISK', f'Trade rechazado {symbol}: {reason}', symbol)
            return

        # Abrir nueva posición
        position = await self.risk_manager.open_position(symbol, signal, confidence, current_price)

        if position:
            # ✅ CORREGIDO: Usar trade_id (que ahora es el order_id) como clave
            self.active_positions[position.order_id] = position

            # Guardar en base de datos, incluyendo el order_id de Binance
            trade_data = {
                'symbol': symbol,
                'side': signal,
                'quantity': position.quantity,
                'entry_price': position.entry_price,
                'entry_time': position.entry_time,
                'stop_loss': position.stop_loss,
                'take_profit': position.take_profit,
                'confidence': confidence,
                'strategy': 'TCN_MODEL_SIGNALS',
                'is_active': True,
                'metadata': {
                    'signal_reason': signal_data.get('reason'),
                    'signal_time': signal_data['timestamp'].isoformat()
                },
                'order_id': position.order_id # ✅ CRÍTICO: Guardar el ID de la orden de Binance
            }

            # El ID interno de la DB se genera automáticamente, no necesitamos guardarlo aquí.
            await self.database.save_trade(trade_data)

            self.trade_count += 1

            # Log del trade
            await self.database.log_event(
                'INFO', 'TRADING',
                f'Nueva posición: {signal} {symbol} @ ${current_price:.4f}',
                symbol
            )

            # Enviar notificación Discord si está configurado
            trade_notification_data = {
                'symbol': symbol,
                'side': signal,
                'value_usd': position.quantity * position.entry_price,
                'price': current_price,
                'confidence': confidence,
                'pnl_percent': 0,
                'pnl_usd': 0,
                'reason': 'NEW_POSITION'
            }

            # Usar Smart Discord Notifier para trades
            if hasattr(self, 'discord_notifier'):
                await self.discord_notifier.send_trade_notification(trade_notification_data)
            else:
                await self._send_discord_notification(f"🟢 **NUEVA POSICIÓN**\n"
                                                     f"📊 {symbol}: {signal}\n"
                                                     f"💰 Precio: ${current_price:.4f}\n"
                                                     f"🎯 Confianza: {confidence:.1%}\n"
                                                     f"📈 Cantidad: {position.quantity:.6f}")

    async def _manage_existing_position(self, symbol: str, signal_data: Dict):
        """🔄 Gestionar posición existente"""

        existing_positions = self._get_positions_for_symbol(symbol)
        if not existing_positions:
            return

        current_price = signal_data['current_price']
        signal = signal_data['signal']
        confidence = signal_data['confidence']

        # Actualizar PnL de todas las posiciones para este símbolo
        for position in existing_positions:
            position.current_price = current_price
            if position.side == 'BUY':
                position.pnl_percent = ((current_price - position.entry_price) / position.entry_price) * 100
            else: # Futuros
                position.pnl_percent = ((position.entry_price - current_price) / position.entry_price) * 100
            position.pnl_usd = (position.pnl_percent / 100) * (position.quantity * position.entry_price)

        # Si la señal es de venta, cerrar TODAS las posiciones para este símbolo
        if signal == 'SELL':
            print(f"🔥 Señal de VENTA para {symbol}. Cerrando {len(existing_positions)} posición(es).")
            for position in existing_positions:
                await self._close_position(position.order_id, "SIGNAL_SELL")

        # Lógica de reversión (ej. de BUY a SELL con alta confianza)
        reversal_threshold = float(os.getenv('SIGNAL_REVERSAL_THRESHOLD', '0.85'))
        if confidence > reversal_threshold:
            for position in existing_positions:
                if (position.side == 'BUY' and signal == 'SELL') or (position.side == 'SELL' and signal == 'BUY'):
                    await self._close_position(position.order_id, "SIGNAL_REVERSAL")

    async def _close_position(self, order_id: str, reason: str):
        """📉 Cerrar posición específica por ID de orden"""

        # ✅ **LA SOLUCIÓN**: Comprobación robusta al inicio.
        # Buscar la posición y verificar que esté activa.
        position = self.active_positions.get(order_id)
        if not position or not position.is_active:
            # Si no se encuentra o ya está inactiva, es probable que ya se esté procesando.
            print(f"ℹ️ Intento de cerrar posición {order_id} omitido. No encontrada o ya marcada para cierre.")
            return

        # ✅ **CRÍTICO**: Marcar la posición como inactiva INMEDIATAMENTE.
        # Esto previene cualquier intento de doble cierre en el mismo ciclo.
        position.is_active = False

        print(f"👇 Iniciando cierre para {position.symbol} (ID Orden: {order_id}) por motivo: {reason}")

        # Ahora el resto de la función puede proceder de forma segura.
        symbol = position.symbol
        current_price = await self.get_current_price(symbol)

        # Simular ejecución de la orden de venta (en un sistema real, aquí iría la llamada a la API)
        print(f"   💸 Ejecutando orden de VENTA simulada para {position.quantity} de {symbol} a ${current_price:.4f}")

        # Calcular PnL final
        if position.side == 'BUY':
            pnl_percent = ((current_price - position.entry_price) / position.entry_price) * 100
        else:
            pnl_percent = ((position.entry_price - current_price) / position.entry_price) * 100

        pnl_usd = (pnl_percent / 100) * (position.quantity * position.entry_price)

        # Actualizar estadísticas
        self.session_pnl += pnl_usd

        # Actualizar en base de datos
        if hasattr(position, 'position_id') and position.position_id:
            exit_data = {
                'exit_price': current_price,
                'exit_time': datetime.now(),
                'pnl_percent': pnl_percent,
                'pnl_usd': pnl_usd,
                'exit_reason': reason
            }
            await self.database.update_trade_exit(position.position_id, exit_data)

        # Remover de posiciones activas
        # La clave es el order_id, lo que es correcto.
        if order_id in self.active_positions:
            del self.active_positions[order_id]

        # Log y notificación
        color = "🟢" if pnl_usd > 0 else "🔴"
        await self.database.log_event(
            'INFO', 'TRADING',
            f'Posición cerrada: {symbol} - PnL: {pnl_percent:.2f}% (${pnl_usd:.2f})',
            symbol
        )

        await self._send_discord_notification(f"{color} **POSICIÓN CERRADA**\n"
                                             f"📊 {symbol}: {position.side}\n"
                                             f"📈 PnL: {pnl_percent:.2f}% (${pnl_usd:.2f})\n"
                                             f"🔄 Razón: {reason}")

        print(f"📉 Posición cerrada: {symbol} (ID de orden: {order_id}) - PnL: {pnl_percent:.2f}% (${pnl_usd:.2f})")

    async def _check_portfolio_diversification_before_trade(self, symbol: str, signal_data: Dict):
        """🎯 Verificar diversificación antes de ejecutar trade"""

        try:
            # Obtener posiciones actuales
            snapshot = await self.portfolio_manager.get_portfolio_snapshot()

            # Convertir a formato PortfolioPosition
            current_positions = []
            for pos in snapshot.active_positions:
                portfolio_pos = PortfolioPosition(
                    symbol=pos.symbol,
                    quantity=pos.size,  # ✅ CORREGIDO: usar 'size' en lugar de 'quantity'
                    entry_price=pos.entry_price,
                    current_price=pos.current_price,
                    value_usd=pos.market_value,  # ✅ CORREGIDO: usar 'market_value' en lugar de 'value_usd'
                    percentage=(pos.market_value / snapshot.total_balance_usd * 100) if snapshot.total_balance_usd > 0 else 0,
                    category=self.diversification_manager.diversification_config['SYMBOL_CATEGORIES'].get(pos.symbol, 'UNKNOWN'),
                    age_minutes=int((datetime.now() - pos.entry_time).total_seconds() / 60),
                    pnl_percent=pos.unrealized_pnl_percent  # ✅ CORREGIDO: usar 'unrealized_pnl_percent'
                )
                current_positions.append(portfolio_pos)

            # Calcular tamaño de posición propuesto
            confidence = signal_data['confidence']
            current_price = signal_data['current_price']

            # Usar el mismo cálculo que el risk manager
            position_size_percent = min(15.0, confidence * 20)  # Máximo 15%
            position_size_usd = (self.current_balance * position_size_percent / 100)

            # Verificar si se permite la nueva posición
            allowed, reason = await self.diversification_manager.should_allow_new_position(
                symbol, position_size_usd, current_positions
            )

            if not allowed:
                print(f"🚫 DIVERSIFICACIÓN: {reason}")
                await self.database.log_event('WARNING', 'DIVERSIFICATION', f'Trade bloqueado: {reason}', symbol)

                # Generar reporte de diversificación
                diversification_report = await self.diversification_manager.generate_diversification_report(current_positions)
                print(diversification_report)

                # Enviar notificación Discord sobre bloqueo por diversificación
                await self._send_discord_notification(
                    f"🚫 **TRADE BLOQUEADO POR DIVERSIFICACIÓN**\n"
                    f"📊 {symbol}: {signal_data['signal']}\n"
                    f"⚠️ Razón: {reason}\n"
                    f"🎯 Confianza perdida: {confidence:.1%}"
                )

                # Lanzar excepción para detener el trade
                raise Exception(f"Trade bloqueado por diversificación: {reason}")

            # Ajustar tamaño de posición si es necesario
            adjusted_size = self.diversification_manager.calculate_diversification_adjusted_size(
                symbol, position_size_usd, current_positions
            )

            if adjusted_size < position_size_usd:
                reduction_percent = ((position_size_usd - adjusted_size) / position_size_usd) * 100
                print(f"📏 DIVERSIFICACIÓN: Tamaño reducido {reduction_percent:.1f}% para {symbol}")
                print(f"   💰 Original: ${position_size_usd:.2f} → Ajustado: ${adjusted_size:.2f}")

                # Actualizar signal_data con el tamaño ajustado
                signal_data['adjusted_size_usd'] = adjusted_size
                signal_data['diversification_adjustment'] = True

            # Generar análisis de diversificación cada 10 trades
            if self.trade_count % 10 == 0:
                analysis = await self.diversification_manager.analyze_portfolio_diversification(current_positions)
                print(f"📊 DIVERSIFICACIÓN SCORE: {analysis.diversification_score:.1f}/100")

                if analysis.diversification_score < 60:
                    print("⚠️ ADVERTENCIA: Score de diversificación bajo")
                    for rec in analysis.recommendations[:3]:  # Solo las 3 principales
                        print(f"   💡 {rec}")

        except Exception as e:
            if "Trade bloqueado por diversificación" in str(e):
                raise  # Re-lanzar bloqueos de diversificación
            else:
                print(f"⚠️ Error en verificación de diversificación: {e}")
                # No bloquear el trade por errores técnicos

    async def _heartbeat_monitor(self):
        """💓 Monitor de latido del sistema"""
        while self.status == TradingManagerStatus.RUNNING:
            try:
                # Verificar conectividad cada 5 minutos
                await asyncio.sleep(300)

                # Ping a Binance
                test_price = await self.get_current_price("BTCUSDT")
                if test_price <= 0:
                    raise Exception("No se pudo obtener precio de BTC")

                # Log heartbeat
                await self.database.log_event('INFO', 'SYSTEM', f'Heartbeat OK - BTC: ${test_price:.2f}')

            except Exception as e:
                await self.database.log_event('ERROR', 'SYSTEM', f'Heartbeat failed: {e}')
                await asyncio.sleep(60)

    async def _position_monitor(self):
        """🔍 Monitoreo continuo de posiciones y gestión de riesgo"""
        try:
            # 1. Obtener posiciones actuales
            snapshot = await self.portfolio_manager.get_portfolio_snapshot()

            if not snapshot.active_positions:
                print("   📊 Sin posiciones activas para monitorear")
                return

            print(f"🔍 Monitoreando {len(snapshot.active_positions)} posición(es)...")

            # 2. Actualizar precios para cada posición
            symbols_to_update = list(set([pos.symbol for pos in snapshot.active_positions]))
            current_prices = await self.portfolio_manager.update_all_prices(symbols_to_update)

            positions_to_close = []
            trailing_updates = []

            # 3. ✅ NUEVO: Procesar cada posición individualmente con trailing stop
            for i, position in enumerate(snapshot.active_positions):
                try:
                    current_price = current_prices.get(position.symbol, position.current_price)

                    # Actualizar precio actual en la posición
                    position.current_price = current_price

                    # 🔄 Recalcular PnL con precio actual
                    if position.side == 'BUY':
                        entry_value = position.size * position.entry_price
                        current_value = position.size * current_price
                        position.unrealized_pnl_usd = current_value - entry_value
                        position.unrealized_pnl_percent = (position.unrealized_pnl_usd / entry_value) * 100 if entry_value > 0 else 0.0
                        position.market_value = current_value

                    # ✅ NUEVO: Aplicar trailing stop profesional
                    updated_position, stop_triggered, trigger_reason = self.portfolio_manager.update_trailing_stop_professional(
                        position, current_price
                    )

                    # ✅ CRÍTICO: Actualizar la posición en el snapshot con el estado del trailing stop
                    snapshot.active_positions[i] = updated_position

                    # Si se actualiza el trailing, registrar el cambio
                    if hasattr(updated_position, 'trailing_stop_active') and updated_position.trailing_stop_active:
                        if updated_position.trailing_movements > position.trailing_movements:
                            trailing_updates.append(f"📈 {updated_position.symbol} Pos #{updated_position.order_id}: Trail movido a ${updated_position.trailing_stop_price:.4f}")

                    # Verificar condiciones de cierre
                    should_close, close_reason = await self._check_position_exit_conditions(updated_position, current_price)

                    if stop_triggered or should_close:
                        reason = trigger_reason if stop_triggered else close_reason
                        positions_to_close.append((updated_position, reason))

                        print(f"🛑 Marcando para cierre: {updated_position.symbol} Pos #{updated_position.order_id} - {reason}")

                except Exception as e:
                    print(f"❌ Error monitoreando {position.symbol}: {e}")
                    continue

            # 4. Mostrar actualizaciones de trailing stops
            if trailing_updates:
                print("📈 ACTUALIZACIONES TRAILING STOPS:")
                for update in trailing_updates:
                    print(f"   {update}")

            # 5. Cerrar posiciones marcadas
            if positions_to_close:
                await self._close_positions_batch(positions_to_close)
            else:
                print("   ✅ Todas las posiciones dentro de parámetros")

                # Mostrar resumen de trailing stops activos
                active_trailing = [pos for pos in snapshot.active_positions
                                 if hasattr(pos, 'trailing_stop_active') and pos.trailing_stop_active]

                if active_trailing:
                    print(f"   📈 Trailing stops activos: {len(active_trailing)}")
                    for pos in active_trailing:
                        protection = ((pos.trailing_stop_price - pos.entry_price) / pos.entry_price * 100) if pos.trailing_stop_price else 0
                        print(f"      {pos.symbol} Pos #{pos.order_id}: ${pos.trailing_stop_price:.4f} (+{protection:.2f}%)")

        except Exception as e:
            print(f"❌ Error en monitoreo de posiciones: {e}")

    async def _check_position_exit_conditions(self, position, current_price: float) -> tuple:
        """🛡️ Verificar condiciones de salida para una posición"""
        try:
            # Calcular PnL actual
            if position.side == 'BUY':
                pnl_percent = ((current_price - position.entry_price) / position.entry_price) * 100
            else:
                pnl_percent = ((position.entry_price - current_price) / position.entry_price) * 100

            # ✅ TRADICIONAL: Stop Loss y Take Profit (solo si trailing no está activo)
            if hasattr(position, 'trailing_stop_active') and not position.trailing_stop_active:
                # Stop Loss tradicional
                if pnl_percent <= -3.0:
                    return True, f"STOP_LOSS_TRADICIONAL (-{abs(pnl_percent):.2f}%)"

                # Take Profit tradicional
                if pnl_percent >= 6.0:
                    return True, f"TAKE_PROFIT_TRADICIONAL (+{pnl_percent:.2f}%)"

            # ✅ CIRCUITO: Pérdida máxima diaria
            if await self._daily_loss_exceeds_limit():
                return True, "CIRCUIT_BREAKER_DAILY_LOSS"

            return False, ""

        except Exception as e:
            print(f"❌ Error verificando condiciones de salida: {e}")
            return False, ""

    async def _close_positions_batch(self, positions_and_reasons: List[Tuple]) -> None:
        """🚀 Cerrar múltiples posiciones en lote - CON ÓRDENES REALES"""
        try:
            print(f"🔥 Iniciando cierre de {len(positions_and_reasons)} posición(es)...")

            for position, reason in positions_and_reasons:
                try:
                    print(f"🛑 CERRANDO POSICIÓN {position.symbol} Pos #{position.order_id}:")
                    print(f"   📍 Entrada: ${position.entry_price:.4f}")
                    print(f"   💰 Actual: ${position.current_price:.4f}")
                    print(f"   📊 PnL: {position.unrealized_pnl_percent:.2f}% (${position.unrealized_pnl_usd:.2f})")
                    print(f"   🏷️ Razón: {reason}")

                    # ✅ EJECUTAR ORDEN REAL DE CIERRE EN BINANCE
                    order_result = await self._execute_sell_order(position)

                    if order_result:
                        # Usar precio real de ejecución
                        real_close_price = float(order_result.get('fills', [{}])[0].get('price', position.current_price))
                        real_quantity = float(order_result.get('executedQty', position.size))

                        # Calcular PnL real con precio de ejecución
                        if position.side == 'BUY':
                            real_pnl_percent = ((real_close_price - position.entry_price) / position.entry_price) * 100
                        else:
                            real_pnl_percent = ((position.entry_price - real_close_price) / position.entry_price) * 100

                        real_pnl_usd = (real_pnl_percent / 100) * (real_quantity * position.entry_price)

                        print(f"✅ ORDEN REAL EJECUTADA:")
                        print(f"   🆔 Order ID: {order_result.get('orderId')}")
                        print(f"   💲 Precio real: ${real_close_price:.4f}")
                        print(f"   📊 PnL real: {real_pnl_percent:.2f}% (${real_pnl_usd:.2f})")

                        # Actualizar métricas con datos reales
                        self.session_pnl += real_pnl_usd

                    else:
                        print(f"❌ Error ejecutando orden real - usando datos estimados")
                        # Fallback a datos estimados si falla la orden
                        real_pnl_percent = position.unrealized_pnl_percent
                        real_pnl_usd = position.unrealized_pnl_usd
                        self.session_pnl += real_pnl_usd

                    # Logging de la operación
                    await self.database.log_event(
                        'TRADE',
                        'POSITION_CLOSED',
                        f"{position.symbol}: {reason} - PnL: {real_pnl_percent:.2f}% - Order: {order_result.get('orderId', 'FAILED') if order_result else 'FAILED'}"
                    )

                    # Actualizar métricas
                    self.metrics['total_trades'] += 1
                    if real_pnl_usd > 0:
                        self.metrics['profitable_trades'] += 1

                    print(f"✅ Posición {position.symbol} cerrada exitosamente")

                except Exception as e:
                    print(f"❌ Error cerrando {position.symbol}: {e}")
                    await self.database.log_event('ERROR', 'TRADING', f'Error cerrando posición {position.symbol}: {e}')
                    continue

            print(f"🎯 Proceso de cierre completado")

        except Exception as e:
            print(f"❌ Error en cierre de posiciones en lote: {e}")

    async def _execute_sell_order(self, position) -> Optional[Dict]:
        """🔥 EJECUTAR ORDEN REAL DE VENTA EN BINANCE"""
        try:
            # Determinar lado de la orden de cierre
            close_side = 'SELL' if position.side == 'BUY' else 'BUY'

            # Obtener precio actual para la orden
            current_price = await self.get_current_price(position.symbol)

            # Preparar parámetros de orden
            timestamp = int(time.time() * 1000)

            # Ajustar cantidad según filtros del símbolo
            adjusted_quantity = await self._adjust_quantity_for_symbol(position.symbol, position.size)

            params = {
                'symbol': position.symbol,
                'side': close_side,
                'type': 'MARKET',  # Orden de mercado para cierre inmediato
                'quantity': f"{adjusted_quantity:.8f}".rstrip('0').rstrip('.'),
                'timestamp': timestamp,
                'recvWindow': 10000
            }

            # Crear signature
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

            print(f"📡 Ejecutando orden de cierre: {close_side} {params['quantity']} {position.symbol}")

            # Ejecutar orden POST /api/v3/order
            async with aiohttp.ClientSession() as session:
                url = f"{self.config.base_url}/api/v3/order"

                async with session.post(url, data=params, headers=headers) as response:
                    if response.status == 200:
                        result = await response.json()
                        print(f"🎉 ORDEN DE CIERRE EJECUTADA: {result['orderId']}")
                        return result
                    else:
                        error_text = await response.text()
                        print(f"❌ Error Binance API: {response.status} - {error_text}")
                        return None

        except Exception as e:
            print(f"❌ ERROR ejecutando orden de cierre: {e}")
            return None

    async def _adjust_quantity_for_symbol(self, symbol: str, quantity: float) -> float:
        """🔧 Ajustar cantidad según filtros del símbolo"""
        try:
            # Obtener información del símbolo
            async with aiohttp.ClientSession() as session:
                url = f"{self.config.base_url}/api/v3/exchangeInfo"
                params = {'symbol': symbol}

                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()

                        if 'symbols' in data and len(data['symbols']) > 0:
                            symbol_info = data['symbols'][0]

                            # Buscar filtro LOT_SIZE
                            for filter_info in symbol_info.get('filters', []):
                                if filter_info['filterType'] == 'LOT_SIZE':
                                    step_size = float(filter_info['stepSize'])
                                    min_qty = float(filter_info['minQty'])

                                    # Ajustar cantidad al step size
                                    adjusted_qty = max(min_qty, quantity)
                                    adjusted_qty = round(adjusted_qty / step_size) * step_size

                                    return adjusted_qty

            # Si no se puede obtener filtros, usar cantidad original
            return quantity

        except Exception as e:
            print(f"❌ Error ajustando cantidad para {symbol}: {e}")
            return quantity

    async def _daily_loss_exceeds_limit(self, max_daily_loss_percent: float = None) -> bool:
        """🚨 Verificar si se ha excedido la pérdida máxima diaria"""
        try:
            # Obtener límite desde .env si no se proporciona
            if max_daily_loss_percent is None:
                max_daily_loss_percent = float(os.getenv('MAX_DAILY_LOSS_PERCENT', '10.0'))
            # Obtener snapshot actual
            snapshot = await self.portfolio_manager.get_portfolio_snapshot()

            # Calcular pérdida porcentual del día
            if snapshot.total_balance_usd > 0:
                daily_pnl_percent = (snapshot.total_unrealized_pnl / snapshot.total_balance_usd) * 100

                if daily_pnl_percent <= -max_daily_loss_percent:
                    print(f"🚨 CIRCUIT BREAKER: Pérdida diaria {daily_pnl_percent:.2f}% >= {max_daily_loss_percent}%")
                    return True

            return False

        except Exception as e:
            print(f"❌ Error verificando pérdida diaria: {e}")
            return False

    async def _metrics_collector(self):
        """📊 Recolector de métricas del sistema"""
        while self.status == TradingManagerStatus.RUNNING:
            try:
                await asyncio.sleep(120)  # Cada 2 minutos

                # Recolectar métricas básicas
                await self._update_metrics()

            except Exception as e:
                await self.database.log_event('ERROR', 'METRICS', f'Error collecting metrics: {e}')
                await asyncio.sleep(60)

    async def _save_periodic_metrics(self):
        """💾 Guardar métricas periódicamente"""
        try:
            # Actualizar balance actual en risk manager
            total_balance = self.current_balance + self.session_pnl
            await self.risk_manager.update_balance(total_balance)

            trades_today = await self._get_total_trades_today()
            win_rate = await self._calculate_win_rate()

            # Calcular exposición total
            total_exposure = 0
            for position in self.active_positions.values():
                if hasattr(position, 'current_price') and position.current_price > 0:
                    total_exposure += position.quantity * position.current_price
                else:
                    total_exposure += position.quantity * position.entry_price

            exposure_percent = (total_exposure / self.current_balance) * 100 if self.current_balance > 0 else 0

            metrics_data = {
                'timestamp': datetime.now(),
                'total_balance': total_balance,
                'daily_pnl': self.session_pnl,
                'total_pnl': self.session_pnl,  # Para sesión actual
                'daily_return_percent': (self.session_pnl / self.current_balance) * 100 if self.current_balance > 0 else 0,
                'total_return_percent': (self.session_pnl / self.current_balance) * 100 if self.current_balance > 0 else 0,
                'current_drawdown': 0.0,  # Calcular en futuras versiones
                'max_drawdown': 0.0,
                'sharpe_ratio': None,
                'win_rate': win_rate,
                'profit_factor': None,
                'active_positions_count': len(self.active_positions),
                'total_exposure_usd': total_exposure,
                'exposure_percent': exposure_percent,
                'trades_today': trades_today,
                'avg_trade_duration_minutes': None,
                'api_calls_today': self.metrics.get('api_calls_count', 0),
                'error_count_today': self.metrics.get('error_count', 0),
                'last_balance_update': self.metrics.get('last_balance_update', None)
            }

            await self.database.save_performance_metrics(metrics_data)

            # Mostrar resumen de métricas cada 10 ciclos
            if self.metrics['total_checks'] % 10 == 0:
                print(f"\n📊 RESUMEN DE MÉTRICAS:")
                print(f"   📈 Balance total: ${total_balance:.2f}")
                print(f"   💰 PnL sesión: ${self.session_pnl:.2f}")
                print(f"   📊 Trades hoy: {trades_today}")
                print(f"   🎯 Win rate: {win_rate:.1f}%")
                print(f"   💼 Exposición: {exposure_percent:.1f}%")
                print(f"   🔧 API calls: {self.metrics.get('api_calls_count', 0)}")
                print(f"   ❌ Errores: {self.metrics.get('error_count', 0)}")

        except Exception as e:
            print(f"❌ Error guardando métricas: {e}")

    async def _get_total_trades_today(self) -> int:
        """📊 Obtener total de trades de hoy"""
        try:
            trades = await self.database.get_trades_history(days=1)
            return len(trades)
        except:
            return self.trade_count

    async def _calculate_win_rate(self) -> float:
        """🎯 Calcular win rate"""
        try:
            trades = await self.database.get_trades_history(days=7, is_active=False)
            if not trades:
                return 0.0

            wins = sum(1 for trade in trades if trade.get('pnl_usd', 0) > 0)
            return (wins / len(trades)) * 100
        except:
            return 0.0

    async def _update_metrics(self):
        """📈 Actualizar métricas internas"""
        self.metrics['active_positions'] = len(self.active_positions)
        self.metrics['session_pnl'] = self.session_pnl

    async def _handle_pause_state(self):
        """⏸️ Manejar estado de pausa"""
        while self.status == TradingManagerStatus.PAUSED:
            print("⏸️ Sistema pausado - esperando reanudación...")
            await asyncio.sleep(10)

    async def _handle_error(self, error: Exception):
        """❌ Manejar errores del sistema"""
        error_msg = f"Error en loop principal: {error}"
        print(f"❌ {error_msg}")

        await self.database.log_event('ERROR', 'SYSTEM', error_msg)

        # Si hay muchos errores consecutivos, pausar el sistema
        self.metrics['error_count'] += 1

        if self.metrics['error_count'] > 10:
            await self.pause_trading_with_reason("Demasiados errores consecutivos")

    async def _send_discord_notification(self, message: str):
        """💬 Enviar notificación a Discord usando Smart Notifier"""
        try:
            # Importar Smart Discord Notifier si no está disponible
            if not hasattr(self, 'discord_notifier'):
                from smart_discord_notifier import SmartDiscordNotifier, NotificationPriority
                self.discord_notifier = SmartDiscordNotifier()

                # Configurar filtros conservadores para evitar spam
                self.discord_notifier.configure_filters(
                    min_trade_value_usd=12.0,          # Solo trades > $12
                    min_pnl_percent_notify=2.0,        # Solo PnL > 2%
                    max_notifications_per_hour=8,      # Max 8/hora
                    max_notifications_per_day=40,      # Max 40/día
                    suppress_similar_minutes=10,       # 10 min entre similares
                    only_profitable_trades=False,      # Notificar pérdidas también
                    emergency_only_mode=False          # Todas las prioridades
                )

            # Determinar prioridad basada en el mensaje
            from smart_discord_notifier import NotificationPriority

            if "EMERGENCIA" in message or "PARADA" in message:
                priority = NotificationPriority.CRITICAL
            elif "ERROR" in message or "❌" in message:
                priority = NotificationPriority.HIGH
            elif "NUEVA POSICIÓN" in message or "CERRADA" in message:
                priority = NotificationPriority.MEDIUM
            else:
                priority = NotificationPriority.LOW

            # Enviar usando el Smart Notifier
            await self.discord_notifier.send_system_notification(message, priority)

        except Exception as e:
            print(f"⚠️ Error enviando notificación Discord: {e}")

    # Métodos de control del sistema

    async def pause_trading_with_reason(self, reason: str):
        """⏸️ Pausar trading con razón específica"""
        self.status = TradingManagerStatus.PAUSED
        await self.database.log_event('WARNING', 'SYSTEM', f'Trading pausado: {reason}')
        print(f"⏸️ Trading pausado: {reason}")

    async def resume_trading(self):
        """▶️ Reanudar trading"""
        if self.status == TradingManagerStatus.PAUSED:
            self.status = TradingManagerStatus.RUNNING
            await self.database.log_event('INFO', 'SYSTEM', 'Trading reanudado')
            print("▶️ Trading reanudado")

    async def emergency_stop(self):
        """🚨 Parada de emergencia"""
        self.status = TradingManagerStatus.EMERGENCY_STOP

        # Cerrar todas las posiciones activas
        for order_id in list(self.active_positions.keys()):
            await self._close_position(order_id, "EMERGENCY_STOP")

        await self.database.log_event('CRITICAL', 'SYSTEM', 'Parada de emergencia activada')
        print("🚨 PARADA DE EMERGENCIA ACTIVADA")

    async def get_system_status(self) -> Dict:
        """📊 Obtener estado completo del sistema"""

        # Calcular uptime
        uptime_seconds = 0
        if self.start_time:
            uptime_seconds = time.time() - self.start_time

        # Calcular exposición total
        total_exposure = 0
        for position in self.active_positions.values():
            if hasattr(position, 'current_price') and position.current_price > 0:
                total_exposure += position.quantity * position.current_price
            else:
                total_exposure += position.quantity * position.entry_price

        return {
            'status': self.status,
            'environment': self.config.environment,
            'symbols_trading': self.symbols,
            'check_interval': self.check_interval,
            'uptime_minutes': uptime_seconds / 60,
            'current_balance_usdt': self.current_balance,
            'session_pnl': self.session_pnl,
            'total_balance': self.current_balance + self.session_pnl,
            'active_positions': len(self.active_positions),
            'total_exposure_usd': total_exposure,
            'exposure_percent': (total_exposure / self.current_balance) * 100 if self.current_balance > 0 else 0,
            'trade_count': self.trade_count,
            'current_prices': self.current_prices,
            'last_check': self.last_check_time.isoformat() if self.last_check_time else None,
            'last_balance_update': self.last_balance_update.isoformat() if self.last_balance_update else None,
            'account_info': {
                'usdt_balance': self.account_info.usdt_balance if self.account_info else 0,
                'total_balance_usd': self.account_info.total_balance_usd if self.account_info else 0,
                'other_balances': {k: v for k, v in self.account_info.balances.items()
                                 if k != 'USDT' and v['total'] > 0} if self.account_info else {}
            },
            'metrics': self.metrics
        }

    async def shutdown(self):
        """🔄 Apagado controlado del sistema"""
        print("🔄 Iniciando apagado del sistema...")

        self.status = TradingManagerStatus.STOPPED

        # Cerrar posiciones si hay alguna activa
        if self.active_positions:
            print(f"📉 Cerrando {len(self.active_positions)} posiciones activas...")
            for order_id in list(self.active_positions.keys()):
                await self._close_position(order_id, "SYSTEM_SHUTDOWN")

        # Guardar métricas finales
        await self._save_periodic_metrics()

        # Log final
        await self.database.log_event('INFO', 'SYSTEM', 'Sistema apagado correctamente')

        print("✅ Sistema apagado correctamente")

async def main():
    """🎯 Función principal para testing directo"""
    print("🧪 Modo de prueba - Simple Professional Trading Manager")

    manager = SimpleProfessionalTradingManager()
    try:
        await manager.initialize()
        print("✅ Manager inicializado correctamente")

        # Mostrar estado
        status = await manager.get_system_status()
        print(f"📊 Estado: {status}")

    except Exception as e:
        print(f"❌ Error en testing: {e}")
        if manager:
            await manager.emergency_stop()

if __name__ == "__main__":
    asyncio.run(main())
