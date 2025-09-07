#!/usr/bin/env python3
"""
🚀 SIMPLE PROFESSIONAL TRADING MANAGER
====================================

Gestor de trading automatizado profesional con:
- Sistema TCN avanzado para predicciones
- Gestión inteligente de riesgo y diversificación
- Monitoreo en tiempo real de posiciones
- Notificaciones automáticas vía Discord
- Análisis de contexto de mercado
"""

import os
import sys
import json
import hmac
import hashlib
import asyncio
import aiohttp
import sqlite3
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN
from typing import Dict, List, Optional, Any, Tuple
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from binance import AsyncClient
from binance.enums import *
from config.trading_config import get_trading_config
from loss_protection import LossProtectionManager

# Importar nuestros módulos de risk y database
from advanced_risk_manager import AdvancedRiskManager, Position, RiskLimits
from trading_database import TradingDatabase

# Importar el módulo de Smart Discord Notifier
from smart_discord_notifier import SmartDiscordNotifier

# ✅ NUEVO: Importar Professional Portfolio Manager
from professional_portfolio_manager import ProfessionalPortfolioManager, Position as PortfolioManagerPosition

# ✅ NUEVO: Importar Portfolio Diversification Manager
from portfolio_diversification_manager import PortfolioDiversificationManager, PortfolioPosition

# ✅ ELIMINADO: Real Market Data Provider no se usa - el TCN Ensemble Predictor ya usa el motor centralizado
# from real_market_data_provider import RealMarketDataProvider

# ✅ NUEVO: Importar Loss Protection Manager
from loss_protection import LossProtectionManager

# Importar otros módulos necesarios
import asyncio
import json
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import aiohttp
import ccxt.async_support as ccxt
import numpy as np
import pandas as pd
from binance.client import Client
from binance.enums import *

from config.trading_config import get_trading_config
from loss_protection import LossProtectionManager
import asyncio
import hashlib
import json
import logging
import time
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from binance.exceptions import BinanceAPIException

# from config.api_config import get_api_config  # No usado
from config.trading_config import get_trading_config
from loss_protection import LossProtectionManager
#from utils.binance_client import create_binance_client
#from utils.discord_webhook import send_discord_notification

# ✅ CORREGIDO: Cargar .env desde el directorio raíz del proyecto
import os
from pathlib import Path

# Obtener la ruta del directorio raíz del proyecto (donde está el archivo .env)
project_root = Path(__file__).parent
env_path = project_root / '.env'

# Cargar variables de entorno desde .env
if env_path.exists():
    load_dotenv(env_path)
    print(f"✅ Archivo .env cargado desde: {env_path}")
else:
    print(f"⚠️ Archivo .env no encontrado en: {env_path}")
    # Intentar cargar desde el directorio actual como fallback
    load_dotenv()
    print("⚠️ Cargando .env desde directorio actual como fallback")

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
    """🚀 Trading Manager Profesional"""

    def __init__(self):
        """🏗️ Constructor del Trading Manager Profesional"""
        print("🚀 Inicializando Simple Professional Trading Manager - TCN V2 INTEGRADO...")

        # Estados y configuración
        self.status = TradingManagerStatus.STOPPED
        self.session_pnl = 0.0
        self.current_balance = 0.0
        self.binance_config = self._load_config()

        # ✅ NUEVO: Sistema de cooldown y estabilidad de señales (RELAJADO PARA ENSEMBLE)
        self.signal_history = {}  # Historial de señales por símbolo
        self.last_position_action = {}  # Última acción de posición por símbolo
        self.signal_cooldown = {  # Tiempos de enfriamiento por símbolo (en minutos) - RELAJADOS
            'ETHUSDT': 0,  # ETH: Sin cooldown (era 15 minutos)
            'BTCUSDT': 0,  # BTC: Sin cooldown (era 10 minutos)
            'BNBUSDT': 0,  # BNB: Sin cooldown (era 12 minutos)
            'XRPUSDT': 0,  # XRP: Sin cooldown (era 12 minutos)
            'DOTUSDT': 0,  # DOT: Sin cooldown (era 12 minutos)
            'ADAUSDT': 0,  # ADA: Sin cooldown (nuevo par activado)
            'SOLUSDT': 0,  # SOL: Sin cooldown (nuevo par activado)
            'LINKUSDT': 0,  # LINK: Sin cooldown (nuevo par activado)
        }
        
        # ✅ NUEVO: Sistema de cooldown post-cierre RELAJADO - ENSEMBLE DE 3 MODELOS
        self.post_close_cooldown = {  # Cooldown después de cerrar posiciones (en minutos) - RELAJADO
            'ETHUSDT': {
                'profit_close': 2,     # ✅ RELAJADO: 2 min después de cerrar con ganancias (era 5)
                'loss_close': 5,       # ✅ RELAJADO: 3 min después de cerrar con pérdidas (era 6)
                'trailing_stop_close': 3, # ✅ RELAJADO: 2 min después de cerrar por trailing stop (era 5)
                'reversal_close': 6    # ✅ RELAJADO: 3 min después de cerrar por reversión (era 8)
            },
            'BTCUSDT': {
                'profit_close': 2,     # ✅ RELAJADO: 2 min después de cerrar con ganancias (era 5)
                'loss_close': 5,       # ✅ RELAJADO: 3 min después de cerrar con pérdidas (era 6)
                'trailing_stop_close': 3,  # ✅ RELAJADO: 2 min después de cerrar por trailing stop (era 5)
                'reversal_close': 6    # ✅ RELAJADO: 3 min después de cerrar por reversión (era 7)
            },
            'BNBUSDT': {
                'profit_close': 2,     # ✅ RELAJADO: 2 min después de cerrar con ganancias (era 5)
                'loss_close': 5,       # ✅ RELAJADO: 3 min después de cerrar con pérdidas (era 6)
                'trailing_stop_close': 3,  # ✅ RELAJADO: 2 min después de cerrar por trailing stop (era 5)
                'reversal_close': 6    # ✅ RELAJADO: 3 min después de cerrar por reversión (era 8)
            },
            'XRPUSDT': {
                'profit_close': 2,     # ✅ RELAJADO: 2 min después de cerrar con ganancias (era 5)
                'loss_close': 5,       # ✅ RELAJADO: 2 min después de cerrar con pérdidas (era 6)
                'trailing_stop_close': 3,  # ✅ RELAJADO: 2 min después de cerrar por trailing stop (era 5)
                'reversal_close': 6    # ✅ RELAJADO: 2 min después de cerrar por reversión (era 6)
            },
            'DOTUSDT': {
                'profit_close': 2,     # ✅ RELAJADO: 2 min después de cerrar con ganancias (era 5)
                'loss_close': 5,       # ✅ RELAJADO: 3 min después de cerrar con pérdidas (era 6)
                'trailing_stop_close': 3,  # ✅ RELAJADO: 2 min después de cerrar por trailing stop (era 5)
                'reversal_close': 6    # ✅ RELAJADO: 3 min después de cerrar por reversión (era 8)
            },
            'ADAUSDT': {
                'profit_close': 2,     # ✅ RELAJADO: 2 min después de cerrar con ganancias (era 5)
                'loss_close': 5,       # ✅ RELAJADO: 3 min después de cerrar con pérdidas (era 6)
                'trailing_stop_close': 3,  # ✅ RELAJADO: 2 min después de cerrar por trailing stop (era 5)
                'reversal_close': 6    # ✅ RELAJADO: 3 min después de cerrar por reversión (era 7)
            },
            'SOLUSDT': {
                'profit_close': 2,     # ✅ RELAJADO: 2 min después de cerrar con ganancias (era 5)
                'loss_close': 5,       # ✅ RELAJADO: 3 min después de cerrar con pérdidas (era 6)
                'trailing_stop_close': 3,  # ✅ RELAJADO: 2 min después de cerrar por trailing stop (era 5)
                'reversal_close': 6    # ✅ RELAJADO: 3 min después de cerrar por reversión (era 7)
            },
            'POLUSDT': {
                'profit_close': 2,     # ✅ RELAJADO: 2 min después de cerrar con ganancias (era 5)
                'loss_close': 5,       # ✅ RELAJADO: 3 min después de cerrar con pérdidas (era 6)
                'trailing_stop_close': 3,  # ✅ RELAJADO: 2 min después de cerrar por trailing stop (era 5)
                'reversal_close': 6    # ✅ RELAJADO: 3 min después de cerrar por reversión (era 7)
            }
        }
        
        # ✅ NUEVO: Tracking de cierres recientes para aplicar cooldowns
        self.recent_closes = {}  # {symbol: {'last_close_time': datetime, 'close_type': str, 'pnl_percent': float}}
        
        # ✅ NUEVO: Tracking de cierres recientes para aplicar cooldowns
        self.recent_closes = {}  # {symbol: {'last_close_time': datetime, 'close_type': str, 'pnl_percent': float}}
        
        self.eth_position_protection = {  # Protección específica para ETH (RELAJADA)
            'last_close_time': None,
            'min_hold_time_minutes': 2,  # ✅ RELAJADO: De 20 min a 10 min para ETH
            'consecutive_signals': 0,     # Contador de señales consecutivas
            'signal_confirmation_required': 1 # ✅ RELAJADO: Requiere solo 1 señal consecutiva para ETH (era 2)
        }

        # ✅ NUEVO: Sistema de reversión de señal mejorado con señales consecutivas - MUY ESTRICTO PARA MAYOR ESTABILIDAD
        self.reversal_tracking = {}  # Tracking de señales de reversión por símbolo
        self.reversal_config = {
            'ETHUSDT': {
                'required_consecutive_signals': 50,  # ✅ MUY ESTRICTO: Aumentado de 2 a 20 para mayor estabilidad
                'timeout_minutes': 1.8,  # ✅ MANTENER: Timeout de 1.8 min
                'min_confidence_per_signal': 78.0,  # ✅ MANTENER: 78% confianza mínima por señal
                'cumulative_confidence_threshold': 80.0,  # ✅ MANTENER: 80% confianza acumulativa
                'min_interval_between_signals_minutes': 1.0  # ✅ MANTENER: 1.0 min entre señales
            },
            'BTCUSDT': {
                'required_consecutive_signals': 50,  # ✅ MUY ESTRICTO: Aumentado de 2 a 20 para mayor estabilidad
                'timeout_minutes': 1.5,  # ✅ MANTENER: Timeout de 1.5 min
                'min_confidence_per_signal': 76.0,  # ✅ MANTENER: 76% confianza mínima por señal
                'cumulative_confidence_threshold': 78.0,  # ✅ MANTENER: 78% confianza acumulativa
                'min_interval_between_signals_minutes': 0.8  # ✅ MANTENER: 0.8 min entre señales
            },
            'BNBUSDT': {
                'required_consecutive_signals': 50,  # ✅ MUY ESTRICTO: Aumentado de 2 a 20 para mayor estabilidad
                'timeout_minutes': 1.8,  # ✅ MANTENER: Timeout de 1.8 min
                'min_confidence_per_signal': 77.0,  # ✅ MANTENER: 77% confianza mínima por señal
                'cumulative_confidence_threshold': 79.0,  # ✅ MANTENER: 79% confianza acumulativa
                'min_interval_between_signals_minutes': 1.0  # ✅ MANTENER: 1.0 min entre señales
            },
            'XRPUSDT': {
                'required_consecutive_signals': 50,  # ✅ MUY ESTRICTO: Aumentado de 2 a 20 para mayor estabilidad
                'timeout_minutes': 1.2,  # ✅ MANTENER: Timeout de 1.2 min
                'min_confidence_per_signal': 74.0,  # ✅ MANTENER: 74% confianza mínima por señal
                'cumulative_confidence_threshold': 76.0,  # ✅ MANTENER: 76% confianza acumulativa
                'min_interval_between_signals_minutes': 0.6  # ✅ MANTENER: 0.6 min entre señales
            },
            'DOTUSDT': {
                'required_consecutive_signals': 50,  # ✅ MUY ESTRICTO: Aumentado de 2 a 20 para mayor estabilidad
                'timeout_minutes': 1.5,  # ✅ MANTENER: Timeout de 1.5 min
                'min_confidence_per_signal': 75.0,  # ✅ MANTENER: 75% confianza mínima por señal
                'cumulative_confidence_threshold': 77.0,  # ✅ MANTENER: 77% confianza acumulativa
                'min_interval_between_signals_minutes': 0.8  # ✅ MANTENER: 0.8 min entre señales
            },
            'ADAUSDT': {
                'required_consecutive_signals': 50,  # ✅ MUY ESTRICTO: Aumentado de 2 a 20 para mayor estabilidad
                'timeout_minutes': 1.3,  # ✅ MANTENER: Timeout de 1.3 min
                'min_confidence_per_signal': 76.0,  # ✅ MANTENER: 76% confianza mínima por señal
                'cumulative_confidence_threshold': 78.0,  # ✅ MANTENER: 78% confianza acumulativa
                'min_interval_between_signals_minutes': 0.7  # ✅ MANTENER: 0.7 min entre señales
            },
            'SOLUSDT': {
                'required_consecutive_signals': 50,  # ✅ ESTRICTO: Para SOL volátil
                'timeout_minutes': 1.2,  # ✅ Timeout de 1.2 min
                'min_confidence_per_signal': 75.0,  # ✅ 75% confianza mínima por señal
                'cumulative_confidence_threshold': 77.0,  # ✅ 77% confianza acumulativa
                'min_interval_between_signals_minutes': 0.6  # ✅ 0.6 min entre señales
            },
            'POLUSDT': {
                'required_consecutive_signals': 50,  # ✅ ESTRICTO: Para POL (token nuevo)
                'timeout_minutes': 1.3,  # ✅ Timeout de 1.3 min
                'min_confidence_per_signal': 75.0,  # ✅ 75% confianza mínima por señal
                'cumulative_confidence_threshold': 77.0,  # ✅ 77% confianza acumulativa
                'min_interval_between_signals_minutes': 0.7  # ✅ 0.7 min entre señales
            }
        }

        # Configuración de símbolos y gestores
        # ✅ ACTUALIZADO: Símbolos con modelos TCN entrenados disponibles
        self.symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'DOTUSDT', 'ADAUSDT', 'SOLUSDT', 'POLUSDT']  # ✅ AGREGADO: SOLUSDT y POLUSDT (reemplazo de POLUSDT)

        # ⚠️ PARES PENDIENTES (sin modelos): []
        self.excluded_symbols = []  # ✅ REMOVIDO: SOLUSDT - ahora está activo

        print(f"📊 Pares activos: {self.symbols}")
        print(f"⏸️ Pares excluidos (sin modelos): {self.excluded_symbols}")

        self.check_interval = 60  # 1 minuto

        # Estado del sistema
        self.status = TradingManagerStatus.STOPPED
        self.database: Optional[TradingDatabase] = None
        self.risk_manager: Optional[AdvancedRiskManager] = None
        self.client = None

        # ✅ NUEVO: Professional Portfolio Manager
        self.portfolio_manager: Optional[ProfessionalPortfolioManager] = None

        # ✅ NUEVO: Portfolio Diversification Manager
        self.diversification_manager = PortfolioDiversificationManager()

        # ✅ NUEVO: Loss Protection Manager
        self.loss_protection = LossProtectionManager()

        # Balance y trading - ✅ CORREGIDO: Inicializar en 0, obtener de Binance
        self.current_balance = 0.0  # Se actualizará desde Binance
        self.trade_count = 0
        # ✅ CORREGIDO: Clave por order_id para múltiples posiciones por símbolo
        self.active_positions: Dict[str, Position] = {}
        self.account_info = None

        # ✅ NUEVO: Sistema de priorización de señales
        self.pending_signals = {}  # {symbol: signal_data} para priorización

        # ✅ NUEVO: Portfolio tracking
        self.last_portfolio_snapshot = None
        self.last_tcn_report_time = None

        # Smart Discord Notifier
        self.discord_notifier = SmartDiscordNotifier()

        # 🧠 INICIALIZAR TCN REAL OBLIGATORIO
        self.tcn_predictor = None
        self._initialize_tcn_predictor_sync()

        # Configurar filtros conservadores para evitar spam
        self.discord_notifier.configure_filters(
            min_trade_value_usd=12.0,          # Solo trades > $12
            min_pnl_percent_notify=0.5,        # Solo PnL > 2%
            max_notifications_per_hour=10,      # Max 8/hora
            max_notifications_per_day=60,      # Max 40/día
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

        # ✅ NUEVO: Loss Protection Manager
        self.loss_protection = LossProtectionManager()

    def _initialize_tcn_predictor_sync(self):
        """🧠 Inicialización síncrona básica del predictor ENSEMBLE V3 DINÁMICO"""
        try:
            from tcn_ensemble_predictor import TCNEnsemblePredictor
            self.tcn_predictor = TCNEnsemblePredictor()

            # ✅ NUEVO: Cargar modelos dinámicamente (TCN + Técnico)
            load_success = self.tcn_predictor.load_definitivo_v3_models()
            
            if load_success:
                model_info = self.tcn_predictor.get_model_info()
                operation_mode = model_info.get('operation_mode', 'technical_only')
                
                print("🎯 Predictor ENSEMBLE V3 DINÁMICO inicializado correctamente")
                print(f"   🚀 Modo de operación: {operation_mode.upper()}")
                print(f"   📊 Modelos cargados: {model_info['loaded_models']}")
                print(f"   ⏰ Timeframes disponibles: {', '.join(model_info['available_timeframes'])}")
                print(f"   🎯 Símbolos soportados: {self.tcn_predictor.symbols}")
                print(f"   🏗️ Tipo: {model_info['model_type']}")
                print("   ✅ DOTUSDT incluido en predicciones y notificaciones")
                print("   🔧 Sistema híbrido TCN + Técnico")
                print("   ✅ MODELOS ENSEMBLE CARGADOS CORRECTAMENTE")
                
                if operation_mode == "hybrid":
                    print("   🔧 FUNCIONANDO EN MODO HÍBRIDO TCN + TÉCNICO")
                    print("   📈 Predictores disponibles: 1m, 3m, 5m + TCN")
                    print("   💡 Ensemble híbrido con análisis técnico y ML")
                    print("   🎯 Análisis de convergencia entre timeframes")
                    print("   🚀 Boost de confianza para señales alineadas")
                    
                return True
            else:
                raise Exception("No se pudieron inicializar capacidades de predicción técnicas")
        except Exception as e:
            print(f"❌ ERROR CRÍTICO: No se pudo inicializar ENSEMBLE PREDICTOR: {e}")
            print("🚨 SISTEMA REQUIERE CAPACIDADES DE PREDICCIÓN ENSEMBLE")
            import traceback
            print(f"🔍 Traceback completo: {traceback.format_exc()}")
            raise Exception(f"Ensemble Predictor requerido pero falló en constructor: {e}")

    async def _initialize_tcn_predictor(self):
        """🧠 Verificación adicional de autenticidad de datos de Binance para Ensemble Predictor"""
        try:
            # ✅ NUEVO: Verificar autenticidad de datos de Binance
            print("🔍 VERIFICANDO AUTENTICIDAD DE DATOS DE BINANCE...")
            try:
                binance_verified = await self.tcn_predictor.verify_binance_data_authenticity("BTCUSDT", "5m")
                if not binance_verified:
                    print("❌ ERROR: No se pudieron verificar datos de Binance")
                    print("💡 Verifica tu conexión a internet y la API de Binance")
                    raise Exception("Datos de Binance no verificados")
                print("✅ Autenticidad de datos de Binance verificada")
                return True
            except Exception as e:
                print(f"⚠️ Advertencia: No se pudo verificar autenticidad de datos: {e}")
                print("🔄 Continuando con inicialización...")
                return False
        except Exception as e:
            print(f"❌ ERROR CRÍTICO: No se pudo verificar autenticidad de datos: {e}")
            return False

    def _load_config(self) -> BinanceConfig:
        """⚙️ Cargar configuración desde variables de entorno"""
        api_key = os.getenv('BINANCE_API_KEY')
        secret_key = os.getenv('BINANCE_SECRET_KEY')

        if not api_key or not secret_key:
            raise ValueError("❌ BINANCE_API_KEY y BINANCE_SECRET_KEY son requeridos en .env")

        return BinanceConfig(
            api_key=api_key,
            secret_key=secret_key,
            base_url=os.getenv('BINANCE_BASE_URL', 'https://testnet.binance.vision'),
            environment=os.getenv('ENVIRONMENT', 'testnet')
        )

    def _generate_signature(self, params: str) -> str:
        """🔐 Generar firma para API de Binance"""
        return hmac.new(
            self.binance_config.secret_key.encode('utf-8'),
            params.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

    async def get_account_info(self) -> Optional[AccountInfo]:
        """💰 Obtener información completa de la cuenta de Binance"""
        try:
            params = {
                'timestamp': int(time.time() * 1000),
                'recvWindow': 10000
            }
            query_string = '&'.join([f"{key}={value}" for key, value in params.items()])
            signature = self._generate_signature(query_string)

            headers = {
                'X-MBX-APIKEY': self.binance_config.api_key
            }

            url = f"{self.binance_config.base_url}/api/v3/account"
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
                api_key=self.binance_config.api_key,
                secret_key=self.binance_config.secret_key,
                base_url=self.binance_config.base_url,
                discord_notifier=self.discord_notifier  # ✅ NUEVO: Pasar Discord notifier para trailing stops
            )
            
            # ✅ NUEVO: Inicializar el portfolio manager de manera asíncrona
            await self.portfolio_manager.initialize()
            print("✅ Portfolio Manager inicializado con Discord notifier")

            # 4. Inicializar Risk Manager
            await self._initialize_risk_manager()

            # ✅ 5. SINCRONIZAR POSICIONES EXISTENTES AL ARRANCAR
            await self._sync_positions_on_startup()

            # 6. Verificar conectividad
            await self._verify_connectivity()

            # 7. ✅ NUEVO: Verificar autenticidad de datos de Binance
            print("🔍 Verificando autenticidad de datos de Binance...")
            await self._initialize_tcn_predictor()

            # 8. Configurar monitoreo
            await self._setup_monitoring()

            self.start_time = time.time()
            self.last_heartbeat = datetime.now()
            self.status = TradingManagerStatus.RUNNING

            # Log inicial
            if self.database:
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
            # Verificar que portfolio_manager esté inicializado
            if not self.portfolio_manager:
                print("   ⚠️ Portfolio manager no inicializado, saltando sincronización")
                return

            # Obtener el estado real del portafolio desde el exchange
            snapshot = await self.portfolio_manager.get_portfolio_snapshot()
            if not snapshot or not snapshot.active_positions:
                print("   ✅ No se encontraron posiciones activas en el exchange.")
                return

            print(f"   🔍 Encontradas {len(snapshot.active_positions)} posiciones en el exchange. Sincronizando con DB...")

            # Verificar que database esté inicializado
            if not self.database:
                print("   ❌ Database no inicializado, no se puede sincronizar")
                return

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
        # ✅ CORREGIDO: Usar portfolio_manager.position_registry en lugar de active_positions
        return [pos for pos in self.portfolio_manager.position_registry.values() if pos.symbol == symbol]

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
        self.risk_manager = AdvancedRiskManager(self.binance_config)
        await self.risk_manager.initialize()
        print("🛡️ Risk Manager inicializado")

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
                url = f"{self.binance_config.base_url}/api/v3/ticker/price"
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
                """Loop de monitoreo de posiciones con trailing stops - ALTA FRECUENCIA"""
                while self.status == TradingManagerStatus.RUNNING:
                    try:
                        await self._position_monitor()
                        await asyncio.sleep(30)  # ✅ ALTA FRECUENCIA: Cada 30 segundos para TCN
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
        """🎯 Ejecutar loop principal de trading

        ✅ ALTA FRECUENCIA TCN: Actualizaciones frecuentes para trading en tiempo real
        - Bucle principal: cada 60 segundos (genera señales TCN, updates balance)
        - Monitor de posiciones: cada 30 segundos (máxima frecuencia TCN)
        - Reporte TCN: cada 1 minuto y 30 segundos (completo)
        - PRIORIDAD: Datos frescos > logs limpios
        """
        print("🎯 Iniciando loop principal de trading...")

        # ✅ NUEVO: Contador de errores por ciclo para reset automático
        consecutive_errors = 0
        max_consecutive_errors = 5  # Reducido de 10 a 5
        error_reset_time = 300  # 5 minutos para reset de errores
        last_successful_cycle = datetime.now()

        while self.status == TradingManagerStatus.RUNNING:
            try:
                loop_start_time = datetime.now()

                # Verificar si está pausado
                if self.pause_trading:
                    await self._handle_pause_state()
                    await asyncio.sleep(10)
                    continue

                # ✅ MEJORADO: Reset automático de contador de errores
                if consecutive_errors > 0:
                    time_since_success = (datetime.now() - last_successful_cycle).total_seconds()
                    if time_since_success > error_reset_time:
                        consecutive_errors = 0
                        print(f"🔄 Reset automático de errores después de {error_reset_time/60:.1f} minutos")

                # ✅ NUEVO: Generar reporte TCN cada 1 minuto y 30 segundos
                await self._generate_tcn_report_if_needed()

                # ✅ NUEVO: Limpiar tracking de reversión colgado (cada 5 minutos)
                await self._cleanup_stale_reversal_tracking()
                
                # ✅ NUEVO: Limpiar cooldowns post-cierre expirados automáticamente
                self.cleanup_expired_post_close_cooldowns()

                # ✅ MEJORADO: Mostrar información profesional en tiempo real
                await self._display_professional_info()

                # 1. Actualizar balance cada 3 minutos
                time_since_balance_update = None
                if self.last_balance_update:
                    time_since_balance_update = (datetime.now() - self.last_balance_update).total_seconds()

                if not self.last_balance_update or (time_since_balance_update is not None and time_since_balance_update > 180):  # 5 minutos
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

                # ✅ MEJORADO: Marcar ciclo exitoso
                consecutive_errors = 0
                last_successful_cycle = datetime.now()

                # ✅ NUEVO: Mostrar resumen cada ciclo
                loop_duration = (datetime.now() - loop_start_time).total_seconds()
                print(f"⏱️ Ciclo completado en {loop_duration:.1f}s")

                # 7. Esperar siguiente ciclo
                await asyncio.sleep(self.check_interval)

            except Exception as e:
                consecutive_errors += 1
                await self._handle_error_improved(e, consecutive_errors, max_consecutive_errors)

                # ✅ MEJORADO: Pausa adaptativa basada en el número de errores
                if consecutive_errors <= 2:
                    await asyncio.sleep(10)  # Pausa corta para errores menores
                elif consecutive_errors <= 4:
                    await asyncio.sleep(20)  # Pausa media
                else:
                    await asyncio.sleep(30)  # Pausa larga solo para errores críticos

    async def _handle_error_improved(self, error: Exception, consecutive_errors: int, max_consecutive_errors: int):
        """❌ Manejo mejorado de errores del sistema"""
        error_msg = f"Error en loop principal: {error}"
        error_type = type(error).__name__

        print(f"❌ {error_msg} (Error #{consecutive_errors})")
        print(f"🔍 Tipo de error: {error_type}")

        if self.database:
            await self.database.log_event('ERROR', 'SYSTEM', f"{error_msg} | Consecutivos: {consecutive_errors}")

        # ✅ MEJORADO: Manejo inteligente de diferentes tipos de errores
        if "timeout" in str(error).lower() or "connection" in str(error).lower():
            print("🌐 Error de conectividad detectado - continuando con pausa corta")
            return

        if "rate limit" in str(error).lower():
            print("⏳ Rate limit detectado - pausando 60 segundos")
            await asyncio.sleep(60)
            return

        # ✅ MEJORADO: Solo pausar después de muchos errores críticos
        if consecutive_errors >= max_consecutive_errors:
            print(f"🚨 {consecutive_errors} errores consecutivos - pausando sistema temporalmente")
            await self.pause_trading_with_reason(f"Demasiados errores consecutivos ({consecutive_errors})")

            # ✅ NUEVO: Auto-reanudar después de 10 minutos
            await asyncio.sleep(600)  # 10 minutos
            if self.status == TradingManagerStatus.PAUSED:
                print("🔄 Auto-reanudando trading después de pausa por errores")
                await self.resume_trading()

        # Actualizar métricas de errores
        self.metrics['error_count'] += 1
        self.metrics['last_error'] = {
            'type': error_type,
            'message': str(error)[:100],
            'timestamp': datetime.now().isoformat()
        }

    async def _generate_tcn_report_if_needed(self):
        """📊 Generar reporte TCN cada 1 minuto y 30 segundos"""
        try:
            now = datetime.now()

            # Verificar si es hora de generar reporte (cada 1 minuto y 30 segundos)
            should_generate = False

            if self.last_tcn_report_time is None:
                should_generate = True
            else:
                time_since_last = (now - self.last_tcn_report_time).total_seconds()
                if time_since_last >= 90:  # 1 minuto y 30 segundos
                    should_generate = True

            if should_generate:
                print("📊 Generando reporte TCN profesional (cada 1m 30s)...")

                # Verificar que portfolio_manager esté inicializado
                if not self.portfolio_manager:
                    print("⚠️ Portfolio manager no inicializado para reporte TCN")
                    return

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

                # ✅ NUEVO: Agregar estado de protección post-pérdidas al reporte
                protection_report = self.loss_protection.format_protection_report()

                # Combinar reportes
                full_report = tcn_report + tcn_models_report + diversification_report + protection_report

                # ✅ NUEVO: Agregar estado del tracking de reversión
                reversal_tracking_report = self._display_reversal_tracking_status()
                full_report += reversal_tracking_report

                # ✅ NUEVO: Agregar diagnóstico de sincronización de posiciones (SOLO PARA CONSOLA)
                sync_diagnosis = await self._generate_sync_diagnosis_section()
                full_report += sync_diagnosis
                
                # ✅ NUEVO: Limpieza automática si se detecta desincronización
                await self._auto_cleanup_if_desynchronized()

                # Mostrar en consola
                print("\n" + "="*80)
                print("🎯 REPORTE TCN PROFESSIONAL")
                print("="*80)
                print(full_report)
                print("="*80)
                
                # ✅ NUEVO: Comandos de diagnóstico rápido
                print("\n🔧 COMANDOS DE DIAGNÓSTICO DISPONIBLES:")
                print("   📊 await manager.diagnose_position_synchronization() - Diagnóstico completo")
                print("   🔧 await manager.force_synchronization() - Sincronización forzada")
                print("   🧹 await manager._auto_cleanup_if_desynchronized() - Limpieza automática")

                # Enviar a Discord si está configurado (SIN diagnóstico de sincronización)
                if hasattr(self, 'discord_notifier'):
                    # Crear reporte para Discord sin el diagnóstico de sincronización
                    discord_report = full_report.replace(sync_diagnosis, "")
                    await self._send_tcn_discord_notification(discord_report)
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

            # ✅ USAR EL ENSEMBLE PREDICTOR YA CONFIGURADO
            if not hasattr(self, 'tcn_predictor') or self.tcn_predictor is None:
                models_section += f"❌ **Predictor TCN ENSEMBLE no inicializado**\n"
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

            # ✅ NUEVO: Analizar contexto de mercado para Discord
            try:
                market_context = await self._analyze_market_context(current_prices)
                regime = market_context['regime']
                regime_confidence = market_context['confidence']
                market_score = market_context['score']
                volatility = market_context['volatility_level']

                # Emoji según régimen
                regime_emoji = {
                    'BULLISH': '🟢',
                    'BEARISH': '🔴',
                    'NEUTRAL': '🟡'
                }.get(regime, '⚪')

                # Emoji según volatilidad
                vol_emoji = {
                    'HIGH': '⚡',
                    'MEDIUM': '📊',
                    'LOW': '😴'
                }.get(volatility, '📊')

                # Obtener detalles del sistema robusto para Discord
                components = market_context.get('components', {})
                regime_system = components.get('regime_system', 'UNKNOWN')
                pairs_analyzed = components.get('pairs_analyzed', [])

                models_section += f"""
🌍 **RÉGIMEN DE MERCADO** ({regime_system})
{regime_emoji} **{regime}** (Conf: {regime_confidence:.1%}) {vol_emoji} Vol: {volatility}
📊 Score: {market_score:+.3f} | 🔗 Correlación: {market_context.get('correlation_strength', 0):.2f}
🔍 Pares: {len(pairs_analyzed)} analizados | 🎯 Consenso requerido: >60%

"""
            except Exception as e:
                models_section += f"⚠️ **Contexto de mercado**: Error al analizar ({str(e)[:30]}...)\n\n"

            # ✅ GENERAR PREDICCIONES ENSEMBLE PARA REPORTE
            try:
                print("🔍 Generando predicciones ENSEMBLE para reporte Discord...")
                all_predictions = await self.tcn_predictor.predict_all_symbols_v3()

                if not all_predictions:
                    models_section += "❌ **No se pudieron generar predicciones ensemble**\n"
                    return models_section

            except Exception as e:
                models_section += f"❌ **Error generando predicciones**: {str(e)[:50]}...\n"
                return models_section

            # Procesar predicciones para cada símbolo
            for symbol in self.symbols:
                try:
                    if symbol not in current_prices:
                        models_section += f"❌ **{symbol}**: Sin precio disponible\n"
                        continue

                    # Obtener predicción ensemble
                    prediction = all_predictions.get(symbol)
                    if prediction:
                        signal = prediction['ensemble_signal']
                        confidence = prediction['ensemble_confidence']
                        probabilities = prediction['ensemble_probabilities']

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
                    quantity=pos.quantity,  # ✅ CORREGIDO: usar 'quantity' en lugar de 'size'
                    entry_price=pos.entry_price,
                    current_price=pos.current_price,
                    value_usd=pos.market_value,  # ✅ CORREGIDO: usar 'market_value' en lugar de 'value_usd'
                    percentage=(pos.market_value / snapshot.total_balance_usd * 1) if snapshot.total_balance_usd > 0 else 0,
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
            # Verificar que start_time esté inicializado
            if not self.start_time:
                print("⚠️ Start time no inicializado")
                return

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
                                print(f"      #{i}: {pos.quantity:.6f} @ ${pos.entry_price:,.2f} ({pos_pnl_sign}{pos.unrealized_pnl_percent:.1f}%) {duration_str}")

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

            # ✅ NUEVO: Mostrar estado de cooldowns post-cierre
            print("=" * 80)
            self._display_cooldown_status()
            print("=" * 80)

            # Mostrar métricas
            print(f"📊 MÉTRICAS: API calls: {self.metrics.get('api_calls_count', 0)} | Errores: {self.metrics.get('error_count', 0)} | Reportes TCN: {self.metrics.get('tcn_reports_sent', 0)}")

            print("🔥" * 80)

        except Exception as e:
            print(f"❌ Error en display: {e}")

    def _display_cooldown_status(self):
        """⏰ Mostrar estado actual de cooldowns post-cierre"""
        try:
            if not self.recent_closes:
                print("⏰ COOLDOWNS: Ningún cooldown activo")
                return

            from datetime import datetime
            
            active_cooldowns = []
            expired_cooldowns = []
            
            current_time = datetime.now()
            
            for symbol, close_info in self.recent_closes.items():
                close_time = close_info['last_close_time']
                close_type = close_info['close_type']
                pnl_percent = close_info.get('pnl_percent', 0.0)
                
                # Obtener cooldown específico
                cooldown_minutes = self.post_close_cooldown.get(symbol, {}).get(close_type, 5)
                elapsed_minutes = (current_time - close_time).total_seconds() / 60
                
                if elapsed_minutes < cooldown_minutes:
                    remaining_minutes = cooldown_minutes - elapsed_minutes
                    if remaining_minutes >= 1:
                        remaining_time = f"{remaining_minutes:.0f}m"
                    else:
                        remaining_seconds = remaining_minutes * 60
                        remaining_time = f"{remaining_seconds:.0f}s"
                    
                    pnl_emoji = "📈" if pnl_percent > 0 else "📉"
                    close_type_emoji = "🛑" if close_type == "trailing_stop_close" else ("💰" if close_type == "profit_close" else "🔴")
                    
                    active_cooldowns.append(f"   {close_type_emoji} {symbol}: {remaining_time} ({close_type.replace('_', ' ').title()}) {pnl_emoji}{pnl_percent:.1f}%")
                else:
                    expired_cooldowns.append(symbol)
            
            if active_cooldowns:
                print("⏰ COOLDOWNS ACTIVOS:")
                for cooldown in active_cooldowns:
                    print(cooldown)
            else:
                print("⏰ COOLDOWNS: Todos expirados - Trading libre")
                
        except Exception as e:
            print(f"❌ Error mostrando cooldowns: {e}")

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

            # El método send_system_notification devuelve un booleano, no un objeto con status_code
            if result:
                print("✅ Discord: Reporte TCN enviado correctamente")
            else:
                print("⚠️ Discord: Error enviando reporte TCN")

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
                if self.database:
                    await self.database.log_event('ERROR', 'MARKET_DATA', f'Error precio {symbol}: {e}', symbol)

        self.last_check_time = datetime.now()
        self.metrics['api_calls_count'] += len(self.symbols)

        return prices

    async def _generate_tcn_signals(self, prices: Dict[str, float]) -> Dict:
        """
        🧠 Genera señales de trading (BUY/SELL) basadas en la confianza del Global Ensemble Predictor.
        ---
        CORREGIDO: Asegura que tanto las señales BUY como SELL se añadan a la cola de procesamiento.
        Versión técnica que combina análisis de 1m, 3m y 5m timeframes con convergencia inteligente.
        """
        signals = {}

        # Inicializar last_signals si no existe
        if not hasattr(self, 'last_signals'):
            self.last_signals = {}

        # ✅ VERIFICAR que Global Ensemble Predictor está inicializado (debe estar desde constructor)
        if not hasattr(self, 'tcn_predictor') or self.tcn_predictor is None:
            print("❌ ERROR CRÍTICO: Global Ensemble Predictor no está inicializado")
            print("🚨 SISTEMA REQUIERE ENSEMBLE TÉCNICO - REINTENTANDO INICIALIZACIÓN")
            self._initialize_tcn_predictor()

        # ✅ NUEVO: Analizar contexto de mercado como capa de seguridad
        try:
            market_context = await self._analyze_market_context(prices)
            print(f"🌍 CONTEXTO DE MERCADO: {market_context['regime']} (Score: {market_context['score']:.2f}, Confianza: {market_context['confidence']:.1%})")
        except Exception as e:
            print(f"⚠️ Error analizando contexto de mercado: {e}")
            # Usar contexto neutral por defecto
            market_context = {
                'regime': 'NEUTRAL',
                'score': 0.0,
                'confidence': 0.0,
                'market_fear_factor': 0.5,
                'trend_strength': 0.0,
                'volatility_level': 'MEDIUM'
            }

        # ✅ CENTRALIZADO: Umbral de confianza desde .env con ajustes por régimen
        base_threshold = float(os.getenv('MIN_CONFIDENCE_THRESHOLD', '0.65'))  # ✅ CORREGIDO: SIN multiplicar por 100

        # Ajustar umbral según régimen de mercado
        if market_context['regime'] == 'BEARISH':
            threshold = base_threshold * 1.1  # ✅ BEARISH: +10% sobre base
            print(f"🎯 UMBRAL BEARISH: {threshold * 100:.1f}% (base: {base_threshold * 100:.1f}%) - Mercado BEARISH {market_context['confidence']:.1%}")
        elif market_context['regime'] == 'BULLISH' and market_context['confidence'] > 0.9:
            threshold = base_threshold * 0.9  # 90% del umbral base para mercado muy bullish
            print(f"🎯 UMBRAL ADAPTATIVO: {threshold * 100:.1f}% (base: {base_threshold * 100:.1f}%) - Mercado BULLISH {market_context['confidence']:.1%}")
        elif market_context['regime'] == 'BULLISH' and market_context['confidence'] > 0.7:
            threshold = base_threshold * 0.95  # 95% del umbral base para mercado bullish
            print(f"🎯 UMBRAL ADAPTATIVO: {threshold * 100:.1f}% (base: {base_threshold * 100:.1f}%) - Mercado BULLISH {market_context['confidence']:.1%}")
        else:
            threshold = base_threshold
            print(f"🎯 UMBRAL ESTÁNDAR: {threshold * 100:.1f}% - Mercado {market_context['regime']}")

        # ✅ GLOBAL ENSEMBLE PREDICTOR: Usar predict_all_symbols_v3 para todos los símbolos
        try:
            print(f"🔍 Generando predicciones GLOBAL ENSEMBLE para todos los símbolos...")
            all_predictions = await self.tcn_predictor.predict_all_symbols_v3()

            if not all_predictions:
                print("❌ No se pudieron generar predicciones ensemble")
                return signals

            print(f"✅ Predicciones ensemble generadas para {len(all_predictions)} símbolos")

        except Exception as e:
            print(f"❌ Error generando predicciones ensemble: {e}")
            return signals

        # ✅ PROCESAR PREDICCIONES ENSEMBLE: Una por una
        for symbol in self.symbols:
            current_price = prices.get(symbol)
            if not current_price:
                continue

            try:
                print(f"🔍 Procesando {symbol} con predicción GLOBAL ENSEMBLE...")

                # Obtener predicción global ensemble
                prediction = all_predictions.get(symbol)
                if not prediction:
                    print(f"  ❌ No se pudo obtener predicción global ensemble para {symbol}")
                    continue

                signal = prediction['ensemble_signal']
                confidence_level = prediction['ensemble_confidence'] * 100  # Convertir a porcentaje

                # ✅ NUEVO: Aplicar filtros de estabilidad y cooldown
                filtered_signal, filter_reason = self._apply_signal_stability_filter(
                    symbol, signal, confidence_level, current_price, market_context
                )

                if filtered_signal != signal:
                    print(f"🛡️ FILTRO DE ESTABILIDAD aplicado en {symbol}: {signal} → {filtered_signal} ({filter_reason})")
                    signal = filtered_signal
                    # Si la señal fue filtrada a HOLD, reducir confianza
                    if signal == 'HOLD':
                        confidence_level = min(confidence_level, 65.0)

                # ✅ NUEVO: Aplicar filtro de contexto de mercado
                try:
                    filtered_signal, context_reason = self._apply_market_context_filter(
                        signal, confidence_level, market_context, symbol
                    )

                    if filtered_signal != signal:
                        print(f"🛡️ FILTRO DE CONTEXTO aplicado en {symbol}: {signal} → {filtered_signal} ({context_reason})")
                        signal = filtered_signal
                except Exception as e:
                    print(f"⚠️ Error en filtro de contexto para {symbol}: {e}")
                    # Continuar con la señal original si hay error

                # ✅ CORREGIDO: Procesar señales válidas independientemente de si han cambiado
                # Actualizar registro de última señal
                self.last_signals[symbol] = signal
                print(f"💡 Señal TCN para {symbol}: {signal} (Confianza: {confidence_level:.2f}%) (Umbral: {threshold * 100:.1f}%)")

                # ✅ CENTRALIZADO: SELL también requiere confianza mínima desde .env
                # Aplicar umbral de confianza tanto para BUY como para SELL
                if market_context['regime'] == 'BEARISH':
                    sell_threshold = base_threshold * 1.1 + 0.05  # ✅ BEARISH: +10% base + 5% extra
                elif market_context['regime'] == 'BULLISH' and market_context['confidence'] > 0.9:
                    sell_threshold = base_threshold * 0.95  # 95% del umbral base para mercado muy bullish
                    print(f"🎯 UMBRAL SELL ADAPTATIVO: {sell_threshold * 100:.1f}% (base: {base_threshold * 100:.1f}%) - Mercado BULLISH {market_context['confidence']:.1%}")
                elif market_context['regime'] == 'BULLISH' and market_context['confidence'] > 0.7:
                    sell_threshold = base_threshold  # Umbral base para mercado bullish
                    print(f"🎯 UMBRAL SELL ADAPTATIVO: {sell_threshold * 100:.1f}% (base: {base_threshold * 100:.1f}%) - Mercado BULLISH {market_context['confidence']:.1%}")
                else:
                    sell_threshold = base_threshold + 0.05  # +5% sobre base para SELL

                if (signal == 'BUY' and confidence_level >= threshold * 100) or (signal == 'SELL' and confidence_level >= sell_threshold * 100):
                    log_emoji = "📈" if signal == "BUY" else "📉"
                    log_action = "COMPRA" if signal == "BUY" else "VENTA"
                    print(f"{log_emoji} Oportunidad de {log_action} detectada para {symbol}. Preparando para posible operación.")

                    # Verificaciones adicionales específicas para BUY
                    if signal == 'BUY':
                        # ✅ CORREGIDO: Verificar límite de 3 posiciones por símbolo
                        existing_positions = self._get_positions_for_symbol(symbol)
                        if len(existing_positions) >= 3:
                            print(f"  ⏸️ Señal BUY ignorada - Máximo 3 posiciones alcanzado para {symbol} (actual: {len(existing_positions)})")
                            continue

                        # Verificar balance suficiente
                        min_position_value = 11.0  # Valor por defecto si risk_manager no está disponible
                        if self.risk_manager and hasattr(self.risk_manager, 'limits'):
                            min_position_value = self.risk_manager.limits.min_position_value_usdt

                        if self.current_balance < min_position_value:
                            print(f"  💰 Señal BUY generada (solo análisis) - Balance insuficiente para trade")
                            continue

                    elif signal == 'SELL':
                        # Verificar si tenemos posición para vender
                        existing_positions = self._get_positions_for_symbol(symbol)
                        if len(existing_positions) == 0:
                            print(f"  ⏸️ Señal SELL ignorada - No hay posición que vender en {symbol}")
                            continue

                        # ✅ NUEVO: Información sobre umbral SELL aplicado
                        print(f"  📊 Señal SELL con confianza {confidence_level:.1f}% (umbral: {sell_threshold:.1f}%)")
                        print(f"  📋 {len(existing_positions)} posición(es) existente(s) serán evaluadas para cierre")

                    # ✅ SEÑAL VÁLIDA - Este bloque ahora se ejecuta para ambas señales
                    signals[symbol] = {
                        'signal': signal,
                        'price': current_price,
                        'confidence': confidence_level,
                        'timestamp': datetime.utcnow(),
                        'current_price': current_price,
                        'reason': 'TCN_MODEL_PREDICTION',
                        'available_usdt': self.current_balance,
                        'probabilities': prediction.get('probabilities', {}),
                        'balance_sufficient': self.current_balance >= (self.risk_manager.limits.min_position_value_usdt if self.risk_manager and self.risk_manager.limits else 11.0),
                        # ✅ NUEVO: Información del contexto de mercado
                        'market_context': market_context,
                        'context_filter_applied': filtered_signal != prediction['ensemble_signal']
                    }
                    print(f"  ✅ SEÑAL AÑADIDA A LA COLA: {symbol} {signal} ({confidence_level:.1f}%)")

            except Exception as e:
                print(f"  ❌ Error procesando {symbol}: {e}")
                continue

        # ✅ NUEVO: Implementar priorización de BTC en señales simultáneas
        if signals:
            print(f"🎯 Total señales TCN generadas: {len(signals)}")

            # Aplicar priorización: BTC > ETH > BNB > XRP
            prioritized_signals = self._apply_signal_prioritization(signals)

            print(f"⚖️ Señales después de priorización: {len(prioritized_signals)}")
            return prioritized_signals
        else:
            print("📊 No se generaron señales TCN válidas en este ciclo")
            # ✅ INFORMACIÓN: Mostrar por qué no se generaron señales válidas
            print(f"💡 Umbrales aplicados - BUY: {threshold * 100:.1f}% | SELL: {sell_threshold * 100:.1f}%")
            if market_context['regime'] == 'BULLISH' and market_context['confidence'] > 0.7:
                print(f"🎯 Umbrales relajados por mercado BULLISH ({market_context['confidence']:.1%})")
            return signals

    def _apply_signal_prioritization(self, signals: Dict) -> Dict:
        """
        🏆 SISTEMA DE PRIORIZACIÓN DE SEÑALES CON PRIVILEGIO PARA BTC
        ---
        En caso de señales simultáneas, prioriza BTC sobre todas las demás
        Orden de prioridad: BTC > ETH > BNB > XRP

        Lógica:
        - Si hay señal BUY de BTC, se ejecuta inmediatamente
        - Otras señales BUY se evalúan según disponibilidad de balance
        - Señales SELL no se priorizan (se ejecutan todas)
        """

        if not signals:
            return signals

        try:
            print(f"🏆 Aplicando priorización de señales...")

            # Orden de prioridad definido
            PRIORITY_ORDER = ['ETHUSDT', 'XRPUSDT', 'BTCUSDT', 'DOTUSDT', 'ADAUSDT', 'BNBUSDT']

            # Separar señales BUY y SELL
            buy_signals = {symbol: data for symbol, data in signals.items() if data['signal'] == 'BUY'}
            sell_signals = {symbol: data for symbol, data in signals.items() if data['signal'] == 'SELL'}

            print(f"   📊 Señales BUY: {list(buy_signals.keys())}")
            print(f"   📊 Señales SELL: {list(sell_signals.keys())}")

            # ✅ SELL: Todas las señales SELL se procesan (no hay priorización)
            prioritized_signals = sell_signals.copy()

            # ✅ BUY: Aplicar priorización estricta
            if buy_signals:
                # Ordenar según prioridad
                sorted_buy_symbols = []
                for priority_symbol in PRIORITY_ORDER:
                    if priority_symbol in buy_signals:
                        sorted_buy_symbols.append(priority_symbol)

                # Agregar cualquier símbolo no contemplado al final
                for symbol in buy_signals:
                    if symbol not in sorted_buy_symbols:
                        sorted_buy_symbols.append(symbol)

                print(f"   🏆 Orden de prioridad BUY: {sorted_buy_symbols}")

                # ✅ PRIVILEGIO ABSOLUTO PARA BTC
                if 'XRPUSDT' in buy_signals:
                    btc_signal = buy_signals['XRPUSDT']
                    prioritized_signals['XRPUSDT'] = btc_signal
                    print(f"   👑 XRPUSDT PRIVILEGIADO: Señal BUY de XRPUSDT tiene prioridad absoluta")

                    # Verificar si hay balance suficiente para otros después de BTC
                    xrp_confidence = btc_signal['confidence']
                    estimated_xrp_size = min(18.0, max(10.0, (xrp_confidence/100) * 20.0))
                    estimated_xrp_value = (self.current_balance * estimated_xrp_size / 100)
                    remaining_balance = self.current_balance - estimated_xrp_value

                    print(f"   💰 Balance estimado después de XRPUSDT: ${remaining_balance:.2f}")

                    # Solo agregar otras señales BUY si queda balance significativo
                    min_position_value = 11.0
                    if remaining_balance >= min_position_value * 1.5:  # Buffer del 50%
                        # Agregar siguientes en orden de prioridad
                        for symbol in sorted_buy_symbols[1:]:  # Saltar XRPUSDT ya agregado
                            if symbol in buy_signals:
                                prioritized_signals[symbol] = buy_signals[symbol]
                                print(f"   ✅ Agregado {symbol} (prioridad {PRIORITY_ORDER.index(symbol) + 1})")
                    else:
                        print(f"   ⏸️ Otras señales BUY pausadas - Balance insuficiente después de BTC")

                else:
                    # No hay XRPUSDT, procesar en orden de prioridad normal
                    for symbol in sorted_buy_symbols:
                        prioritized_signals[symbol] = buy_signals[symbol]
                        print(f"   ✅ Agregado {symbol} (prioridad {PRIORITY_ORDER.index(symbol) + 1 if symbol in PRIORITY_ORDER else 'N/A'})")

            # ✅ ACTUALIZAR pending_signals para el sistema de diversificación
            self.pending_signals = {symbol: data for symbol, data in prioritized_signals.items() if data['signal'] == 'BUY'}

            # Mostrar resultado final
            if len(prioritized_signals) != len(signals):
                removed_signals = set(signals.keys()) - set(prioritized_signals.keys())
                print(f"   ⏸️ Señales pausadas por priorización: {list(removed_signals)}")

            print(f"   ✅ Señales finales priorizadas: {list(prioritized_signals.keys())}")

            return prioritized_signals

        except Exception as e:
            print(f"❌ Error en priorización de señales: {e}")
            # En caso de error, devolver señales originales
            return signals

    async def _analyze_market_context(self, prices: Dict[str, float]) -> Dict:
        """
        🌍 ANÁLISIS ROBUSTO DE CONTEXTO DE MERCADO - VERSIÓN RELAJADA

        Sistema mejorado que considera múltiples pares y factores de mercado
        para determinar el régimen actual de forma precisa.

        RELAJADO: Sistema menos restrictivo con relajación temporal automática

        Returns:
            Dict con régimen, score, confianza y factores de riesgo
        """
        try:
            print("🔍 Analizando contexto de mercado robusto (versión relajada)...")

            # Obtener datos históricos para análisis robusto
            market_data = {}
            for symbol in ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'DOTUSDT']:
                try:
                    # Obtener datos de 5 minutos para análisis detallado
                    url = f"https://api.binance.com/api/v3/klines"
                    params = {
                        'symbol': symbol,
                        'interval': '15m',
                        'limit': 100      # 100 * 15m = ~16.7 horas de datos
                    }

                    async with aiohttp.ClientSession() as session:
                        async with session.get(url, params=params) as response:
                            if response.status == 200:
                                klines = await response.json()
                                market_data[symbol] = [float(k[4]) for k in klines]  # Precios de cierre
                except Exception as e:
                    print(f"  ⚠️ Error obteniendo datos para {symbol}: {e}")
                    market_data[symbol] = [prices.get(symbol, 0)] * 200  # Fallback

            # Usar el sistema robusto de detección de régimen
            regime, confidence = await self._detect_market_regime_robust(market_data)

            # Calcular métricas adicionales para compatibilidad
            fear_factor = 0.5  # Neutral por defecto
            correlation_strength = 0.5

            try:
                # Calcular volatilidad del mercado
                if 'BTCUSDT' in market_data and len(market_data['BTCUSDT']) > 24:
                    import numpy as np
                    btc_returns = np.diff(market_data['BTCUSDT']) / market_data['BTCUSDT'][:-1]
                    btc_volatility = np.std(btc_returns)
                    fear_factor = min(1, max(0, btc_volatility * 1000))  # Escalar a [0,1]

                # Calcular correlación entre pares principales
                if len(market_data) >= 2:
                    btc_returns = np.diff(market_data['BTCUSDT']) / market_data['BTCUSDT'][:-1]
                    eth_returns = np.diff(market_data['ETHUSDT']) / market_data['ETHUSDT'][:-1]

                    if len(btc_returns) > 10 and len(eth_returns) > 10:
                        min_len = min(len(btc_returns), len(eth_returns))
                        corr = np.corrcoef(btc_returns[:min_len], eth_returns[:min_len])[0, 1]
                        correlation_strength = max(0, min(1, (corr + 1) / 2))  # Normalizar a [0,1]
            except Exception as e:
                print(f"  ⚠️ Error calculando métricas adicionales: {e}")

            # ✅ NUEVO: Sistema de relajación temporal automática
            current_time = time.time()
            regime_start_time = getattr(self, '_regime_start_time', current_time)
            regime_duration_hours = (current_time - regime_start_time) / 3600

            # Si el régimen cambió, actualizar el tiempo de inicio
            if getattr(self, '_last_regime', None) != regime:
                self._regime_start_time = current_time
                self._last_regime = regime
                regime_duration_hours = 0

            # ✅ RELAJACIÓN TEMPORAL: Reducir restricciones con el tiempo
            time_relaxation_factor = 1.0
            if regime_duration_hours > 6:  # Después de 6 horas
                time_relaxation_factor = 0.9  # Reducir 10%
            if regime_duration_hours > 12:  # Después de 12 horas
                time_relaxation_factor = 0.8  # Reducir 20%
            if regime_duration_hours > 24:  # Después de 24 horas
                time_relaxation_factor = 0.7  # Reducir 30%
            if regime_duration_hours > 48:  # Después de 48 horas
                time_relaxation_factor = 0.6  # Reducir 40%

            # Mapear régimen a score para compatibilidad
            if regime == 'BULLISH':
                score = 0.3 + (confidence - 0.5) * 0.4  # Score positivo
            elif regime == 'BEARISH':
                score = -0.3 - (confidence - 0.5) * 0.4  # Score negativo
            else:  # NEUTRAL
                score = 0.0

            # Nivel de volatilidad
            if fear_factor > 0.7:
                volatility_level = 'HIGH'
            elif fear_factor < 0.3:
                volatility_level = 'LOW'
            else:
                volatility_level = 'MEDIUM'

            context = {
                'regime': regime,
                'score': score,
                'confidence': confidence,
                'market_fear_factor': fear_factor,
                'trend_strength': abs(score),
                'volatility_level': volatility_level,
                'correlation_strength': correlation_strength,
                'altcoin_strength': 0.0,  # Calculado internamente por el sistema robusto
                'btc_trend_score': score,
                'regime_duration_hours': regime_duration_hours,  # ✅ NUEVO
                'time_relaxation_factor': time_relaxation_factor,  # ✅ NUEVO
                'btc_leading_down': regime == 'BEARISH' and confidence > 0.7,  # ✅ NUEVO
                'components': {
                    'regime_system': 'ROBUST_MULTI_PAIR_RELAXED',
                    'pairs_analyzed': list(market_data.keys()),
                    'confidence': confidence,
                    'volatility': fear_factor,
                    'time_relaxation': time_relaxation_factor
                }
            }

            print(f"  🎯 RÉGIMEN DETECTADO: {regime} (Confianza: {confidence:.2f})")
            print(f"  📊 Score compuesto: {score:.3f}")
            print(f"  😨 Factor miedo: {fear_factor:.3f}")
            print(f"  🔗 Correlación: {correlation_strength:.3f}")
            print(f"  ⚡ Volatilidad: {volatility_level}")
            print(f"  ⏰ Duración régimen: {regime_duration_hours:.1f}h")
            print(f"  🔓 Factor relajación temporal: {time_relaxation_factor:.1f}x")

            return context

        except Exception as e:
            print(f"❌ Error en análisis robusto de contexto: {e}")
            # Contexto neutral por defecto
            return {
                'regime': 'NEUTRAL',
                'score': 0.0,
                'confidence': 0.5,
                'market_fear_factor': 0.5,
                'trend_strength': 0.0,
                'volatility_level': 'MEDIUM',
                'correlation_strength': 0.5,
                'altcoin_strength': 0.0,
                'btc_trend_score': 0.0,
                'regime_duration_hours': 0,
                'time_relaxation_factor': 1.0,
                'btc_leading_down': False,
                'components': {'error': str(e)}
            }

    def _apply_market_context_filter(self, signal: str, confidence: float, market_context: Dict, symbol: str) -> tuple:
        """
        🛡️ FILTRO DE CONTEXTO DE MERCADO RELAJADO - VERSIÓN OPTIMIZADA
        ---
        Sistema relajado para permitir más oportunidades de trading:
        1. Umbrales de confianza reducidos significativamente
        2. Sistema de relajación temporal automática
        3. Filtros menos restrictivos en mercados neutros
        4. Favorecer trading en lugar de restringirlo

        Returns:
            tuple: (señal_filtrada, razón_del_filtro)
        """

        try:
            # ✅ CENTRALIZADO: Definir base_threshold al inicio para uso global
            base_threshold = float(os.getenv('MIN_CONFIDENCE_THRESHOLD', '0.65'))  # ✅ CORREGIDO: SIN multiplicar por 100

            # Extraer información del contexto
            regime = market_context.get('regime', 'NEUTRAL')
            market_confidence = market_context.get('confidence', 0.0)
            market_score = market_context.get('score', 0.0)
            fear_factor = market_context.get('market_fear_factor', 0.5)
            volatility = market_context.get('volatility_level', 'MEDIUM')

            # ✅ NUEVO: Información de duración del régimen
            regime_duration_hours = market_context.get('regime_duration_hours', 0)
            btc_leading_down = market_context.get('btc_leading_down', False)

            # Por defecto, no filtrar
            filter_reason = f"Sin filtro aplicado - {regime} con confianza {market_confidence:.1%}"

            # 🔴 FILTROS BEARISH RELAJADOS - Sistema gradual por intensidad (MUCHO MÁS PERMISIVO)
            if regime == 'BEARISH' and market_confidence > 0.8:  # Solo aplicar si confianza muy alta

                # ✅ CENTRALIZADO: AJUSTE BEARISH
                bearish_threshold = base_threshold * 1.1  # ✅ BEARISH: +10% sobre base

                if market_confidence > 0.9:  # BEARISH MUY FUERTE
                    buy_thresholds = {
                        'BTCUSDT': bearish_threshold,   # ✅ BEARISH: +10% sobre base
                        'ETHUSDT': bearish_threshold,   # ✅ BEARISH: +10% sobre base
                        'BNBUSDT': bearish_threshold,   # ✅ BEARISH: +10% sobre base
                        'XRPUSDT': bearish_threshold,   # ✅ BEARISH: +10% sobre base
                        'DOTUSDT': bearish_threshold    # ✅ BEARISH: +10% sobre base
                    }
                    sell_threshold = bearish_threshold + 0.05   # ✅ BEARISH: +10% base + 5%
                    intensity_level = "MUY_FUERTE"

                elif market_confidence > 0.85:  # BEARISH FUERTE
                    buy_thresholds = {
                        'BTCUSDT': bearish_threshold,   # ✅ BEARISH: +10% sobre base
                        'ETHUSDT': bearish_threshold,   # ✅ BEARISH: +10% sobre base
                        'BNBUSDT': bearish_threshold,   # ✅ BEARISH: +10% sobre base
                        'XRPUSDT': bearish_threshold,   # ✅ BEARISH: +10% sobre base
                        'DOTUSDT': bearish_threshold    # ✅ BEARISH: +10% sobre base
                    }
                    sell_threshold = bearish_threshold + 0.05   # ✅ BEARISH: +10% base + 5%
                    intensity_level = "FUERTE"

                else:  # BEARISH MODERADO
                    buy_thresholds = {
                        'BTCUSDT': bearish_threshold,   # ✅ BEARISH: +10% sobre base
                        'ETHUSDT': bearish_threshold,   # ✅ BEARISH: +10% sobre base
                        'BNBUSDT': bearish_threshold,   # ✅ BEARISH: +10% sobre base
                        'XRPUSDT': bearish_threshold,   # ✅ BEARISH: +10% sobre base
                        'DOTUSDT': bearish_threshold    # ✅ BEARISH: +10% sobre base
                    }
                    sell_threshold = bearish_threshold + 0.05   # ✅ BEARISH: +10% base + 5%
                    intensity_level = "MODERADO"

                # ✅ NUEVO: Factor de correlación con BTC (RELAJADO)
                if symbol != 'BTCUSDT' and btc_leading_down:
                    correlation_penalty = 0.03  # ✅ RELAJADO: De 5% a 3%
                    for key in buy_thresholds:
                        if key != 'BTCUSDT':
                            buy_thresholds[key] += correlation_penalty
                    filter_reason += f" + Penalidad correlación BTC ({correlation_penalty*100:.0f}%)"

                # ✅ NUEVO: Sistema de relajación temporal automática
                time_relaxation_factor = market_context.get('time_relaxation_factor', 1.0)
                if time_relaxation_factor < 1.0:
                    for key in buy_thresholds:
                        buy_thresholds[key] = buy_thresholds[key] * time_relaxation_factor
                    sell_threshold = sell_threshold * time_relaxation_factor
                    filter_reason += f" + Relax temporal automático ({time_relaxation_factor:.1f}x)"

                if signal == 'BUY':
                    required_confidence = buy_thresholds.get(symbol, base_threshold)  # ✅ CENTRALIZADO: Usar .env

                    if confidence >= required_confidence * 100:
                        filter_reason = f"{symbol.replace('USDT', '')} permitido en BEARISH {intensity_level} por alta confianza ({confidence:.1f}% > {required_confidence * 100:.1f}%)"
                    else:
                        signal = 'HOLD'
                        filter_reason = f"Mercado BEARISH {intensity_level} (score: {market_score:.2f}) - {symbol} BUY requiere >{required_confidence * 100:.1f}% confianza (actual: {confidence:.1f}%)"

                elif signal == 'SELL':
                    # ✅ CENTRALIZADO: SELL en bearish requiere confianza razonable
                    if confidence < sell_threshold * 100:
                        signal = 'HOLD'
                        filter_reason = f"SELL en BEARISH {intensity_level} requiere >{sell_threshold * 100:.1f}% confianza (actual: {confidence:.1f}%)"
                    else:
                        filter_reason = f"SELL favorecido en mercado BEARISH {intensity_level} con confianza {confidence:.1f}%"

            # 🟢 FILTROS BULLISH - Aprovechar momentum alcista (CENTRALIZADOS)
            elif regime == 'BULLISH' and market_confidence > 0.7:
                if signal == 'BUY':
                    # ✅ CENTRALIZADO: En bullish, usar umbral base del .env
                    min_buy_confidence = base_threshold * 0.9  # 90% del umbral base para bullish

                    if confidence < min_buy_confidence * 100:
                        signal = 'HOLD'
                        filter_reason = f"BUY en BULLISH requiere >{min_buy_confidence * 100:.1f}% confianza"
                    else:
                        filter_reason = f"BUY favorecido en mercado BULLISH (conf: {market_confidence:.1%})"

                elif signal == 'SELL':
                    # ✅ CENTRALIZADO: En bullish, usar umbral base del .env
                    min_sell_confidence = base_threshold + 0.05  # +5% sobre base para SELL

                    if confidence < min_sell_confidence * 100:
                        signal = 'HOLD'
                        filter_reason = f"Mercado BULLISH (score: {market_score:.2f}) - SELL requiere >{min_sell_confidence * 100:.1f}% confianza (actual: {confidence:.1f}%)"
                    else:
                        filter_reason = f"SELL permitido en BULLISH para tomar ganancias ({confidence:.1f}%)"

            # 🟡 FILTROS DE VOLATILIDAD - Ajustar según volatilidad del mercado (CENTRALIZADOS)
            # ✅ CORREGIDO: Umbrales UNIFORMES con delta ADICIONAL por volatilidad
            # 🎯 LÓGICA: En alta volatilidad, aplicar delta ADICIONAL sobre el umbral base
            # 📊 Ejemplo: MIN_CONFIDENCE_THRESHOLD=0.70 (70%) + 0.02 (2%) = 0.72 (72%)
            # 📊 ANTES: Umbral fijo 63% → INCORRECTO (menor que el base)
            # 📊 AHORA: 70% + 2% = 72% → CORRECTO (mayor que el base)
            if volatility == 'HIGH' and fear_factor > 0.8:
                # ✅ CENTRALIZADO: En mercado BULLISH, la volatilidad puede ser oportunidad
                if regime == 'BULLISH' and market_confidence > 0.9:
                    # 🎯 MODERADO: En mercado muy bullish, permitir algunas oportunidades más
                    volatility_thresholds = {
                        'BTCUSDT': base_threshold * 0.85,   # 🎯 MODERADO: 85% del umbral base (era 90%)
                        'ETHUSDT': base_threshold * 0.85,   # 🎯 MODERADO: 85% del umbral base (era 90%)
                        'BNBUSDT': base_threshold * 0.85,   # 🎯 MODERADO: 85% del umbral base (era 90%)
                        'XRPUSDT': base_threshold * 0.85,   # 🎯 MODERADO: 85% del umbral base (era 90%)
                        'DOTUSDT': base_threshold * 0.85,   # 🎯 MODERADO: 85% del umbral base (era 90%)
                        'ADAUSDT': base_threshold * 0.85    # 🎯 MODERADO: 85% del umbral base (era 90%)
                    }
                    vol_adjustment = "MODERADO_BULLISH"
                else:
                    # ✅ CORREGIDO: Umbrales UNIFORMES con delta ADICIONAL por alta volatilidad
                    # 🎯 OBJETIVO: Aplicar delta ADICIONAL sobre el umbral base (70% + 2% = 72%)
                    # 📊 ANTES: Umbral fijo 63% → INCORRECTO
                    # 📊 AHORA: base_threshold + 0.02 = 0.70 + 0.02 = 0.72 (72%) → CORRECTO
                    volatility_delta = 0.02  # 2% adicional por alta volatilidad
                    volatility_thresholds = {
                        'BTCUSDT': base_threshold + volatility_delta,   # ✅ CORRECTO: 70% + 2% = 72%
                        'ETHUSDT': base_threshold + volatility_delta,   # ✅ CORRECTO: 70% + 2% = 72%
                        'BNBUSDT': base_threshold + volatility_delta,   # ✅ CORRECTO: 70% + 2% = 72%
                        'XRPUSDT': base_threshold + volatility_delta,   # ✅ CORRECTO: 70% + 2% = 72%
                        'DOTUSDT': base_threshold + volatility_delta,   # ✅ CORRECTO: 70% + 2% = 72%
                        'ADAUSDT': base_threshold + volatility_delta    # ✅ CORRECTO: 70% + 2% = 72%
                    }
                    vol_adjustment = "ALTA_VOLATILIDAD"

                if signal == 'BUY':
                    required_vol_confidence = volatility_thresholds.get(symbol, base_threshold + 0.02)  # ✅ CORRECTO: 70% + 2% = 72%
                    if confidence < required_vol_confidence * 100:
                        signal = 'HOLD'
                        filter_reason = f"Alta volatilidad ({vol_adjustment}) - {symbol} BUY requiere >{required_vol_confidence * 100:.1f}% confianza"
                elif signal == 'SELL':
                    # ✅ CORREGIDO: Umbral UNIFORME con delta ADICIONAL para SELL en alta volatilidad
                    # 🎯 OBJETIVO: Aplicar delta ADICIONAL sobre el umbral base (70% + 2% = 72%)
                    # 📊 ANTES: Umbral fijo 63% → INCORRECTO
                    # 📊 AHORA: base_threshold + 0.02 = 0.70 + 0.02 = 0.72 (72%) → CORRECTO
                    min_sell_vol_conf = base_threshold + 0.02  # ✅ CORRECTO: 70% + 2% = 72%

                    if confidence < min_sell_vol_conf * 100:
                        signal = 'HOLD'
                        filter_reason = f"Alta volatilidad ({vol_adjustment}) - SELL requiere >{min_sell_vol_conf * 100:.1f}% confianza (actual: {confidence:.1f}%)"

            # 🔵 FILTROS ESPECÍFICOS POR ACTIVO (CENTRALIZADOS)
            if symbol == 'BTCUSDT':
                # ✅ CENTRALIZADO: BTC como líder - permitir señales más flexibles
                btc_trend = market_context.get('btc_trend_score', 0)
                if signal == 'BUY' and btc_trend < -0.4:  # Solo en tendencia MUY bajista
                    if confidence < base_threshold * 0.9 * 100:  # ✅ CENTRALIZADO: 90% del umbral base
                        signal = 'HOLD'
                        filter_reason = f"BTC en tendencia muy bajista fuerte (trend: {btc_trend:.2f}) - BUY requiere >{base_threshold * 0.9 * 100:.1f}% confianza"
                    else:
                        filter_reason = f"BTC BUY permitido pese a tendencia bajista por alta confianza ({confidence:.1f}%)"

            elif symbol in ['ETHUSDT', 'BNBUSDT', 'XRPUSDT']:
                # ✅ CENTRALIZADO: ALTCOINS - Umbrales menos restrictivos por underperformance
                altcoin_strength = market_context.get('altcoin_strength', 0)
                if signal == 'BUY' and altcoin_strength < -0.2:  # Altcoins underperforming
                    # ✅ CENTRALIZADO: Umbrales específicos para cada altcoin
                    altcoin_thresholds = {
                        'ETHUSDT': base_threshold * 0.9,   # ✅ CENTRALIZADO: 90% del umbral base
                        'BNBUSDT': base_threshold * 0.9,   # ✅ CENTRALIZADO: 90% del umbral base
                        'XRPUSDT': base_threshold * 0.9    # ✅ CENTRALIZADO: 90% del umbral base
                    }

                    required_alt_confidence = altcoin_thresholds.get(symbol, base_threshold * 0.9)

                    if confidence >= required_alt_confidence * 100:
                        filter_reason = f"{symbol.replace('USDT', '')} permitido por alta confianza ({confidence:.1f}% > {required_alt_confidence * 100}%) pese a underperformance"
                    else:
                        signal = 'HOLD'
                        filter_reason = f"Altcoins underperforming vs BTC - {symbol} BUY requiere >{required_alt_confidence * 100}% confianza"

            return signal, filter_reason

        except Exception as e:
            print(f"⚠️ Error en filtro de contexto para {symbol}: {e}")
            return signal, f"Error en filtro: {e}"

    def _apply_signal_stability_filter(self, symbol: str, signal: str, confidence: float, current_price: float, market_context: Optional[Dict] = None) -> tuple:
        """
        🛡️ FILTRO DE ESTABILIDAD DESACTIVADO - ENSEMBLE DE 3 MODELOS (1M, 3M, 5M)
        ---
        DESACTIVADO: El ensemble de 3 modelos (1M, 3M, 5M) proporciona estabilidad natural
        No se requiere filtro adicional de estabilidad ya que los modelos se complementan
        Solo se mantiene protección específica para ETH

        Returns:
            tuple: (señal_original, razón_del_filtro)
        """

        try:
            current_time = datetime.now()

            # Inicializar historial del símbolo si no existe
            if symbol not in self.signal_history:
                self.signal_history[symbol] = {
                    'last_signal': None,
                    'last_signal_time': None,
                    'signal_count': 0,
                    'consecutive_same_signal': 0
                }

            history = self.signal_history[symbol]
            cooldown_minutes = self.signal_cooldown.get(symbol, 10)

            # ✅ COOLDOWN GENERAL DESACTIVADO - ENSEMBLE DE 3 MODELOS
            # No se aplica cooldown general ya que el ensemble proporciona estabilidad natural
            # Solo se mantiene tracking para estadísticas

            # ✅ PROTECCIÓN ESPECÍFICA PARA ETH
            if symbol == 'ETHUSDT':
                eth_protection = self.eth_position_protection

                # Verificar si hay posiciones existentes de ETH
                existing_positions = self._get_positions_for_symbol(symbol)

                if existing_positions and signal == 'SELL':
                    # ✅ PROTECCIÓN: Tiempo mínimo de retención para posiciones ETH
                    for position in existing_positions:
                        # Obtener el tiempo de creación de la posición (approximation)
                        if hasattr(position, 'entry_time'):
                            position_age_minutes = (current_time - position.entry_time).total_seconds() / 60
                        else:
                            # Si no tenemos entry_time, usar última acción conocida
                            last_action_time = self.last_position_action.get(symbol, current_time - timedelta(minutes=eth_protection['min_hold_time_minutes']))
                            position_age_minutes = (current_time - last_action_time).total_seconds() / 60

                        if position_age_minutes < eth_protection['min_hold_time_minutes']:
                            # ✅ EXCEPCIÓN: Permitir SELL solo con confianza EXTREMA (>90%) y pérdida significativa
                            if confidence >= 90.0 and position.pnl_percent < -3.0:
                                filter_reason = f"ETH SELL permitido por confianza extrema ({confidence:.1f}%) y pérdida > 3%"
                                break
                            else:
                                return 'HOLD', f"ETH protegido: posición muy reciente ({position_age_minutes:.1f}min < {eth_protection['min_hold_time_minutes']}min)"

                # ✅ SISTEMA DE CONFIRMACIÓN PARA ETH
                # Requerir múltiples señales consecutivas iguales
                if history['last_signal'] == signal:
                    history['consecutive_same_signal'] += 1
                else:
                    history['consecutive_same_signal'] = 1

                required_confirmations = eth_protection['signal_confirmation_required']

                # Solo aplicar confirmación para señales de SELL en ETH
                if signal == 'SELL' and history['consecutive_same_signal'] < required_confirmations:
                    return 'HOLD', f"ETH SELL requiere {required_confirmations} confirmaciones consecutivas (actual: {history['consecutive_same_signal']})"

            # ✅ FILTRO DE ESTABILIDAD DESACTIVADO - ENSEMBLE DE 3 MODELOS (1M, 3M, 5M)
            # No se aplica filtro de confianza para cambios de señal
            # El ensemble de 1M, 3M y 5M proporciona estabilidad natural
            # DESACTIVADO: El ensemble de 3 modelos proporciona estabilidad natural

            # ✅ ACTUALIZAR HISTORIAL
            history['last_signal'] = signal
            history['last_signal_time'] = current_time
            history['signal_count'] += 1

            # Actualizar timestamp de última acción para el símbolo
            self.last_position_action[symbol] = current_time

            return signal, f"FILTRO DESACTIVADO - Ensemble 3M (1M,3M,5M) proporciona estabilidad natural"

        except Exception as e:
            print(f"⚠️ Error en filtro de estabilidad para {symbol}: {e}")
            return signal, f"Error en filtro estabilidad: {e}"

    async def _process_signal(self, symbol: str, signal_data: Dict):
        """⚡ Procesar una señal individual - CON DEBUG DETALLADO"""

        signal = signal_data['signal']
        confidence = signal_data['confidence']
        current_price = signal_data['current_price']
        balance_sufficient = signal_data.get('balance_sufficient', True)

        print(f"🔍 PROCESANDO SEÑAL: {symbol} {signal} ({confidence:.1f}%)")

        # ✅ CRÍTICO: Verificar cooldown post-cierre ANTES de procesar cualquier señal BUY
        if signal == 'BUY':
            print(f"🔍 VERIFICANDO COOLDOWN POST-CIERRE PARA: {symbol}")
            is_in_cooldown, remaining_time = self.is_in_post_close_cooldown(symbol)
            if is_in_cooldown:
                print(f"  🚫 SEÑAL BUY BLOQUEADA POR COOLDOWN POST-CIERRE: {symbol}")
                print(f"  ⏰ Tiempo restante: {remaining_time}")
                print(f"  🛑 RETORNANDO SIN PROCESAR SEÑAL")
                await self._send_discord_notification(
                    f"⏰ **COOLDOWN POST-CIERRE**\n"
                    f"📊 {symbol}: {signal}\n"
                    f"🚫 Razón: Cooldown activo\n"
                    f"⏱️ Tiempo restante: {remaining_time}\n"
                    f"🎯 Confianza: {confidence:.1f}%"
                )
                return
            else:
                print(f"  ✅ COOLDOWN CHECK PASADO: {symbol} puede procesar señal BUY")

        # Skip si es HOLD
        if signal == 'HOLD':
            print(f"  ⏸️ Señal HOLD ignorada para {symbol}")
            return

        # Verificar si el balance es suficiente para nuevas posiciones BUY
        min_position_value = 11.0  # Valor por defecto
        if self.risk_manager and hasattr(self.risk_manager, 'limits'):
            min_position_value = self.risk_manager.limits.min_position_value_usdt

        print(f"  💰 Balance check: ${self.current_balance:.2f} vs min ${min_position_value:.2f}")

        if signal == 'BUY' and not balance_sufficient:
            print(f"  ❌ Señal BUY {symbol} BLOQUEADA - Balance insuficiente")
            return

        # ✅ CORREGIDO: Verificar posiciones existentes para múltiples posiciones por símbolo
        existing_positions = self._get_positions_for_symbol(symbol)
        print(f"  📊 Posiciones existentes en {symbol}: {len(existing_positions)}/3")

        # Si es señal SELL, gestionar posiciones existentes
        if signal == 'SELL' and existing_positions:
            print(f"  🔄 Señal SELL - Gestionando {len(existing_positions)} posición(es) existente(s) para {symbol}")
            await self._manage_existing_position(symbol, signal_data)

        # Si es señal BUY, considerar nueva posición (independiente de posiciones existentes)
        elif signal == 'BUY':
            if len(existing_positions) >= 3:
                print(f"  ❌ Señal BUY BLOQUEADA - Máximo 3 posiciones alcanzado para {symbol} ({len(existing_positions)}/3)")
                return
            else:
                print(f"  📈 Señal BUY - Considerando nueva posición para {symbol} (será {len(existing_positions)+1}/3)")
                await self._consider_new_position(symbol, signal_data)

        # ✅ NUEVO: Limpiar señales pendientes después de procesar
        if symbol in self.pending_signals:
            del self.pending_signals[symbol]
            print(f"  🧹 Señal pendiente limpiada para {symbol}")

    async def _consider_new_position(self, symbol: str, signal_data: Dict):
        """📈 Considerar nueva posición con diversificación - CON DEBUG DETALLADO"""

        signal = signal_data['signal']
        confidence = signal_data['confidence']
        current_price = signal_data['current_price']

        print(f"    🚀 EVALUANDO NUEVA POSICIÓN: {symbol} {signal} ({confidence:.1f}%)")

        # ✅ NUEVO: Verificar cooldown post-cierre (PREVENIR RECOMPRAS INMEDIATAS)
        is_in_cooldown, remaining_time = self.is_in_post_close_cooldown(symbol)
        if is_in_cooldown:
            print(f"    ⏰ BLOQUEADO POR COOLDOWN POST-CIERRE: {symbol} - Tiempo restante: {remaining_time}")
            await self._send_discord_notification(
                f"⏰ **COOLDOWN POST-CIERRE**\n"
                f"📊 {symbol}: {signal}\n"
                f"🚫 Razón: Cooldown activo\n"
                f"⏱️ Tiempo restante: {remaining_time}\n"
                f"🎯 Confianza: {confidence:.1f}%"
            )
            return

        # ✅ NUEVO: Verificar protección post-pérdidas
        can_trade, protection_reason = self.loss_protection.can_open_position(symbol, confidence)
        if not can_trade:
            print(f"    🛡️ BLOQUEADO POR PROTECCIÓN POST-PÉRDIDAS: {protection_reason}")
            await self._send_discord_notification(
                f"🛡️ **PROTECCIÓN POST-PÉRDIDAS**\n"
                f"📊 {symbol}: {signal}\n"
                f"🚫 Razón: {protection_reason}\n"
                f"🎯 Confianza: {confidence:.1f}%"
            )
            return

        # 🔧 CORRECCIÓN: SELL no debe crear nuevas posiciones
        if signal == 'SELL':
            print(f"    ❌ BLOQUEADO: Señal SELL - No hay posición existente que vender en {symbol}")
            return

        # ✅ NUEVO: Verificar diversificación del portafolio ANTES de risk management
        print(f"    🎯 PASO 1: Verificando diversificación para {symbol}...")
        adjusted_position_usd = None
        try:
            # ✅ CORRECCIÓN: Usar el tamaño ajustado de la posición
            adjusted_position_usd = await self._check_portfolio_diversification_before_trade(symbol, signal_data)
            print(f"    ✅ PASO 1: Diversificación OK para {symbol}. Tamaño ajustado: ${adjusted_position_usd:.2f}")
        except Exception as e:
            if "Trade bloqueado por diversificación" in str(e):
                print(f"    ❌ PASO 1: DIVERSIFICACIÓN BLOQUEÓ: {symbol}: {str(e)}")
                await self._send_discord_notification(f"❌ **DIVERSIFICACIÓN BLOQUEÓ**: {symbol}: {str(e)}")
                return  # Salir sin ejecutar el trade
            else:
                print(f"    ⚠️ PASO 1: Error verificando diversificación para {symbol}: {e}")
                # Continuar con el trade si es un error técnico
                adjusted_position_usd = None # Usar tamaño por defecto

        # Verificar límites de riesgo
        print(f"    🛡️ PASO 2: Verificando límites de riesgo para {symbol}...")
        can_trade = False
        risk_reason = ""
        if self.risk_manager:
            print(f"    🛡️ PASO 2: Risk manager disponible, ejecutando check_risk_limits_before_trade...")
            can_trade, risk_reason = await self.risk_manager.check_risk_limits_before_trade(symbol, signal, confidence)
            print(f"    🛡️ PASO 2: Risk Manager resultado: can_trade={can_trade}, reason='{risk_reason}'")
        else:
            print(f"    ❌ PASO 2: Risk manager no disponible")
            await self._send_discord_notification(f"❌ **RISK MANAGER NO DISPONIBLE**: {symbol}")
            return

        if not can_trade:
            print(f"    ❌ PASO 2: RISK MANAGER BLOQUEÓ: {symbol}: {risk_reason}")
            await self._send_discord_notification(f"❌ **RISK MANAGER BLOQUEÓ**: {symbol}: {risk_reason}")
            return

        # Abrir nueva posición
        print(f"    💰 PASO 3: Intentando abrir posición para {symbol}...")
        position = None
        if self.risk_manager:
            print(f"    💰 PASO 3: Risk manager disponible, ejecutando open_position...")
            # ✅ CORRECCIÓN CRÍTICA: Pasar el tamaño ajustado por diversificación al risk manager
            if adjusted_position_usd is not None:
                # Usar método especial que respeta el tamaño de diversificación
                position = await self._execute_position_with_diversification_size(
                    symbol, signal, confidence, current_price, adjusted_position_usd
                )
            else:
                # Usar método normal del risk manager
                position = await self.risk_manager.open_position(
                    symbol, signal, confidence, current_price
                )
            if position:
                print(f"    ✅ PASO 3: POSICIÓN CREADA: {symbol} - Order ID: {position.order_id}")
            else:
                print(f"    ❌ PASO 3: NO SE PUDO CREAR POSICIÓN: {symbol} - Risk manager devolvió None")
                await self._send_discord_notification(f"❌ **NO SE PUDO CREAR POSICIÓN**: {symbol} - Risk manager falló")
        else:
            print("    ❌ PASO 3: Risk manager no disponible, no se puede abrir posición")
            await self._send_discord_notification(f"❌ **RISK MANAGER NO DISPONIBLE**: {symbol}")
            return

        if position:
            print(f"    ✅ PASO 4: Posición creada exitosamente, guardando en registros...")

            # ✅ CORREGIDO: Usar trade_id (que ahora es el order_id) como clave
            self.active_positions[position.order_id] = position
            print(f"    ✅ PASO 4: Posición guardada en active_positions con ID: {position.order_id}")

            # ✅ CRÍTICO: Actualizar inmediatamente el registry del portfolio manager
            print(f"    🔄 PASO 4.1: Actualizando registry del portfolio manager...")

            # Crear posición compatible para el registry
            registry_position = PortfolioManagerPosition(
                symbol=position.symbol,
                side=position.side,
                quantity=position.quantity,
                entry_price=position.entry_price,
                current_price=position.current_price,
                market_value=position.quantity * position.current_price,
                unrealized_pnl_usd=0.0,  # Inicial
                unrealized_pnl_percent=0.0,  # Inicial
                entry_time=position.entry_time,
                duration_minutes=0,
                order_id=position.order_id,
                batch_id=position.order_id
            )

            # Inicializar stops para la nueva posición
            registry_position = self.portfolio_manager.initialize_position_stops(registry_position)

            # Agregar al registry inmediatamente
            self.portfolio_manager.position_registry[position.order_id] = registry_position
            print(f"    ✅ PASO 4.1: Posición agregada al registry con ID: {position.order_id}")

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
            print(f"    ✅ PASO 4: Trade data preparado para DB")

            # El ID interno de la DB se genera automáticamente, no necesitamos guardarlo aquí.
            if self.database:
                await self.database.save_trade(trade_data)
                print(f"    ✅ PASO 4: Trade guardado en base de datos")
            else:
                print("    ⚠️ PASO 4: Database no disponible, trade no guardado en DB")

            self.trade_count += 1
            print(f"    ✅ PASO 4: Trade count incrementado a: {self.trade_count}")

            # Log del trade
            if self.database:
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
                print(f"    ✅ PASO 5: Notificación enviada vía Smart Discord Notifier")
            else:
                await self._send_discord_notification(f"🟢 **NUEVA POSICIÓN**\n"
                                                     f"📊 {symbol}: {signal}\n"
                                                     f"💰 Precio: ${current_price:.4f}\n"
                                                     f"🎯 Confianza: {confidence:.1%}\n"
                                                     f"📈 Cantidad: {position.quantity:.6f}")
                print(f"    ✅ PASO 5: Notificación enviada vía Discord básico")

            print(f"    🎉 ÉXITO TOTAL: Nueva posición creada para {symbol} - Proceso completado exitosamente")
        else:
            print(f"    ❌ FALLO FINAL: position es None después de todos los pasos - INVESTIGAR RISK MANAGER")

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

        # Si la señal es de venta, evaluar cada posición individualmente (NO cerrar automáticamente)
        if signal == 'SELL':
            print(f"📊 Señal de VENTA para {symbol} con {confidence:.1f}% confianza. Evaluando {len(existing_positions)} posición(es).")

            # ✅ CORRECCIÓN CRÍTICA: Evaluar cada posición según su rentabilidad actual
            for position in existing_positions:
                should_close = False
                close_reason = ""

                # ✅ PROTECCIÓN ESPECÍFICA PARA ETH - Criterios MÁS ESTRICTOS
                if symbol == 'ETHUSDT':
                    # ETH requiere criterios mucho más estrictos para cerrar posiciones
                    current_time = datetime.now()

                    # Verificar tiempo mínimo de retención
                    if hasattr(position, 'entry_time'):
                        position_age_minutes = (current_time - position.entry_time).total_seconds() / 60
                    else:
                        # Usar timestamp de última acción como aproximación
                        last_action_time = self.last_position_action.get(symbol, current_time - timedelta(minutes=30))
                        position_age_minutes = (current_time - last_action_time).total_seconds() / 60

                    min_hold_time = self.eth_position_protection['min_hold_time_minutes']

                    # CRITERIOS ESPECÍFICOS PARA ETH:
                    if confidence >= 90.0:  # Confianza EXTREMA (90%+)
                        if position.pnl_percent < -4.0:  # Pérdida significativa >4%
                            should_close = True
                            close_reason = "ETH_SELL_EXTREME_CONF_BIG_LOSS"
                        elif position.pnl_percent > 5.0 and position_age_minutes > min_hold_time:  # Ganancia >5% y tiempo suficiente
                            should_close = True
                            close_reason = "ETH_SELL_EXTREME_CONF_BIG_PROFIT"
                    elif confidence >= 85.0 and position_age_minutes > min_hold_time:  # Confianza muy alta y tiempo mínimo
                        if position.pnl_percent < -3.0:  # Pérdida >3%
                            should_close = True
                            close_reason = "ETH_SELL_HIGH_CONF_LOSS_PROTECTION"
                        elif position.pnl_percent > 6.0:  # Ganancia muy alta >6%
                            should_close = True
                            close_reason = "ETH_SELL_HIGH_CONF_PROFIT_TAKING"

                    if not should_close:
                        print(f"  🛡️ ETH PROTEGIDO: Manteniendo posición {position.order_id}")
                        print(f"      📊 PnL: {position.pnl_percent:.1f}%, Edad: {position_age_minutes:.1f}min, Conf: {confidence:.1f}%")
                        print(f"      ✅ ETH requiere criterios más estrictos para cierre de posición")

                # ✅ CRITERIOS ORIGINALES PARA OTROS SÍMBOLOS
                else:
                    # Criterios menos estrictos para BTC, BNB, XRP
                    if confidence >= 98.0:  # Confianza alta
                        if position.pnl_percent > 2.0:  # Si está en ganancia > 2%
                            should_close = True
                            close_reason = "SIGNAL_SELL_HIGH_CONF_PROFIT"
                        elif position.pnl_percent < -1.5:  # O pérdida > 1.5%
                            should_close = True
                            close_reason = "SIGNAL_SELL_HIGH_CONF_LOSS"
                    elif confidence >= 99.0:  # Confianza muy alta
                        should_close = True  # Cerrar independientemente del PnL
                        close_reason = "SIGNAL_SELL_VERY_HIGH_CONF"

                if should_close:
                    print(f"  🔥 Cerrando posición {position.order_id}: PnL {position.pnl_percent:.1f}% - {close_reason}")
                    await self._close_position(position.order_id, close_reason)
                else:
                    print(f"  ⏸️ Manteniendo posición {position.order_id}: PnL {position.pnl_percent:.1f}% - Confianza SELL insuficiente")

        # ✅ NUEVO: Sistema de reversión mejorado con señales consecutivas
        print(f"🔄 Evaluando reversión con sistema de señales consecutivas para {symbol}...")

        # Verificar si es señal de reversión (opuesta a posiciones existentes)
        is_reversal_signal = False
        for position in existing_positions:
            if (position.side == 'BUY' and signal == 'SELL') or (position.side == 'SELL' and signal == 'BUY'):
                is_reversal_signal = True
                break

        if is_reversal_signal:
            # Evaluar con el nuevo sistema de señales consecutivas
            should_execute, reason = self._evaluate_reversal_with_consecutive_signals(symbol, signal, confidence)

            if should_execute:
                print(f"  ✅ REVERSIÓN APROBADA por sistema consecutivo: {reason}")

                # Ejecutar reversión para todas las posiciones del símbolo
                for position in existing_positions:
                    if (position.side == 'BUY' and signal == 'SELL') or (position.side == 'SELL' and signal == 'BUY'):
                        print(f"  🔄 Ejecutando reversión para {position.order_id}: PnL {position.pnl_percent:.1f}%")
                        await self._close_position(position.order_id, f"REVERSIÓN_CONSECUTIVA_{reason}")

                # ✅ LIMPIAR TRACKING después de ejecutar reversión exitosa
                if symbol in self.reversal_tracking:
                    self.reversal_tracking[symbol] = {
                        'consecutive_signals': [],
                        'last_signal_time': None,
                        'current_signal_direction': None,
                        'started_tracking': None
                    }
                    print(f"  🧹 Tracking de reversión limpiado para {symbol} después de ejecución")

            else:
                print(f"  ⏸️ Reversión en progreso para {symbol}: {reason}")
                print(f"      📊 Sistema requiere múltiples señales consecutivas para mayor seguridad")
        else:
            print(f"  ℹ️ Señal {signal} no es de reversión para posiciones existentes en {symbol}")

    async def _close_position(self, order_id: str, reason: str):
        """📉 Cerrar posición específica por ID de orden"""

        # ✅ CORREGIDO: Buscar en portfolio_manager.position_registry
        position = self.portfolio_manager.position_registry.get(order_id)
        if not position:
            print(f"ℹ️ Intento de cerrar posición {order_id} omitido. No encontrada en registry.")
            return

        # ✅ NUEVO: Verificar que no esté ya marcada para cierre
        if hasattr(position, 'is_closing') and position.is_closing:
            print(f"ℹ️ Posición {order_id} ya está siendo cerrada.")
            return

        # ✅ CRÍTICO: Marcar como en proceso de cierre
        position.is_closing = True

        print(f"👇 Iniciando cierre para {position.symbol} (ID Orden: {order_id}) por motivo: {reason}")

        # Ahora el resto de la función puede proceder de forma segura.
        symbol = position.symbol
        current_price = await self.get_current_price(symbol)

        # Ejecutar orden REAL de venta en Binance
        print(f"   💸 Ejecutando orden de VENTA REAL para {position.quantity} de {symbol} a ${current_price:.4f}")

        # ✅ CRÍTICO: Ejecutar orden real de cierre
        close_order_result = await self._execute_sell_order(position)
        if close_order_result:
            print(f"🎉 Orden de cierre ejecutada: ID {close_order_result['orderId']}")
            # Usar precio real de la orden ejecutada si está disponible
            if 'fills' in close_order_result and len(close_order_result['fills']) > 0:
                current_price = float(close_order_result['fills'][0]['price'])
        else:
            print(f"⚠️ No se pudo ejecutar orden de cierre, usando precio de mercado")

        # Calcular PnL final
        if position.side == 'BUY':
            pnl_percent = ((current_price - position.entry_price) / position.entry_price) * 100
        else:
            pnl_percent = ((position.entry_price - current_price) / position.entry_price) * 100

        pnl_usd = (pnl_percent / 100) * (position.quantity * position.entry_price)

        # ✅ NUEVO: Registrar cierre para protección post-pérdidas
        self.loss_protection.register_position_close(
            symbol=symbol,
            pnl_percent=pnl_percent,
            pnl_usd=pnl_usd,
            close_reason=reason,
            entry_time=position.entry_time
        )
        
        # ✅ NUEVO: Registrar cierre para cooldown post-cierre (PREVENIR RECOMPRAS INMEDIATAS)
        print(f"🔔 REGISTRANDO COOLDOWN POST-CIERRE: {symbol}")
        print(f"   📝 Razón de cierre: '{reason}'")
        print(f"   💰 PnL: {pnl_percent:.2f}%")
        
        cooldown_minutes = self.register_position_close(
            symbol=symbol,
            close_type=reason,
            pnl_percent=pnl_percent
        )
        
        print(f"🛡️ COOLDOWN REGISTRADO EXITOSAMENTE: {symbol} por {cooldown_minutes} minutos")

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

        # ✅ CORREGIDO: Remover del portfolio_manager.position_registry
        if order_id in self.portfolio_manager.position_registry:
            del self.portfolio_manager.position_registry[order_id]
            print(f"🗑️ Posición {order_id} eliminada del registry")

        # ✅ NUEVO: Limpiar también el cache de trailing stop
        if hasattr(self.portfolio_manager, 'trailing_cache') and order_id in self.portfolio_manager.trailing_cache:
            del self.portfolio_manager.trailing_cache[order_id]
            print(f"🗑️ Cache de trailing eliminado para {order_id}")

        # ✅ NUEVO: Actualizar timestamp de última acción para protección ETH
        if symbol == 'ETHUSDT':
            self.eth_position_protection['last_close_time'] = datetime.now()
            print(f"🛡️ ETH: Timestamp de cierre actualizado para protección")

        # ✅ NUEVO: Limpiar tracking de reversión cuando se cierra una posición
        if symbol in self.reversal_tracking and self.reversal_tracking[symbol]['consecutive_signals']:
            print(f"🧹 LIMPIEZA AUTOMÁTICA: Tracking de reversión limpiado para {symbol} (posición cerrada)")
            self.reversal_tracking[symbol] = {
                'consecutive_signals': [],
                'last_signal_time': None,
                'current_signal_direction': None,
                'started_tracking': None
            }

        # Actualizar registro de última acción por símbolo
        self.last_position_action[symbol] = datetime.now()

        # Log y notificación
        color = "🟢" if pnl_usd > 0 else "🔴"
        await self.database.log_event(
            'INFO', 'TRADING',
            f'Posición cerrada: {symbol} - PnL: {pnl_percent:.2f}% (${pnl_usd:.2f})',
            symbol
        )

        # ✅ MEJORADO: Usar Smart Discord Notifier para cierres (consistencia)
        if hasattr(self, 'discord_notifier'):
            trade_notification_data = {
                'symbol': symbol,
                'side': position.side,
                'value_usd': position.quantity * position.entry_price,
                'price': current_price,
                'confidence': 0.0,  # No aplicable para cierres
                'pnl_percent': pnl_percent,
                'pnl_usd': pnl_usd,
                'reason': reason
            }
            await self.discord_notifier.send_trade_notification(trade_notification_data)
        else:
            # Fallback al sistema básico
            await self._send_discord_notification(f"{color} **POSICIÓN CERRADA**\n"
                                                 f"📊 {symbol}: {position.side}\n"
                                                 f"📈 PnL: {pnl_percent:.2f}% (${pnl_usd:.2f})\n"
                                                 f"🔄 Razón: {reason}")

        print(f"📉 Posición cerrada: {symbol} (ID de orden: {order_id}) - PnL: {pnl_percent:.2f}% (${pnl_usd:.2f})")

        # ✅ NUEVO: Registrar cierre para protección post-pérdidas
        self.loss_protection.register_position_close(
            symbol=symbol,
            pnl_percent=pnl_percent,
            pnl_usd=pnl_usd,
            close_reason=reason,
            entry_time=position.entry_time
        )

    async def _check_portfolio_diversification_before_trade(self, symbol: str, signal_data: Dict) -> Optional[float]:
        """
        🎯 SISTEMA DE DIVERSIFICACIÓN INTELIGENTE CON LÍMITES POR PAR
        ---
        Implementa límites específicos de exposición por activo y priorización de BTC

        Límites por par:
        - BTC: máximo 50% del portafolio
        - ETH: máximo 20% del portafolio
        - BNB: máximo 15% del portafolio
        - XRP: máximo 15% del portafolio

        Priorización: BTC > ETH > BNB > XRP en señales simultáneas

        Returns:
            Optional[float]: El tamaño ajustado de la posición en USD si es válida,
                             o lanza una excepción si se bloquea.
        """

        try:
            print(f"🎯 Verificación de diversificación inteligente para {symbol}")

            # ✅ CRÍTICO: Actualizar balance antes de verificar diversificación
            await self.update_balance_from_binance()

            # Obtener snapshot actual del portafolio
            if self.portfolio_manager is None:
                raise Exception("Portfolio Manager no inicializado")
            snapshot = await self.portfolio_manager.get_portfolio_snapshot()

            # ✅ CONFIGURACIÓN: Límites específicos por par
            SYMBOL_LIMITS = {
                'ADAUSDT': 20.0,  # ADA máximo 25% del portafolio
                'BTCUSDT': 30.0,  # BTC máximo 25% del portafolio
                'ETHUSDT': 30.0,  # ETH máximo 30% del portafolio
                'BNBUSDT': 25.0,  # BNB máximo 25% del portafolio
                'XRPUSDT': 20.0,  # XRP máximo 20% del portafolio
                'DOTUSDT': 30.0,  # DOT máximo 30% del portafolio
                'SOLUSDT': 20.0,  # SOL máximo 20% del portafolio (nuevo par activado)
                'POLUSDT': 20.0  # POL máximo 20% del portafolio (nuevo par activado)
            }

            # ✅ PRIORIZACIÓN: Orden de preferencia para señales simultáneas
            PRIORITY_ORDER = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'DOTUSDT', 'SOLUSDT','ADAUSDT', 'POLUSDT']

            # ✅ DEBUG: Información del balance
            print(f"🔍 DEBUG DIVERSIFICACIÓN INTELIGENTE:")
            print(f"   💰 Balance actual: ${self.current_balance:.2f}")
            print(f"   📊 Posiciones activas: {len(snapshot.active_positions)}")

            # Calcular exposición actual por símbolo
            symbol_exposures = {}
            total_invested_value = 0.0  # Total invertido originalmente
            total_current_value = 0.0   # Valor actual de mercado

            for pos in snapshot.active_positions:
                symbol_key = pos.symbol
                if symbol_key not in symbol_exposures:
                    symbol_exposures[symbol_key] = {
                        'positions': [],
                        'total_invested': 0.0,
                        'current_value': 0.0,
                        'count': 0
                    }

                symbol_exposures[symbol_key]['positions'].append(pos)
                symbol_exposures[symbol_key]['count'] += 1

                # Valor invertido originalmente (entry_price * quantity)
                invested_value = pos.quantity * pos.entry_price
                symbol_exposures[symbol_key]['total_invested'] += invested_value
                total_invested_value += invested_value

                # Valor actual de mercado (market_value ya calculado)
                current_market_value = pos.market_value
                symbol_exposures[symbol_key]['current_value'] += current_market_value
                total_current_value += current_market_value

                print(f"   📍 {pos.symbol}: {pos.quantity:.6f} x ${pos.entry_price:.4f} = ${invested_value:.2f} (actual: ${current_market_value:.2f})")

            # ✅ CORRECCIÓN CRÍTICA: Total del portafolio = balance disponible + valor actual de mercado
            total_portfolio_value = self.current_balance + total_current_value

            # ✅ DEBUG: Mostrar cálculos correctos
            print(f"🔍 CÁLCULO DE PORTAFOLIO CORREGIDO:")
            print(f"   💰 Balance disponible: ${self.current_balance:.2f}")
            print(f"   📊 Valor actual invertido: ${total_current_value:.2f}")
            print(f"   💼 Total portafolio: ${total_portfolio_value:.2f}")
            print(f"   📈 Total originalmente invertido: ${total_invested_value:.2f}")

            # ✅ VALIDACIÓN 1: Máximo 3 posiciones por símbolo
            current_positions_count = symbol_exposures.get(symbol, {}).get('count', 0)
            
            # ✅ CRÍTICO: Verificar sincronización entre contadores
            portfolio_positions_count = len([pos for pos in self.portfolio_manager.position_registry.values() if pos.symbol == symbol])
            risk_manager_positions_count = len([pos for pos in self.active_positions.values() if pos.symbol == symbol])
            
            print(f"🔍 VERIFICACIÓN SINCRONIZACIÓN POSICIONES {symbol}:")
            print(f"   📊 Portfolio Manager: {portfolio_positions_count} posiciones")
            print(f"   📊 Risk Manager: {risk_manager_positions_count} posiciones")
            print(f"   📊 Symbol Exposures: {current_positions_count} posiciones")
            
            # ✅ CORRECCIÓN: Usar el contador más confiable (portfolio_manager)
            if portfolio_positions_count >= 3:
                print(f"🚫 DIVERSIFICACIÓN: Máximo 3 posiciones por símbolo alcanzado para {symbol}")
                print(f"   📊 Posiciones actuales en {symbol}: {portfolio_positions_count}/3")
                
                # ✅ NUEVO: Limpiar contador desincronizado del risk manager
                if portfolio_positions_count != risk_manager_positions_count:
                    print(f"   🔧 DESINCRONIZACIÓN DETECTADA: Limpiando contador del risk manager...")
                    await self._cleanup_desynchronized_positions(symbol)
                
                raise Exception(f"Trade bloqueado por diversificación: Máximo 3 posiciones por símbolo en {symbol}")

            # ✅ CÁLCULO: Tamaño de nueva posición propuesta
            confidence = signal_data['confidence']

            # Tamaño base según confianza (entre 10% y 18% del balance disponible)
            base_size_percent = min(18.0, max(10.0, (confidence/100) * 20.0))
            proposed_position_usd = (self.current_balance * base_size_percent / 100)

            # ✅ VALIDACIÓN 2: Límites específicos por símbolo
            print(f"🔍 DEBUG DIVERSIFICACIÓN: symbol='{symbol}' | SYMBOL_LIMITS={SYMBOL_LIMITS}")
            symbol_limit = SYMBOL_LIMITS.get(symbol, 10.0)  # Default 10% para otros
            print(f"🔍 DEBUG DIVERSIFICACIÓN: symbol_limit={symbol_limit}% para {symbol}")

            # ✅ CORRECCIÓN CRÍTICA: Usar valor actual de mercado para exposición
            current_symbol_market_value = symbol_exposures.get(symbol, {}).get('current_value', 0.0)
            new_total_symbol_value = current_symbol_market_value + proposed_position_usd
            new_symbol_exposure_percent = (new_total_symbol_value / (total_portfolio_value + proposed_position_usd)) * 100 if total_portfolio_value > 0 else 0

            # Exposición actual sin la nueva posición
            current_symbol_exposure_percent = (current_symbol_market_value / total_portfolio_value) * 100 if total_portfolio_value > 0 else 0

            # ✅ VALIDACIÓN CRÍTICA: Verificar si ya excede el límite SIN nueva posición
            if current_symbol_exposure_percent > symbol_limit:
                print(f"🚫 DIVERSIFICACIÓN: {symbol} YA EXCEDE el límite de exposición")
                print(f"   📊 Exposición actual: {current_symbol_exposure_percent:.1f}% > {symbol_limit}%")
                print(f"   💰 Valor actual en {symbol}: ${current_symbol_market_value:.2f}")
                print(f"   💼 Total portafolio: ${total_portfolio_value:.2f}")
                print(f"   🚫 NO SE PERMITE nueva posición hasta reducir exposición")
                raise Exception(f"Trade bloqueado por diversificación: {symbol} ya excede límite de {symbol_limit}% (actual: {current_symbol_exposure_percent:.1f}%)")

            if new_symbol_exposure_percent > symbol_limit:
                # Calcular el máximo permitido para este símbolo
                max_allowed_total_value = (total_portfolio_value + proposed_position_usd) * symbol_limit / 100
                max_new_position = max_allowed_total_value - current_symbol_market_value

                if max_new_position <= 11.0:  # Mínimo Binance
                    print(f"🚫 DIVERSIFICACIÓN: Nueva posición excedería límite para {symbol}")
                    print(f"   📊 Exposición actual: {current_symbol_exposure_percent:.1f}%")
                    print(f"   📊 Nueva exposición: {new_symbol_exposure_percent:.1f}% > {symbol_limit}%")
                    print(f"   💰 Máximo permitido: ${max_new_position:.2f} (mínimo: $11)")
                    raise Exception(f"Trade bloqueado por diversificación: {symbol} excedería límite de {symbol_limit}%")
                else:
                    # Ajustar el tamaño de la posición al máximo permitido
                    proposed_position_usd = max_new_position
                    print(f"⚖️ AJUSTE DE DIVERSIFICACIÓN: {symbol} ajustado a ${proposed_position_usd:.2f}")

            # ✅ VALIDACIÓN 3: Priorización en señales simultáneas
            # Verificar si hay otras señales pendientes con mayor prioridad
            if hasattr(self, 'pending_signals') and self.pending_signals:
                current_priority = PRIORITY_ORDER.index(symbol) if symbol in PRIORITY_ORDER else len(PRIORITY_ORDER)

                for pending_symbol in self.pending_signals:
                    if pending_symbol in PRIORITY_ORDER:
                        pending_priority = PRIORITY_ORDER.index(pending_symbol)
                        if pending_priority < current_priority:
                            print(f"⏸️ PRIORIZACIÓN: {symbol} pausado, {pending_symbol} tiene mayor prioridad")
                            print(f"   📊 Orden de prioridad: {' > '.join(PRIORITY_ORDER)}")
                            raise Exception(f"Trade pausado por priorización: {pending_symbol} tiene mayor prioridad que {symbol}")

            # ✅ VALIDACIÓN 4: Exposición total del portafolio
            total_exposure_percent = ((total_portfolio_value - self.current_balance) / total_portfolio_value) * 100 if total_portfolio_value > 0 else 0
            max_total_exposure = 85.0  # Máximo 85% del portafolio invertido

            if confidence >= 80.0:
                max_total_exposure = 90.0  # Permitir hasta 90% con alta confianza

            new_total_exposure = total_exposure_percent + (proposed_position_usd / total_portfolio_value) * 100

            if new_total_exposure > max_total_exposure:
                print(f"🚫 DIVERSIFICACIÓN: Exposición total muy alta")
                print(f"   📊 Exposición actual: {total_exposure_percent:.1f}%")
                print(f"   📊 Nueva exposición: {new_total_exposure:.1f}% > {max_total_exposure:.0f}%")
                raise Exception(f"Trade bloqueado por diversificación: Exposición total > {max_total_exposure:.0f}%")

            # ✅ INFORMACIÓN: Mostrar análisis detallado
            print(f"✅ DIVERSIFICACIÓN: Trade aprobado para {symbol}")
            print(f"   🎯 Límite específico: {symbol_limit}% (BTC:35%, ETH:25%, BNB:25%, XRP:15%, ADA:20%)")
            print(f"   📊 Exposición actual en {symbol}: {current_symbol_exposure_percent:.1f}%")
            print(f"   📊 Nueva exposición en {symbol}: {new_symbol_exposure_percent:.1f}%/{symbol_limit}%")
            print(f"   💰 Tamaño propuesto: ${proposed_position_usd:.2f} ({base_size_percent:.1f}%)")
            print(f"   📊 Posiciones en {symbol}: {current_positions_count}/3")
            print(f"   🔥 Confianza: {confidence:.1f}%")
            print(f"   🏆 Prioridad: {PRIORITY_ORDER.index(symbol) + 1 if symbol in PRIORITY_ORDER else 'N/A'}")

            # ✅ RESUMEN: Estado del portafolio (CORREGIDO)
            if symbol_exposures:
                print(f"📊 ESTADO ACTUAL DEL PORTAFOLIO:")
                for sym in PRIORITY_ORDER:
                    if sym in symbol_exposures:
                        exposure = symbol_exposures[sym]
                        # ✅ CORRECCIÓN: Usar valor actual de mercado para exposición
                        current_market_value = exposure['current_value']
                        exposure_percent = (current_market_value / total_portfolio_value) * 100
                        limit = SYMBOL_LIMITS[sym]
                        status = "✅" if exposure_percent <= limit * 0.8 else "⚠️" if exposure_percent <= limit else "🔴"
                        print(f"   {status} {sym}: {exposure['count']} pos, {exposure_percent:.1f}%/{limit}% (${current_market_value:.2f} actual, ${exposure['total_invested']:.2f} invertido)")

            # ✅ DEVOLVER TAMAÑO AJUSTADO
            return proposed_position_usd

        except Exception as e:
            if "Trade bloqueado por diversificación" in str(e) or "Trade pausado por priorización" in str(e):
                # Bloqueos legítimos de diversificación
                print(f"🚫 {str(e)}")
                if self.database:
                    await self.database.log_event('WARNING', 'DIVERSIFICATION', str(e), symbol)

                # Notificación Discord informativa
                if self.discord_notifier:
                    await self.discord_notifier.send_system_notification(
                        f"⚖️ **DIVERSIFICACIÓN INTELIGENTE**\n"
                        f"📊 {symbol}: {signal_data['signal']}\n"
                        f"💡 {str(e).replace('Trade bloqueado por diversificación: ', '').replace('Trade pausado por priorización: ', '')}\n"
                        f"🎯 Confianza: {signal_data['confidence']:.1%}\n"
                        f"🏆 Límites: BTC≤35%, ETH≤25%, BNB≤25%, XRP≤15%, ADA≤20%"
                    )

                raise  # Re-lanzar bloqueos legítimos
            else:
                # Errores técnicos no deben bloquear trades
                print(f"⚠️ Error técnico en diversificación (ignorado): {e}")
                print(f"✅ Continuando con el trade para {symbol}")
                return None # Devolver None para que se use el tamaño por defecto

    async def _heartbeat_monitor(self):
        """💓 Monitor de latido del sistema"""
        while self.status == TradingManagerStatus.RUNNING:
            try:
                # Verificar conectividad cada 5 minutos
                await asyncio.sleep(180)

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
        """🔍 Monitoreo continuo de posiciones y gestión de riesgo - ALTA FRECUENCIA"""
        try:
            # ✅ MÁXIMA FRECUENCIA: Datos frescos cada 30 segundos para TCN
            snapshot = await self.portfolio_manager.get_portfolio_snapshot()

            if not snapshot.active_positions:
                print("   📊 Sin posiciones activas para monitorear")
                return

            print(f"🔍 Monitoreando {len(snapshot.active_positions)} posición(es)...")

            # 2. Actualizar precios para cada posición - FRECUENCIA MÁXIMA
            symbols_to_update = list(set([pos.symbol for pos in snapshot.active_positions]))
            current_prices = await self.portfolio_manager.update_all_prices(symbols_to_update)

            positions_to_close = []
            trailing_updates = []

            # 3. ✅ NUEVO: Procesar cada posición individualmente con trailing stop
            for i, position in enumerate(snapshot.active_positions):
                try:
                    # ✅ CRÍTICO: Verificar que la posición aún existe en el registry
                    if not position.order_id or position.order_id not in self.portfolio_manager.position_registry:
                        print(f"⚠️ Posición {position.symbol} Pos #{position.order_id} no encontrada en registry, saltando...")
                        continue

                    # ✅ CRÍTICO: Verificar que no esté marcada como cerrada
                    if hasattr(position, 'is_closing') and position.is_closing:
                        print(f"⚠️ Posición {position.symbol} Pos #{position.order_id} ya está siendo cerrada, saltando...")
                        continue

                    current_price = current_prices.get(position.symbol, position.current_price)

                    # Actualizar precio actual en la posición
                    position.current_price = current_price

                    # 🔄 Recalcular PnL con precio actual
                    if position.side == 'BUY':
                        entry_value = position.quantity * position.entry_price
                        current_value = position.quantity * current_price
                        position.unrealized_pnl_usd = current_value - entry_value
                        position.unrealized_pnl_percent = (position.unrealized_pnl_usd / entry_value) * 100 if entry_value > 0 else 0.0
                        position.market_value = current_value

                    # ✅ NUEVO: Aplicar trailing stop profesional
                    updated_position, stop_triggered, trigger_reason = self.portfolio_manager.update_trailing_stop_professional(
                        position, current_price
                    )

                    # ✅ CRÍTICO: Actualizar la posición en el snapshot Y en el registry
                    snapshot.active_positions[i] = updated_position

                    # ✅ PERSISTENCIA: Actualizar también en el registry para mantener estado
                    if updated_position.order_id and updated_position.order_id in self.portfolio_manager.position_registry:
                        self.portfolio_manager.position_registry[updated_position.order_id] = updated_position

                    # Si se actualiza el trailing, registrar el cambio
                    if hasattr(updated_position, 'trailing_stop_active') and updated_position.trailing_stop_active:
                        if updated_position.trailing_movements > position.trailing_movements:
                            trailing_updates.append(f"📈 {updated_position.symbol} Pos #{updated_position.order_id}: Trail movido a ${updated_position.trailing_stop_price:.4f}")

                    # Verificar condiciones de cierre
                    should_close, close_reason = await self._check_position_exit_conditions(updated_position, current_price)

                    if stop_triggered or should_close:
                        reason = trigger_reason if stop_triggered else close_reason
                        
                        # ✅ CRÍTICO: Verificar nuevamente que la posición existe antes de marcarla para cierre
                        if updated_position.order_id in self.portfolio_manager.position_registry:
                            positions_to_close.append((updated_position, reason))
                            print(f"🛑 Marcando para cierre: {updated_position.symbol} Pos #{updated_position.order_id} - {reason}")
                        else:
                            print(f"⚠️ Posición {updated_position.symbol} Pos #{updated_position.order_id} ya no existe en registry, omitiendo cierre")

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
                stop_loss_threshold = float(os.getenv('STOP_LOSS_PERCENT', '1.4'))
                if pnl_percent <= -stop_loss_threshold:
                    return True, f"STOP_LOSS_TRADICIONAL (-{abs(pnl_percent):.2f}%)"

                # Take Profit tradicional
                take_profit_threshold = float(os.getenv('TAKE_PROFIT_PERCENT', '4.0'))
                if pnl_percent >= take_profit_threshold:
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
                    # ✅ CRÍTICO: Verificar que la posición aún existe en el registry
                    if not position.order_id or position.order_id not in self.portfolio_manager.position_registry:
                        print(f"⚠️ Posición {position.symbol} Pos #{position.order_id} ya no existe en registry, omitiendo cierre")
                        continue

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
                        real_quantity = float(order_result.get('executedQty', position.quantity))

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

                    # ✅ CRÍTICO: Registrar cierre para cooldown post-cierre (PREVENIR RECOMPRAS INMEDIATAS)
                    print(f"🔔 REGISTRANDO COOLDOWN POST-CIERRE: {position.symbol}")
                    print(f"   📝 Razón de cierre: '{reason}'")
                    print(f"   💰 PnL: {real_pnl_percent:.2f}%")
                    
                    cooldown_minutes = self.register_position_close(
                        symbol=position.symbol,
                        close_type=reason,
                        pnl_percent=real_pnl_percent
                    )
                    
                    print(f"🛡️ COOLDOWN REGISTRADO EXITOSAMENTE: {position.symbol} por {cooldown_minutes} minutos")

                    # Actualizar métricas
                    self.metrics['total_trades'] += 1
                    if real_pnl_usd > 0:
                        self.metrics['profitable_trades'] += 1

                    # ✅ CRÍTICO: Remover del portfolio_manager.position_registry
                    if position.order_id in self.portfolio_manager.position_registry:
                        del self.portfolio_manager.position_registry[position.order_id]
                        print(f"🗑️ Posición {position.order_id} eliminada del registry")

                    # ✅ NUEVO: Limpiar también el cache de trailing stop
                    if hasattr(self.portfolio_manager, 'trailing_cache') and position.order_id in self.portfolio_manager.trailing_cache:
                        del self.portfolio_manager.trailing_cache[position.order_id]
                        print(f"🗑️ Cache de trailing eliminado para {position.order_id}")

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

            # ✅ NUEVO: Verificar balance real antes de ejecutar
            asset_symbol = position.symbol.replace('USDT', '')  # BTC, ETH, BNB
            available_balance = await self._get_available_balance(asset_symbol)

            if available_balance <= 0:
                print(f"❌ No hay balance disponible de {asset_symbol} para vender")
                return None

            # ✅ CORREGIDO: Usar el menor entre la cantidad de la posición y el balance disponible
            max_sellable = min(position.quantity, available_balance)

            print(f"📊 Verificación de balance {asset_symbol}:")
            print(f"   Posición: {position.quantity:.8f}")
            print(f"   Disponible: {available_balance:.8f}")
            print(f"   A vender: {max_sellable:.8f}")

            # Preparar parámetros de orden
            timestamp = int(time.time() * 1000)

            # Ajustar cantidad según filtros del símbolo
            adjusted_quantity = await self._adjust_quantity_for_symbol(position.symbol, max_sellable)

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
                self.binance_config.secret_key.encode('utf-8'),
                query_string.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()

            params['signature'] = signature

            # Headers de autenticación
            headers = {
                'X-MBX-APIKEY': self.binance_config.api_key,
                'Content-Type': 'application/x-www-form-urlencoded'
            }

            print(f"📡 Ejecutando orden de cierre: {close_side} {params['quantity']} {position.symbol}")

            # Ejecutar orden POST /api/v3/order
            async with aiohttp.ClientSession() as session:
                url = f"{self.binance_config.base_url}/api/v3/order"

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
                url = f"{self.binance_config.base_url}/api/v3/exchangeInfo"
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

                                    # ✅ CORREGIDO: Ajustar cantidad correctamente al step size
                                    if quantity < min_qty:
                                        print(f"⚠️ Cantidad {quantity:.8f} menor que mínimo {min_qty:.8f}")
                                        return 0.0  # No se puede vender cantidad menor al mínimo

                                    # Redondear hacia abajo para no exceder la cantidad disponible
                                    adjusted_qty = (quantity // step_size) * step_size

                                    # Verificar que sigue siendo >= mínimo después del ajuste
                                    if adjusted_qty < min_qty:
                                        print(f"⚠️ Cantidad ajustada {adjusted_qty:.8f} menor que mínimo {min_qty:.8f}")
                                        return 0.0

                                    print(f"🔧 Cantidad ajustada: {quantity:.8f} → {adjusted_qty:.8f} (step: {step_size:.8f})")
                                    return adjusted_qty

            # Si no se puede obtener filtros, usar cantidad original
            print(f"⚠️ No se pudieron obtener filtros para {symbol}, usando cantidad original")
            return quantity

        except Exception as e:
            print(f"❌ Error ajustando cantidad para {symbol}: {e}")
            return quantity

    async def _get_available_balance(self, asset: str) -> float:
        """💰 Obtener balance disponible de un activo específico"""
        try:
            # Obtener información de cuenta
            account_info = await self.get_account_info()
            if not account_info:
                print(f"❌ No se pudo obtener información de cuenta para {asset}")
                return 0.0

            # Buscar el balance del activo
            for asset_symbol, balance_info in account_info.balances.items():
                if asset_symbol == asset:
                    available = balance_info.get('free', 0.0) if isinstance(balance_info, dict) else 0.0
                    print(f"💰 Balance disponible {asset}: {available:.8f}")
                    return available

            print(f"⚠️ No se encontró balance para {asset}")
            return 0.0

        except Exception as e:
            print(f"❌ Error obteniendo balance de {asset}: {e}")
            return 0.0

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
            trades_today = await self._get_total_trades_today()
            win_rate = await self._calculate_win_rate()

            # ✅ CORRECCIÓN: Usar el snapshot del portfolio manager para obtener datos consistentes y reales
            if self.portfolio_manager:
                snapshot = await self.portfolio_manager.get_portfolio_snapshot()
                total_exposure = snapshot.total_balance_usd - snapshot.free_usdt
                total_balance = snapshot.total_balance_usd
                # El PnL de la sesión es el PnL no realizado del snapshot
                self.session_pnl = snapshot.total_unrealized_pnl
            else:
                # Fallback por si el portfolio manager no está listo (menos preciso)
                total_exposure = 0
                for position in self.active_positions.values():
                    if hasattr(position, 'current_price') and position.current_price > 0:
                        total_exposure += position.quantity * position.current_price
                    else:
                        total_exposure += position.quantity * position.entry_price
                total_balance = self.current_balance + total_exposure

            # ✅ CORRECCIÓN: Usar el balance total real para el cálculo de exposición
            exposure_percent = (total_exposure / total_balance) * 100 if total_balance > 0 else 0

            # Actualizar balance actual en risk manager con el valor correcto
            if self.risk_manager:
                await self.risk_manager.update_balance(total_balance)

            metrics_data = {
                'timestamp': datetime.now(),
                'total_balance': total_balance,
                'daily_pnl': self.session_pnl,
                'total_pnl': self.session_pnl,  # Para sesión actual
                'daily_return_percent': (self.session_pnl / total_balance) * 100 if total_balance > 0 else 0,
                'total_return_percent': (self.session_pnl / total_balance) * 100 if total_balance > 0 else 0,
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

            if self.database:
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
        """🚨 Parada de emergencia - SIN CERRAR POSICIONES"""
        print("🚨 Parada de emergencia solicitada")
        self.status = TradingManagerStatus.EMERGENCY_STOP

        # ✅ NO CERRAR POSICIONES - Solo cambiar estado
        if self.active_positions:
            print(f"⚠️ AVISO: {len(self.active_positions)} posiciones activas NO serán cerradas")
            print(f"   📊 Las posiciones permanecen activas en Binance")
            print(f"   🔄 El bot se puede reiniciar para continuar gestionándolas")

        if self.database:
            await self.database.log_event('CRITICAL', 'SYSTEM', 'Parada de emergencia activada - Posiciones preservadas')
        print("🚨 PARADA DE EMERGENCIA ACTIVADA - Posiciones preservadas")

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
            'environment': self.binance_config.environment,
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
        """🔄 Apagado controlado del sistema - SIN CERRAR POSICIONES"""
        print("🔄 Iniciando apagado del sistema...")

        self.status = TradingManagerStatus.STOPPED

        # ✅ NO CERRAR POSICIONES - Solo cambiar estado
        if self.active_positions:
            print(f"⚠️ AVISO: {len(self.active_positions)} posiciones activas NO serán cerradas")
            print(f"   📊 Las posiciones permanecen activas en Binance")
            print(f"   🔄 El bot se puede reiniciar para continuar gestionándolas")

        # Guardar métricas finales
        await self._save_periodic_metrics()

        # Log final
        if self.database:
            await self.database.log_event('INFO', 'SYSTEM', 'Sistema apagado correctamente - Posiciones preservadas')

        print("✅ Sistema apagado correctamente - Posiciones preservadas")
        print("💡 Las posiciones siguen generando PnL en Binance")

    async def _detect_market_regime_robust(self, market_data: Dict[str, List[float]]) -> Tuple[str, float]:
        """
        🔍 DETECCIÓN ROBUSTA DE RÉGIMEN DE MERCADO - VERSIÓN SENSIBLE A BAJISTAS
        Sistema optimizado para detectar movimientos bajistas con mayor sensibilidad

        SENSIBLE: Ajustado para detectar mejor los movimientos bajistas persistentes
        """
        try:
            print(f"🔍 Detectando régimen de mercado robusto (versión sensible a bajistas)...")

            # 1. ANÁLISIS MULTI-PAR
            pair_regimes = {}
            total_signals = 0
            bullish_signals = 0
            bearish_signals = 0

            for symbol, prices in market_data.items():
                if len(prices) < 50:  # Necesitamos datos suficientes
                    continue

                # Convertir a pandas para análisis técnico
                df = pd.DataFrame({'close': prices})

                # === INDICADORES POR PAR ===

                # 1. Momentum multitimeframe (UMBRALES MUY RELAJADOS)
                df['momentum_1h'] = df['close'].pct_change(12)   # 12 * 5m = 1h
                df['momentum_4h'] = df['close'].pct_change(48)   # 48 * 5m = 4h
                df['momentum_12h'] = df['close'].pct_change(144) # 144 * 5m = 12h
                df['momentum_24h'] = df['close'].pct_change(288) # 288 * 5m = 24h

                # 2. Medias móviles
                df['sma_20'] = df['close'].rolling(20).mean()
                df['sma_50'] = df['close'].rolling(50).mean()
                df['ema_20'] = df['close'].ewm(span=20).mean()
                df['ema_50'] = df['close'].ewm(span=50).mean()

                # 3. Trend strength
                df['ma_trend'] = (df['close'] - df['sma_20']) / df['sma_20']
                df['ma_direction'] = df['sma_20'] > df['sma_50']
                df['ema_trend'] = (df['close'] - df['ema_20']) / df['ema_20']

                # 4. Volatilidad relativa
                df['returns'] = df['close'].pct_change()
                df['volatility'] = df['returns'].rolling(20).std()
                df['vol_percentile'] = df['volatility'].rolling(100).rank(pct=True)

                # 5. RSI para momentum
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / loss
                df['rsi'] = 100 - (100 / (1 + rs))

                # 6. Análisis de tendencia reciente (últimos 2 días)
                df['recent_trend_2d'] = df['close'].pct_change(576)  # 576 * 5m = 48h
                df['recent_slope'] = df['close'].rolling(144).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 144 else 0, raw=True)

                # === CLASIFICACIÓN POR PAR (MUY RELAJADA) ===
                latest = df.iloc[-1]

                # Contar señales bullish y bearish
                bullish_count = 0
                bearish_count = 0

                # ✅ SENSIBLE A BAJISTAS: Umbrales más sensibles para detectar movimientos bajistas
                # Momentum 4h (más sensible: de 2.2% a 1.5%)
                if not pd.isna(latest['momentum_4h']):
                    if latest['momentum_4h'] > 0.015:  # +1.5% (más sensible)
                        bullish_count += 2
                    elif latest['momentum_4h'] < -0.015:  # -1.5% (más sensible)
                        bearish_count += 3  # Más peso para señales bajistas

                # Momentum 12h (más sensible: de 4.5% a 2.5%)
                if not pd.isna(latest['momentum_12h']):
                    if latest['momentum_12h'] > 0.025:  # +2.5% (más sensible)
                        bullish_count += 3
                    elif latest['momentum_12h'] < -0.025:  # -2.5% (más sensible)
                        bearish_count += 4  # Más peso para señales bajistas

                # Momentum 24h (más sensible: de 5.5% a 3.0%)
                if not pd.isna(latest['momentum_24h']):
                    if latest['momentum_24h'] > 0.030:  # +3.0% (más sensible)
                        bullish_count += 4
                    elif latest['momentum_24h'] < -0.030:  # -3.0% (más sensible)
                        bearish_count += 5  # Más peso para señales bajistas

                # Trend reciente de 2 días (más sensible: de 6.5% a 3.5%)
                if not pd.isna(latest['recent_trend_2d']):
                    if latest['recent_trend_2d'] > 0.035:  # +3.5% (más sensible)
                        bullish_count += 3
                    elif latest['recent_trend_2d'] < -0.035:  # -3.5% (más sensible)
                        bearish_count += 5  # Más peso para señales bajistas

                # Señales de MA (más sensibles)
                if not pd.isna(latest['ma_trend']):
                    if latest['ma_trend'] > 0.008 and latest['ma_direction']:  # +0.8% (más sensible)
                        bullish_count += 2
                    elif latest['ma_trend'] < -0.008 and not latest['ma_direction']:  # -0.8% (más sensible)
                        bearish_count += 3  # Más peso para señales bajistas

                # EMA trend adicional (más sensible)
                if not pd.isna(latest['ema_trend']):
                    if latest['ema_trend'] < -0.015:  # -1.5% (más sensible)
                        bearish_count += 3  # Más peso para señales bajistas

                # RSI con contexto de momentum (más sensible)
                if not pd.isna(latest['rsi']):
                    if latest['rsi'] > 70 and not pd.isna(latest['momentum_1h']) and latest['momentum_1h'] > 0:
                        bullish_count += 1
                    elif latest['rsi'] < 30 and not pd.isna(latest['momentum_1h']) and latest['momentum_1h'] < 0:
                        bearish_count += 2  # Más peso para señales bajistas
                    # RSI en zona de venta pero no extrema (más sensible)
                    elif 40 < latest['rsi'] < 60 and not pd.isna(latest['momentum_4h']) and latest['momentum_4h'] < -0.008:
                        bearish_count += 2  # Momentum bajista con RSI neutral (más sensible)

                # Pendiente reciente negativa (más sensible)
                if not pd.isna(latest['recent_slope']) and latest['recent_slope'] < -0.8:
                    bearish_count += 3  # Más peso para señales bajistas

                # ✅ CLASIFICACIÓN SENSIBLE A BAJISTAS: Más sensible para detectar movimientos bajistas
                if bearish_count > bullish_count + 2:  # ✅ SENSIBLE: Requiere 2+ diferencia (más sensible)
                    pair_regime = 'BEARISH'
                    bearish_signals += bearish_count
                elif bullish_count > bearish_count + 3:  # ✅ SENSIBLE: Requiere 3+ diferencia (menos sensible para alcistas)
                    pair_regime = 'BULLISH'
                    bullish_signals += bullish_count
                else:
                    pair_regime = 'NEUTRAL'

                pair_regimes[symbol] = {
                    'regime': pair_regime,
                    'bullish_signals': bullish_count,
                    'bearish_signals': bearish_count,
                    'momentum_4h': latest['momentum_4h'] if not pd.isna(latest['momentum_4h']) else 0,
                    'momentum_12h': latest['momentum_12h'] if not pd.isna(latest['momentum_12h']) else 0,
                    'momentum_24h': latest['momentum_24h'] if not pd.isna(latest['momentum_24h']) else 0,
                    'recent_trend_2d': latest['recent_trend_2d'] if not pd.isna(latest['recent_trend_2d']) else 0,
                    'ma_trend': latest['ma_trend'] if not pd.isna(latest['ma_trend']) else 0,
                    'rsi': latest['rsi'] if not pd.isna(latest['rsi']) else 50
                }

                total_signals += bullish_count + bearish_count

                print(f"   📊 {symbol}: {pair_regime} (Bull: {bullish_count}, Bear: {bearish_count})")
                # Mostrar valores reales
                mom_4h_display = latest['momentum_4h'] if not pd.isna(latest['momentum_4h']) else 0.0
                mom_24h_display = latest['momentum_24h'] if not pd.isna(latest['momentum_24h']) else 0.0
                trend_2d_display = latest['recent_trend_2d'] if not pd.isna(latest['recent_trend_2d']) else 0.0
                print(f"      📈 Mom4h: {mom_4h_display:.3f} | Mom24h: {mom_24h_display:.3f} | Trend2d: {trend_2d_display:.3f}")

            # 2. ANÁLISIS AGREGADO DEL MERCADO

            # Contar regímenes por par
            regime_votes = {'BULLISH': 0, 'BEARISH': 0, 'NEUTRAL': 0}
            for data in pair_regimes.values():
                regime_votes[data['regime']] += 1

            total_pairs = len(pair_regimes)
            if total_pairs == 0:
                return 'NEUTRAL', 0.5

            # 3. MÉTRICAS AGREGADAS DE CONFIANZA

            # Ratio de señales
            if total_signals > 0:
                bullish_ratio = bullish_signals / total_signals
                bearish_ratio = bearish_signals / total_signals
            else:
                bullish_ratio = bearish_ratio = 0

            # Consenso entre pares
            max_votes = max(regime_votes.values())
            consensus_strength = max_votes / total_pairs

            # 4. ✅ CLASIFICACIÓN FINAL SENSIBLE A BAJISTAS

            # ✅ SENSIBLE: Más fácil detectar BEARISH
            if regime_votes['BEARISH'] >= total_pairs * 0.6:  # 60% de pares bearish
                final_regime = 'BEARISH'
                confidence = 0.8  # ✅ SENSIBLE: Confianza alta para BEARISH
                print(f"   🔴 MAYORÍA BEARISH: {regime_votes['BEARISH']}/{total_pairs} pares → BEARISH detectado")

            # Umbrales SENSIBLES para detectar mejor movimientos bajistas
            elif (regime_votes['BEARISH'] >= regime_votes['BULLISH'] and
                  bearish_ratio > 0.4 and  # ✅ SENSIBLE: De 0.5 a 0.4 (más sensible)
                  consensus_strength > 0.4):  # ✅ SENSIBLE: De 0.5 a 0.4 (más sensible)
                final_regime = 'BEARISH'
                confidence = min(0.8, 0.5 + (bearish_ratio - 0.4) * 0.8 + (consensus_strength - 0.4) * 0.6)

            elif (regime_votes['BULLISH'] > regime_votes['BEARISH'] + 2 and
                  bullish_ratio > 0.6 and  # ✅ SENSIBLE: De 0.55 a 0.6 (menos sensible para alcistas)
                  consensus_strength > 0.6):  # ✅ SENSIBLE: De 0.5 a 0.6 (menos sensible para alcistas)
                final_regime = 'BULLISH'
                confidence = min(0.8, 0.5 + (bullish_ratio - 0.6) + (consensus_strength - 0.6))

            else:
                final_regime = 'NEUTRAL'
                confidence = 0.5 + (1 - consensus_strength) * 0.3  # Mayor incertidumbre

            # 5. ✅ OVERRIDE SENSIBLE PARA MERCADOS CLARAMENTE BAJISTAS
            # Si la mayoría de pares muestran tendencia bajista de 2 días
            valid_trends = [data['recent_trend_2d'] for data in pair_regimes.values() if data['recent_trend_2d'] != 0]
            if valid_trends:
                avg_trend_2d = np.mean(valid_trends)
                # ✅ SENSIBLE: Más fácil detectar tendencias bajistas
                if avg_trend_2d < -0.03 and final_regime == 'NEUTRAL':  # ✅ SENSIBLE: De -6% a -3% (más sensible)
                    final_regime = 'BEARISH'
                    confidence = 0.7  # ✅ SENSIBLE: De 0.65 a 0.7 (más confianza)
                    print(f"   🔴 OVERRIDE: Tendencia 2d promedio {avg_trend_2d:.3f} < -3% → BEARISH forzado")
                # ✅ SENSIBLE: Override adicional para tendencias muy bajistas
                elif avg_trend_2d < -0.05:  # ✅ SENSIBLE: -5% o más
                    final_regime = 'BEARISH'
                    confidence = 0.75  # ✅ SENSIBLE: Alta confianza
                    print(f"   🔴 OVERRIDE FUERTE: Tendencia 2d promedio {avg_trend_2d:.3f} < -5% → BEARISH forzado")
            else:
                avg_trend_2d = 0.0

            # 6. LOGGING DETALLADO
            print(f"   📊 Votos por régimen: {regime_votes}")
            print(f"   📊 Ratio señales: Bull {bullish_ratio:.2f}, Bear {bearish_ratio:.2f}")
            print(f"   📊 Consenso: {consensus_strength:.2f}")
            print(f"   📊 Tendencia 2d promedio: {avg_trend_2d:.3f}")
            print(f"   🎯 RÉGIMEN FINAL: {final_regime} (Confianza: {confidence:.2f})")

            # Mostrar detalles por par
            for symbol, data in pair_regimes.items():
                print(f"     {symbol}: {data['regime']} | Mom4h: {data['momentum_4h']:.3f} | "
                      f"Mom24h: {data['momentum_24h']:.3f} | Trend2d: {data['recent_trend_2d']:.3f} | "
                      f"MA: {data['ma_trend']:.3f} | RSI: {data['rsi']:.1f}")

            return final_regime, confidence

        except Exception as e:
            print(f"❌ Error en detección de régimen: {e}")
            return 'NEUTRAL', 0.5

    async def _execute_position_with_diversification_size(self, symbol: str, side: str, confidence: float,
                                                        current_price: float, position_size_usd: float) -> Optional:
        """💰 Ejecutar posición con tamaño específico calculado por diversificación"""
        try:
            print(f"🎯 EJECUTANDO POSICIÓN CON TAMAÑO DE DIVERSIFICACIÓN:")
            print(f"   📊 {symbol}: {side}")
            print(f"   💰 Tamaño USD: ${position_size_usd:.2f}")
            print(f"   💲 Precio: ${current_price:.4f}")

            # Calcular cantidad basada en el tamaño USD especificado por diversificación
            quantity = position_size_usd / current_price
            print(f"   🔢 Cantidad calculada: {quantity:.8f}")

            # Usar el método interno del risk manager para ejecutar la orden real
            order_result = await self.risk_manager._execute_real_order(symbol, side, quantity)

            if not order_result or 'orderId' not in order_result:
                print(f"❌ Falló la ejecución de la orden real para {symbol}. No se abre posición.")
                return None

            real_entry_price = float(order_result.get('fills', [{}])[0].get('price', current_price))
            real_quantity = float(order_result.get('executedQty', quantity))
            order_id = str(order_result['orderId'])

            print(f"🎉 Orden real ejecutada para {symbol}: ID {order_id}")
            print(f"   - Precio Real: ${real_entry_price:.4f}, Cantidad Real: {real_quantity:.6f}")

            # Crear posición usando la misma estructura que AdvancedRiskManager
            from advanced_risk_manager import Position
            from datetime import timezone

            position = Position(
                symbol=symbol,
                side=side,
                quantity=real_quantity,
                entry_price=real_entry_price,
                current_price=real_entry_price,
                entry_time=datetime.now(timezone.utc),
                stop_loss=real_entry_price * (1 - self.risk_manager.limits.stop_loss_percent / 100),
                take_profit=real_entry_price * (1 + self.risk_manager.limits.take_profit_percent / 100),
                trade_id=order_id,
                order_id=order_id,
                is_active=True
            )

            # Registrar la posición en el risk manager
            self.risk_manager.active_positions[order_id] = position

            # Actualizar balance del risk manager
            await self.risk_manager.update_balance(self.risk_manager.current_balance - (real_quantity * real_entry_price))

            print(f"✅ POSICIÓN CREADA CON TAMAÑO DE DIVERSIFICACIÓN: {symbol} - Order ID: {order_id}")
            print(f"   💰 Valor real invertido: ${real_quantity * real_entry_price:.2f}")

            return position

        except Exception as e:
            print(f"❌ Error ejecutando posición con tamaño de diversificación para {symbol}: {e}")
            return None

    def _evaluate_reversal_with_consecutive_signals(self, symbol: str, signal: str, confidence: float) -> tuple:
        """
        🔄 SISTEMA DE REVERSIÓN MEJORADO CON SEÑALES CONSECUTIVAS
        ---
        Evalúa si hay suficientes señales consecutivas para ejecutar reversión

        Configuración por símbolo:
        - TODOS LOS SÍMBOLOS: Requieren 20 señales consecutivas de reversión (MUY ESTRICTO)
        - Timeout: 1.5 minutos entre señales para mantener el tracking activo
        - Sistema diseñado para máxima estabilidad y evitar reversiones prematuras

        Returns:
            tuple: (should_execute_reversal: bool, reason: str)
        """

        try:
            current_time = datetime.now()

            # Obtener configuración para este símbolo
            config = self.reversal_config.get(symbol, {
                'required_consecutive_signals': 20,  # ✅ MUY ESTRICTO: 20 señales por defecto
                'timeout_minutes': 5,  # ✅ MANTENER: 5 minutos por defecto
                'min_confidence_per_signal': 80.0,
                'cumulative_confidence_threshold': 85.0
            })

            # Inicializar tracking si no existe
            if symbol not in self.reversal_tracking:
                self.reversal_tracking[symbol] = {
                    'consecutive_signals': [],
                    'last_signal_time': None,
                    'current_signal_direction': None,
                    'started_tracking': None
                }

            tracking = self.reversal_tracking[symbol]

            # ✅ LIMPIEZA AUTOMÁTICA: Verificar timeout (1.5 minutos)
            if tracking['last_signal_time']:
                time_since_last = (current_time - tracking['last_signal_time']).total_seconds() / 60
                if time_since_last > config['timeout_minutes']:
                    print(f"    🧹 LIMPIEZA AUTOMÁTICA: Timeout de {config['timeout_minutes']}min excedido para {symbol}")
                    tracking['consecutive_signals'] = []
                    tracking['current_signal_direction'] = None
                    tracking['started_tracking'] = None

            # ✅ LIMPIEZA AUTOMÁTICA: Verificar cambio de dirección
            if tracking['current_signal_direction'] and tracking['current_signal_direction'] != signal:
                print(f"    🔄 LIMPIEZA AUTOMÁTICA: Cambio de dirección {tracking['current_signal_direction']} → {signal} para {symbol}")
                tracking['consecutive_signals'] = []
                tracking['current_signal_direction'] = signal
                tracking['started_tracking'] = current_time
            elif not tracking['current_signal_direction']:
                tracking['current_signal_direction'] = signal
                tracking['started_tracking'] = current_time

            # Verificar confianza mínima por señal
            if confidence < config['min_confidence_per_signal']:
                return False, f"Confianza insuficiente {confidence:.1f}% < {config['min_confidence_per_signal']:.1f}%"

            # ✅ NUEVO: Verificar intervalo mínimo entre señales válidas
            min_interval = config.get('min_interval_between_signals_minutes', 10)
            if tracking['consecutive_signals']:
                last_valid_signal = tracking['consecutive_signals'][-1]
                time_since_last_valid = (current_time - last_valid_signal['timestamp']).total_seconds() / 60

                if time_since_last_valid < min_interval:
                    print(f"    ⏸️ SEÑAL RECHAZADA: Intervalo {time_since_last_valid:.1f}min < {min_interval}min requerido")
                    return False, f"Intervalo insuficiente: {time_since_last_valid:.1f}min < {min_interval}min"

            # Agregar nueva señal (solo si pasa todas las validaciones)
            signal_data = {
                'signal': signal,
                'confidence': confidence,
                'timestamp': current_time
            }

            tracking['consecutive_signals'].append(signal_data)
            tracking['last_signal_time'] = current_time

            print(f"    ✅ SEÑAL VÁLIDA ACEPTADA: {signal} {confidence:.1f}% (intervalo OK)")

            # Mantener solo las señales necesarias (+ algunas extra para análisis)
            max_signals_to_keep = config['required_consecutive_signals'] + 2
            if len(tracking['consecutive_signals']) > max_signals_to_keep:
                tracking['consecutive_signals'] = tracking['consecutive_signals'][-max_signals_to_keep:]

            current_count = len(tracking['consecutive_signals'])
            required_count = config['required_consecutive_signals']

            print(f"    📊 TRACKING REVERSIÓN {symbol}: {current_count}/{required_count} señales {signal}")
            print(f"        💫 Confianza actual: {confidence:.1f}% (mín: {config['min_confidence_per_signal']:.1f}%)")

            # Verificar si tenemos suficientes señales
            if current_count >= required_count:
                # Verificar consistencia de señales (todas del mismo tipo)
                recent_signals = tracking['consecutive_signals'][-required_count:]
                all_same_signal = all(s['signal'] == signal for s in recent_signals)

                if not all_same_signal:
                    return False, f"Señales inconsistentes en las últimas {required_count}"

                # Calcular confianza acumulada (promedio ponderado, más peso a señales recientes)
                total_weight = 0
                weighted_confidence = 0

                for i, signal_entry in enumerate(recent_signals):
                    weight = i + 1  # Peso creciente para señales más recientes
                    weighted_confidence += signal_entry['confidence'] * weight
                    total_weight += weight

                cumulative_confidence = weighted_confidence / total_weight if total_weight > 0 else 0

                # Verificar umbral de confianza acumulada
                if cumulative_confidence >= config['cumulative_confidence_threshold']:
                    # Calcular tiempo transcurrido desde que se inició el tracking
                    tracking_duration = (current_time - tracking['started_tracking']).total_seconds() / 60

                    print(f"    ✅ REVERSIÓN APROBADA para {symbol}:")
                    print(f"        🎯 Señales consecutivas: {current_count}/{required_count}")
                    print(f"        💫 Confianza acumulada: {cumulative_confidence:.1f}% (req: {config['cumulative_confidence_threshold']:.1f}%)")
                    print(f"        ⏱️ Tiempo de tracking: {tracking_duration:.1f} minutos")

                    # Limpiar tracking después de ejecutar
                    tracking['consecutive_signals'] = []
                    tracking['current_signal_direction'] = None
                    tracking['started_tracking'] = None

                    return True, f"REVERSIÓN_CONSECUTIVA_{required_count}_SEÑALES"
                else:
                    return False, f"Confianza acumulada insuficiente {cumulative_confidence:.1f}% < {config['cumulative_confidence_threshold']:.1f}%"
            else:
                return False, f"Necesita {required_count - current_count} señales más"

        except Exception as e:
            print(f"❌ Error en evaluación de reversión consecutiva para {symbol}: {e}")
            return False, f"Error en evaluación: {e}"

    def _display_reversal_tracking_status(self) -> str:
        """
        📊 MOSTRAR ESTADO ACTUAL DEL TRACKING DE REVERSIÓN
        ---
        Genera reporte del estado de tracking para todos los símbolos
        Incluido en reportes TCN para monitoreo

        Returns:
            str: Reporte formateado del estado de tracking
        """

        try:
            if not self.reversal_tracking:
                return "\n🔄 **TRACKING DE REVERSIÓN:** Sin tracking activo\n"

            status_report = "\n🔄 **TRACKING DE REVERSIÓN:**\n"

            active_tracking_count = 0

            for symbol, tracking in self.reversal_tracking.items():
                if not tracking['consecutive_signals']:
                    continue

                active_tracking_count += 1
                config = self.reversal_config.get(symbol, {})
                required_count = config.get('required_consecutive_signals', 20)  # ✅ MUY ESTRICTO: 20 señales por defecto
                current_count = len(tracking['consecutive_signals'])

                # Calcular tiempo desde primera señal
                if tracking['started_tracking']:
                    tracking_duration = (datetime.now() - tracking['started_tracking']).total_seconds() / 60
                else:
                    tracking_duration = 0

                # Calcular tiempo desde última señal
                if tracking['last_signal_time']:
                    time_since_last = (datetime.now() - tracking['last_signal_time']).total_seconds() / 60
                else:
                    time_since_last = 0

                # Emoji según progreso
                progress_percent = (current_count / required_count) * 100
                if progress_percent >= 100:
                    progress_emoji = "🟢"
                elif progress_percent >= 50:
                    progress_emoji = "🟡"
                else:
                    progress_emoji = "🔵"

                # Información de la última señal
                last_signal = tracking['consecutive_signals'][-1] if tracking['consecutive_signals'] else None
                last_confidence = last_signal['confidence'] if last_signal else 0

                status_report += f"{progress_emoji} **{symbol}**: {tracking['current_signal_direction']} "
                status_report += f"{current_count}/{required_count} señales "
                status_report += f"(Última: {last_confidence:.1f}%)\n"
                status_report += f"   ⏱️ Tracking: {tracking_duration:.1f}min | "
                status_report += f"Última señal: {time_since_last:.1f}min atrás\n"

                # Mostrar progreso detallado
                if tracking['consecutive_signals']:
                    confidences = [s['confidence'] for s in tracking['consecutive_signals']]
                    avg_confidence = sum(confidences) / len(confidences)
                    status_report += f"   📊 Conf. promedio: {avg_confidence:.1f}% | "
                    status_report += f"Conf. mín/máx: {min(confidences):.1f}%/{max(confidences):.1f}%\n"

            if active_tracking_count == 0:
                status_report += "📊 Sin tracking activo en este momento\n"
            else:
                status_report += f"\n📈 **Total activo:** {active_tracking_count} símbolo(s) en tracking\n"

            # Agregar configuración de límites
            status_report += "\n⚙️ **CONFIGURACIÓN DE REVERSIÓN:**\n"
            for symbol, config in self.reversal_config.items():
                status_report += f"   {symbol}: {config['required_consecutive_signals']} señales, "
                status_report += f"intervalo {config.get('min_interval_between_signals_minutes', 10)}min, "
                status_report += f"timeout {config['timeout_minutes']}min, "
                status_report += f"conf. mín {config['min_confidence_per_signal']:.0f}%\n"

            return status_report

        except Exception as e:
            print(f"❌ Error generando reporte de tracking de reversión: {e}")
            return f"\n🔄 **TRACKING DE REVERSIÓN:** Error generando reporte: {e}\n"

    async def _generate_sync_diagnosis_section(self) -> str:
        """
        🔍 Generar sección de diagnóstico de sincronización para el reporte TCN
        """
        try:
            if not self.portfolio_manager:
                return "\n🔍 **SINCRONIZACIÓN:** Portfolio Manager no disponible\n"

            risk_manager_count = len(self.active_positions)
            portfolio_count = len(self.portfolio_manager.position_registry)
            
            sync_section = f"""

🔍 **DIAGNÓSTICO DE SINCRONIZACIÓN**
📊 **Risk Manager:** {risk_manager_count} posiciones
📊 **Portfolio Manager:** {portfolio_count} posiciones
"""

            if risk_manager_count == portfolio_count:
                sync_section += "✅ **Estado:** Sincronización correcta\n"
            else:
                sync_section += f"❌ **Estado:** DESINCRONIZACIÓN detectada ({abs(risk_manager_count - portfolio_count)} diferencia)\n"
                sync_section += "🔧 **Acción:** Se requiere limpieza automática\n"

            # Análisis por símbolo (solo si hay desincronización)
            if risk_manager_count != portfolio_count:
                sync_section += "\n📊 **ANÁLISIS POR SÍMBOLO:**\n"
                
                # Agrupar posiciones por símbolo
                portfolio_by_symbol = {}
                for pos in self.portfolio_manager.position_registry.values():
                    if pos.symbol not in portfolio_by_symbol:
                        portfolio_by_symbol[pos.symbol] = []
                    portfolio_by_symbol[pos.symbol].append(pos)
                
                risk_by_symbol = {}
                for pos in self.active_positions.values():
                    if pos.symbol not in risk_by_symbol:
                        risk_by_symbol[pos.symbol] = []
                    risk_by_symbol[pos.symbol].append(pos)
                
                # Mostrar diferencias
                all_symbols = set(portfolio_by_symbol.keys()) | set(risk_by_symbol.keys())
                for symbol in sorted(all_symbols):
                    portfolio_count = len(portfolio_by_symbol.get(symbol, []))
                    risk_count = len(risk_by_symbol.get(symbol, []))
                    
                    if portfolio_count != risk_count:
                        sync_section += f"   ❌ {symbol}: Portfolio={portfolio_count}, Risk={risk_count}\n"
                    else:
                        sync_section += f"   ✅ {symbol}: {portfolio_count} posiciones\n"

            return sync_section

        except Exception as e:
            print(f"⚠️ Error generando sección de diagnóstico de sincronización: {e}")
            return f"\n🔍 **SINCRONIZACIÓN:** Error generando diagnóstico: {e}\n"

    async def _cleanup_stale_reversal_tracking(self):
        """
        🧹 LIMPIEZA AUTOMÁTICA DE TRACKING COLGADO
        ---
        Se ejecuta periódicamente (cada 5 minutos) para limpiar trackings que han
        excedido el timeout, independientemente de si hay nuevas señales o no.

        Soluciona el problema de trackings "colgados" cuando no hay señales nuevas.
        """
        try:
            # Solo ejecutar cada 5 minutos
            current_time = datetime.now()
            if not hasattr(self, '_last_cleanup_time'):
                self._last_cleanup_time = current_time
                return

            time_since_cleanup = (current_time - self._last_cleanup_time).total_seconds() / 60
            if time_since_cleanup < 1.5:  # 1.5 minutos
                return

            self._last_cleanup_time = current_time

            if not self.reversal_tracking:
                return

            cleaned_count = 0

            for symbol in list(self.reversal_tracking.keys()):
                tracking = self.reversal_tracking[symbol]

                # Solo limpiar si hay tracking activo
                if not tracking.get('consecutive_signals') or not tracking.get('last_signal_time'):
                    continue

                # Verificar timeout
                config = self.reversal_config.get(symbol, {'timeout_minutes': 1.5})  # ✅ AJUSTE: 1.5 minutos por defecto
                timeout_minutes = config.get('timeout_minutes', 1.5)  # ✅ AJUSTE: 1.5 minutos por defecto

                time_since_last = (current_time - tracking['last_signal_time']).total_seconds() / 60

                if time_since_last > timeout_minutes:
                    print(f"🧹 LIMPIEZA AUTOMÁTICA PERIÓDICA: Tracking colgado limpiado para {symbol}")
                    print(f"   ⏰ Tiempo transcurrido: {time_since_last:.1f}min > {timeout_minutes}min")
                    print(f"   📊 Señales perdidas: {len(tracking['consecutive_signals'])}")

                    # Limpiar tracking
                    self.reversal_tracking[symbol] = {
                        'consecutive_signals': [],
                        'last_signal_time': None,
                        'current_signal_direction': None,
                        'started_tracking': None
                    }
                    cleaned_count += 1

            if cleaned_count > 0:
                print(f"✅ Limpieza periódica completada: {cleaned_count} tracking(s) limpiado(s)")

        except Exception as e:
            print(f"❌ Error en limpieza automática de tracking: {e}")

    def register_position_close(self, symbol: str, close_type: str, pnl_percent: float):
        """
        📝 Registrar el cierre de una posición para aplicar cooldown post-cierre
        ---
        Args:
            symbol: Símbolo de la posición cerrada
            close_type: Tipo de cierre detectado por el sistema
            pnl_percent: Porcentaje de PnL al cerrar
        """
        from datetime import datetime, timedelta
        
        current_time = datetime.now()
        
        # ✅ MEJORADO: Detección más robusta del tipo de cierre
        print(f"🔍 ANALIZANDO TIPO DE CIERRE: {symbol}")
        print(f"   📊 close_type recibido: '{close_type}'")
        print(f"   📏 Longitud del close_type: {len(close_type)}")
        print(f"   🔤 close_type uppercase: '{close_type.upper()}'")
        print(f"   💰 PnL: {pnl_percent:.2f}%")
        
        # Mapear diferentes posibles nombres de cierres
        trailing_stop_keywords = ['TRAILING_STOP', 'trailing_stop', 'trailing stop', 'TRAIL', 'trail']
        reversal_keywords = ['REVERSIÓN_CONSECUTIVA', 'reversal', 'reversión', 'REVERSAL']
        
        print(f"   🔍 Keywords de trailing stop: {trailing_stop_keywords}")
        print(f"   🔍 Keywords de reversión: {reversal_keywords}")
        print(f"   🔍 Verificando si '{close_type.upper()}' contiene alguna keyword...")
        
        trailing_matches = []
        reversal_matches = []
        
        for keyword in trailing_stop_keywords:
            if keyword.upper() in close_type.upper():
                trailing_matches.append(keyword)
                
        for keyword in reversal_keywords:
            if keyword.upper() in close_type.upper():
                reversal_matches.append(keyword)
                
        print(f"   ✅ Keywords trailing stop encontradas: {trailing_matches}")
        print(f"   ✅ Keywords reversión encontradas: {reversal_matches}")
        
        is_trailing_stop = len(trailing_matches) > 0
        is_reversal = len(reversal_matches) > 0
        
        if is_reversal:
            actual_close_type = 'reversal_close'
            print(f"   ✅ DETECTADO: Cierre por REVERSIÓN CONSECUTIVA")
            print(f"   🎯 Será mapeado a: {actual_close_type}")
        elif is_trailing_stop:
            actual_close_type = 'trailing_stop_close'
            print(f"   ✅ DETECTADO: Cierre por TRAILING STOP")
            print(f"   🎯 Será mapeado a: {actual_close_type}")
        elif pnl_percent > 0:
            actual_close_type = 'profit_close'
            print(f"   ✅ DETECTADO: Cierre con GANANCIA")
            print(f"   🎯 Será mapeado a: {actual_close_type}")
        else:
            actual_close_type = 'loss_close'
            print(f"   ✅ DETECTADO: Cierre con PÉRDIDA")
            print(f"   🎯 Será mapeado a: {actual_close_type}")
            
        print(f"   📋 RESULTADO FINAL: close_type='{close_type}' → actual_close_type='{actual_close_type}'")
        
        # Registrar el cierre
        self.recent_closes[symbol] = {
            'last_close_time': current_time,
            'close_type': actual_close_type,
            'pnl_percent': pnl_percent,
            'original_close_type': close_type
        }
        
        # Aplicar cooldown específico
        print(f"   🔍 Buscando cooldown para: symbol='{symbol}', close_type='{actual_close_type}'")
        symbol_cooldowns = self.post_close_cooldown.get(symbol, {})
        print(f"   📊 Cooldowns disponibles para {symbol}: {symbol_cooldowns}")
        
        cooldown_minutes = symbol_cooldowns.get(actual_close_type, 5)
        print(f"   ⏰ Cooldown obtenido: {cooldown_minutes} minutos")
        
        if cooldown_minutes == 5:
            print(f"   ⚠️ USANDO COOLDOWN POR DEFECTO (5min) - Verifica configuración")
        
        print(f"🛡️ COOLDOWN POST-CIERRE REGISTRADO: {symbol}")
        print(f"   🕐 Hora: {current_time.strftime('%H:%M:%S')}")
        print(f"   🎯 Tipo final: {actual_close_type}")
        print(f"   📝 Razón original: {close_type}")
        print(f"   💰 PnL: {pnl_percent:.2f}%")
        print(f"   ⏰ Cooldown aplicado: {cooldown_minutes} minutos")
        print(f"   🚫 Próxima compra permitida: {(current_time + timedelta(minutes=cooldown_minutes)).strftime('%H:%M:%S')}")
        
        return cooldown_minutes

    def is_in_post_close_cooldown(self, symbol: str) -> tuple[bool, str]:
        """
        🔍 Verificar si un símbolo está en cooldown post-cierre
        ---
        Returns:
            (is_in_cooldown, remaining_time_str)
        """
        if symbol not in self.recent_closes:
            print(f"🔍 COOLDOWN CHECK: {symbol} - No hay cierres recientes registrados")
            return False, ""
        
        close_info = self.recent_closes[symbol]
        close_time = close_info['last_close_time']
        close_type = close_info['close_type']
        original_close_type = close_info.get('original_close_type', 'N/A')
        pnl_percent = close_info.get('pnl_percent', 0.0)
        
        # Obtener cooldown específico para este tipo de cierre
        cooldown_minutes = self.post_close_cooldown.get(symbol, {}).get(close_type, 5)
        
        # Calcular tiempo transcurrido
        from datetime import datetime
        current_time = datetime.now()
        elapsed_minutes = (current_time - close_time).total_seconds() / 60
        
        print(f"🔍 COOLDOWN CHECK: {symbol}")
        print(f"   🕐 Cerrado: {close_time.strftime('%H:%M:%S')}")
        print(f"   📊 Tipo: {close_type} (original: {original_close_type})")
        print(f"   💰 PnL: {pnl_percent:.2f}%")
        print(f"   ⏰ Cooldown: {cooldown_minutes}m")
        print(f"   ⌛ Transcurrido: {elapsed_minutes:.1f}m")
        
        if elapsed_minutes < cooldown_minutes:
            remaining_minutes = cooldown_minutes - elapsed_minutes
            if remaining_minutes >= 1:
                remaining_time_str = f"{remaining_minutes:.0f}m"
            else:
                remaining_seconds = remaining_minutes * 60
                remaining_time_str = f"{remaining_seconds:.0f}s"
            
            print(f"   🚫 COOLDOWN ACTIVO - Restante: {remaining_time_str}")
            return True, remaining_time_str
        
        print(f"   ✅ COOLDOWN EXPIRADO - Puede operar")
        return False, ""

    def cleanup_expired_post_close_cooldowns(self):
        """
        🧹 Limpiar cooldowns post-cierre expirados automáticamente
        """
        from datetime import datetime
        
        current_time = datetime.now()
        expired_symbols = []
        
        for symbol, close_info in self.recent_closes.items():
            close_time = close_info['last_close_time']
            close_type = close_info['close_type']
                        
            # Obtener cooldown específico para este tipo de cierre
            cooldown_minutes = self.post_close_cooldown.get(symbol, {}).get(close_type, 5)
            
            # Calcular tiempo transcurrido
            elapsed_minutes = (current_time - close_time).total_seconds() / 60
            
            if elapsed_minutes >= cooldown_minutes:
                expired_symbols.append(symbol)
        
        # Limpiar cooldowns expirados
        for symbol in expired_symbols:
            close_info = self.recent_closes.pop(symbol)
            print(f"🧹 COOLDOWN POST-CIERRE EXPIRADO: {symbol} - Tipo: {close_info['close_type']} - PnL: {close_info['pnl_percent']:.2f}%")
        
        if expired_symbols:
            print(f"✅ {len(expired_symbols)} cooldown(s) post-cierre limpiado(s) automáticamente")

    async def _cleanup_desynchronized_positions(self, symbol: str):
        """
        🔧 LIMPIAR POSICIONES DESINCRONIZADAS ENTRE CONTADORES
        ---
        Soluciona el problema de contadores diferentes entre:
        - self.active_positions (Risk Manager)
        - self.portfolio_manager.position_registry (Portfolio Manager)
        """
        try:
            print(f"🔧 LIMPIEZA DE DESINCRONIZACIÓN para {symbol}...")
            
            # Obtener posiciones reales del portfolio manager
            portfolio_positions = [pos for pos in self.portfolio_manager.position_registry.values() if pos.symbol == symbol]
            portfolio_order_ids = {pos.order_id for pos in portfolio_positions if pos.order_id}
            
            # Obtener posiciones del risk manager
            risk_manager_positions = [pos for pos in self.active_positions.values() if pos.symbol == symbol]
            risk_manager_order_ids = {pos.order_id for pos in risk_manager_positions if pos.order_id}
            
            print(f"   📊 Portfolio Manager: {len(portfolio_positions)} posiciones con IDs: {portfolio_order_ids}")
            print(f"   📊 Risk Manager: {len(risk_manager_positions)} posiciones con IDs: {risk_manager_order_ids}")
            
            # Encontrar posiciones fantasma en risk manager
            ghost_order_ids = risk_manager_order_ids - portfolio_order_ids
            
            if ghost_order_ids:
                print(f"   🧟 POSICIONES FANTASMA detectadas: {ghost_order_ids}")
                
                # Eliminar posiciones fantasma del risk manager
                for ghost_id in ghost_order_ids:
                    if ghost_id in self.active_positions:
                        del self.active_positions[ghost_id]
                        print(f"      🗑️ Eliminada posición fantasma {ghost_id}")
                
                print(f"   ✅ Limpieza completada: {len(ghost_order_ids)} posición(es) fantasma eliminada(s)")
            else:
                print(f"   ✅ No se detectaron posiciones fantasma")
                
            # ✅ NUEVO: Verificar si hay posiciones en portfolio que no están en risk manager
            missing_in_risk = portfolio_order_ids - risk_manager_order_ids
            
            if missing_in_risk:
                print(f"   ⚠️ POSICIONES FALTANTES en Risk Manager: {missing_in_risk}")
                print(f"      💡 Estas posiciones están en Portfolio pero no en Risk Manager")
                print(f"      🔧 Se requiere sincronización manual o reinicio del bot")
                
            # Verificar sincronización después de la limpieza
            final_portfolio_count = len([pos for pos in self.portfolio_manager.position_registry.values() if pos.symbol == symbol])
            final_risk_count = len([pos for pos in self.active_positions.values() if pos.symbol == symbol])
            
            print(f"   📊 Sincronización final: Portfolio={final_portfolio_count}, Risk={final_risk_count}")
            
            if final_portfolio_count == final_risk_count:
                print(f"   ✅ SINCRONIZACIÓN RESTAURADA para {symbol}")
            else:
                print(f"   ⚠️ ADVERTENCIA: Sincronización incompleta para {symbol}")
                
        except Exception as e:
            print(f"❌ Error en limpieza de desincronización para {symbol}: {e}")

    async def diagnose_position_synchronization(self):
        """
        🔍 DIAGNÓSTICO COMPLETO DE SINCRONIZACIÓN DE POSICIONES
        ---
        Identifica y reporta problemas de sincronización entre contadores
        """
        try:
            print("🔍 DIAGNÓSTICO DE SINCRONIZACIÓN DE POSICIONES")
            print("=" * 60)
            
            # Contadores principales
            risk_manager_count = len(self.active_positions)
            portfolio_count = len(self.portfolio_manager.position_registry) if self.portfolio_manager else 0
            
            print(f"📊 CONTADORES PRINCIPALES:")
            print(f"   🛡️ Risk Manager (self.active_positions): {risk_manager_count}")
            print(f"   💼 Portfolio Manager (position_registry): {portfolio_count}")
            
            if risk_manager_count != portfolio_count:
                print(f"   ❌ DESINCRONIZACIÓN DETECTADA: Diferencia de {abs(risk_manager_count - portfolio_count)} posiciones")
            else:
                print(f"   ✅ SINCRONIZACIÓN CORRECTA")
            
            # Análisis por símbolo
            print(f"\n📊 ANÁLISIS POR SÍMBOLO:")
            
            if self.portfolio_manager:
                # Agrupar posiciones del portfolio por símbolo
                portfolio_by_symbol = {}
                for pos in self.portfolio_manager.position_registry.values():
                    if pos.symbol not in portfolio_by_symbol:
                        portfolio_by_symbol[pos.symbol] = []
                    portfolio_by_symbol[pos.symbol].append(pos)
                
                # Agrupar posiciones del risk manager por símbolo
                risk_by_symbol = {}
                for pos in self.active_positions.values():
                    if pos.symbol not in risk_by_symbol:
                        risk_by_symbol[pos.symbol] = []
                    risk_by_symbol[pos.symbol].append(pos)
                
                # Comparar por símbolo
                all_symbols = set(portfolio_by_symbol.keys()) | set(risk_by_symbol.keys())
                
                for symbol in sorted(all_symbols):
                    portfolio_pos_count = len(portfolio_by_symbol.get(symbol, []))
                    risk_pos_count = len(risk_by_symbol.get(symbol, []))
                    
                    status = "✅" if portfolio_pos_count == risk_pos_count else "❌"
                    print(f"   {status} {symbol}: Portfolio={portfolio_pos_count}, Risk={risk_pos_count}")
                    
                    if portfolio_pos_count != risk_pos_count:
                        print(f"      🔧 Desincronización: {abs(portfolio_pos_count - risk_pos_count)} posición(es) de diferencia")
                        
                        # Mostrar IDs específicos
                        portfolio_ids = [pos.order_id for pos in portfolio_by_symbol.get(symbol, []) if pos.order_id]
                        risk_ids = [pos.order_id for pos in risk_by_symbol.get(symbol, []) if pos.order_id]
                        
                        print(f"      📋 Portfolio IDs: {portfolio_ids}")
                        print(f"      📋 Risk Manager IDs: {risk_ids}")
            
            # Resumen y recomendaciones
            print(f"\n📋 RESUMEN:")
            if risk_manager_count == portfolio_count:
                print(f"   ✅ Sistema sincronizado correctamente")
            else:
                print(f"   ❌ Sistema desincronizado - Se requiere limpieza")
                print(f"   💡 Recomendación: Ejecutar limpieza automática o reiniciar el bot")
            
            print("=" * 60)
            
        except Exception as e:
            print(f"❌ Error en diagnóstico de sincronización: {e}")

    async def _auto_cleanup_if_desynchronized(self):
        """
        🔧 LIMPIEZA AUTOMÁTICA DE DESINCRONIZACIÓN
        ---
        Se ejecuta automáticamente cuando se detecta desincronización
        """
        try:
            if not self.portfolio_manager:
                return
                
            risk_manager_count = len(self.active_positions)
            portfolio_count = len(self.portfolio_manager.position_registry)
            
            # Solo ejecutar si hay desincronización
            if risk_manager_count == portfolio_count:
                return
                
            print(f"🔧 LIMPIEZA AUTOMÁTICA ACTIVADA por desincronización detectada")
            print(f"   📊 Antes: Risk={risk_manager_count}, Portfolio={portfolio_count}")
            
            # Identificar símbolos con desincronización
            portfolio_by_symbol = {}
            risk_by_symbol = {}
            
            for pos in self.portfolio_manager.position_registry.values():
                if pos.symbol not in portfolio_by_symbol:
                    portfolio_by_symbol[pos.symbol] = []
                portfolio_by_symbol[pos.symbol].append(pos)
            
            for pos in self.active_positions.values():
                if pos.symbol not in risk_by_symbol:
                    risk_by_symbol[pos.symbol] = []
                risk_by_symbol[pos.symbol].append(pos)
            
            # Limpiar cada símbolo desincronizado
            cleaned_symbols = []
            for symbol in set(portfolio_by_symbol.keys()) | set(risk_by_symbol.keys()):
                portfolio_pos_count = len(portfolio_by_symbol.get(symbol, []))
                risk_pos_count = len(risk_by_symbol.get(symbol, []))
                
                if portfolio_pos_count != risk_pos_count:
                    await self._cleanup_desynchronized_positions(symbol)
                    cleaned_symbols.append(symbol)
            
            # Verificar resultado final
            final_risk_count = len(self.active_positions)
            final_portfolio_count = len(self.portfolio_manager.position_registry)
            
            print(f"   📊 Después: Risk={final_risk_count}, Portfolio={final_portfolio_count}")
            
            if final_risk_count == final_portfolio_count:
                print(f"   ✅ LIMPIEZA AUTOMÁTICA COMPLETADA - Sincronización restaurada")
            else:
                print(f"   ⚠️ LIMPIEZA INCOMPLETA - Se requiere intervención manual")
                
        except Exception as e:
            print(f"❌ Error en limpieza automática: {e}")

    async def force_synchronization(self):
        """
        🔧 SINCRONIZACIÓN FORZADA DE POSICIONES
        ---
        Método manual para forzar la sincronización entre contadores
        """
        try:
            print("🔧 SINCRONIZACIÓN FORZADA INICIADA")
            print("=" * 50)
            
            if not self.portfolio_manager:
                print("❌ Portfolio Manager no disponible")
                return False
                
            # Obtener estado actual
            risk_manager_count = len(self.active_positions)
            portfolio_count = len(self.portfolio_manager.position_registry)
            
            print(f"📊 ESTADO ACTUAL:")
            print(f"   🛡️ Risk Manager: {risk_manager_count} posiciones")
            print(f"   💼 Portfolio Manager: {portfolio_count} posiciones")
            
            if risk_manager_count == portfolio_count:
                print("✅ Sistema ya está sincronizado")
                return True
            
            # Ejecutar limpieza automática
            await self._auto_cleanup_if_desynchronized()
            
            # Verificar resultado final
            final_risk_count = len(self.active_positions)
            final_portfolio_count = len(self.portfolio_manager.position_registry)
            
            print(f"\n📊 RESULTADO FINAL:")
            print(f"   🛡️ Risk Manager: {final_risk_count} posiciones")
            print(f"   💼 Portfolio Manager: {final_portfolio_count} posiciones")
            
            if final_risk_count == final_portfolio_count:
                print("✅ SINCRONIZACIÓN FORZADA COMPLETADA")
                return True
            else:
                print("❌ SINCRONIZACIÓN INCOMPLETA - Se requiere reinicio")
                return False
                
        except Exception as e:
            print(f"❌ Error en sincronización forzada: {e}")
            return False

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
