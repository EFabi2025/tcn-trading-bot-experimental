#!/usr/bin/env python3
"""
⚙️ CONFIGURACIÓN CENTRALIZADA DE TRADING
Sistema de Trading Profesional - Eliminación de Valores Hardcodeados
"""

import os
from typing import Dict, Optional
from dataclasses import dataclass
from decimal import Decimal
from dotenv import load_dotenv
from pathlib import Path

# ✅ CORREGIDO: Cargar .env desde el directorio raíz del proyecto
# Obtener la ruta del directorio raíz del proyecto (donde está el archivo .env)
project_root = Path(__file__).parent.parent
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
class RiskParameters:
    """⚠️ Parámetros de riesgo configurables"""
    # Porcentajes de riesgo
    max_position_size_percent: float
    max_total_exposure_percent: float
    max_daily_loss_percent: float
    max_drawdown_percent: float

    # Stop Loss y Take Profit
    stop_loss_percent: float
    take_profit_percent: float
    trailing_stop_percent: float
    min_profit_protection: float  # ✅ NUEVO: Protección mínima de ganancia para trailing stop
    trailing_activation_threshold: float  # ✅ NUEVO: Umbral de activación del trailing stop

    # Límites de posiciones
    max_concurrent_positions: int
    correlation_limit: float

    # Umbrales de confianza
    min_confidence_threshold: float
    signal_reversal_threshold: float

    # Valores monetarios (se obtienen dinámicamente)
    min_position_value_usdt: Optional[float] = None

@dataclass
class TradingConfig:
    """🔧 Configuración completa de trading"""
    risk_params: RiskParameters

    # Configuración de sistema
    heartbeat_interval: int
    position_monitor_interval: int
    metrics_save_interval: int

    # Configuración de notificaciones
    discord_enabled: bool
    discord_webhook_url: Optional[str]

    # Configuración de base de datos
    database_url: str

    # ✅ NUEVO: Configuración de Diversificación de Portafolio
    PORTFOLIO_DIVERSIFICATION = {
        # Límites de concentración por símbolo
        'MAX_SYMBOL_CONCENTRATION_PERCENT': 30.0,  # Máximo 40% del portafolio en un símbolo
        'MAX_POSITIONS_PER_SYMBOL': 3,             # Máximo 3 posiciones por símbolo
        'MIN_SYMBOLS_IN_PORTFOLIO': 2,             # Mínimo 2 símbolos diferentes

        # Diversificación por sectores/categorías
        'SYMBOL_CATEGORIES': {
            'BTCUSDT': 'MAJOR_CRYPTO',
            'ETHUSDT': 'MAJOR_CRYPTO',
            'BNBUSDT': 'EXCHANGE_TOKEN',
            'XRPUSDT': 'ALT_CRYPTO',         # ✅ INTEGRADO desde repositorio externo
            'DOTUSDT': 'ALT_CRYPTO',         # ✅ ACTIVADO: DOTUSDT disponible
            'ADAUSDT': 'ALT_CRYPTO',         # ✅ ACTIVADO: ADAUSDT disponible
            'SOLUSDT': 'ALT_CRYPTO',         # ✅ ACTIVADO: SOLUSDT disponible
            'POLUSDT': 'ALT_CRYPTO',         # ✅ ACTIVADO: POLUSDT disponible (reemplazo de LINKUSDT)
        },
        'MAX_CATEGORY_CONCENTRATION_PERCENT': 90.0,  # Máximo 60% en una categoría

        # Gestión de posiciones existentes
        'RESPECT_EXISTING_POSITIONS': True,         # No liquidar posiciones existentes
        'GRADUAL_REBALANCING': True,               # Rebalanceo gradual con nuevas órdenes
        'DIVERSIFICATION_PRIORITY': 0.3,           # Factor de prioridad para diversificación (0-1)

        # Límites de correlación
        'MAX_CORRELATION_THRESHOLD': 0.8,          # Evitar símbolos muy correlacionados
        'CORRELATION_LOOKBACK_DAYS': 30,           # Días para calcular correlación

        # Configuración de alertas
        'ALERT_ON_HIGH_CONCENTRATION': True,       # Alertar cuando concentración > límite
        'CONCENTRATION_WARNING_THRESHOLD': 35.0,   # Advertir al 35%
    }

    # 🧠 Configuración TensorFlow (Compatible Windows/macOS)
    TENSORFLOW_CONFIG = {
        'use_metal': False,  # Apple Silicon optimization (solo macOS)
        'use_gpu': False,    # GPU NVIDIA (cambiar a True si tienes GPU)
        'memory_growth': True,
        'log_device_placement': False,
        'inter_op_parallelism_threads': 0,  # Auto-detect
        'intra_op_parallelism_threads': 0   # Auto-detect
    }

    # ✅ NUEVO: Sistema de Estabilidad y Cooldown de Señales
    SIGNAL_STABILITY_CONFIG = {
        # Tiempos de cooldown por símbolo (minutos)
        'SIGNAL_COOLDOWN_MINUTES': {
            'ETHUSDT': 15,  # ETH: 15 minutos entre cambios de señal
            'BTCUSDT': 10,  # BTC: 10 minutos
            'BNBUSDT': 12,  # BNB: 12 minutos
            'XRPUSDT': 12,  # XRP: 12 minutos
            'DOTUSDT': 12,  # DOT: 12 minutos
            'ADAUSDT': 12,  # ADA: 12 minutos (nuevo par activado)
            'SOLUSDT': 12,  # SOL: 12 minutos (nuevo par activado)
            'LINKUSDT': 12  # LINK: 12 minutos (nuevo par activado)
        },

        # ✅ NUEVO: Sistema de Penalidad Bearish Gradual
        'BEARISH_PENALTY_CONFIG': {
            # Umbrales BEARISH por intensidad
            'BEARISH_STRONG_BTC_THRESHOLD': 95,
            'BEARISH_STRONG_ALT_THRESHOLD': 98,
            'BEARISH_MODERATE_BTC_THRESHOLD': 85,
            'BEARISH_MODERATE_ALT_THRESHOLD': 90,
            'BEARISH_LEVE_BTC_THRESHOLD': 80,
            'BEARISH_LEVE_ALT_THRESHOLD': 85,

            # Favorecer ventas en BEARISH
            'BEARISH_SELL_THRESHOLD_STRONG': 60,
            'BEARISH_SELL_THRESHOLD_MODERATE': 65,
            'BEARISH_SELL_THRESHOLD_LEVE': 70,

            # Duración para relaxar filtros
            'BEARISH_RELAX_AFTER_HOURS': 48,
            'BEARISH_TIME_RELAXATION_FACTOR': 0.9,

            # Factor de correlación con BTC
            'BTC_CORRELATION_PENALTY': 5,

            # Configuración de intensidad
            'BEARISH_VERY_STRONG_THRESHOLD': 0.9,
            'BEARISH_STRONG_THRESHOLD': 0.8,
            'BEARISH_MODERATE_THRESHOLD': 0.7,
        },

        # Protección específica para ETH
        'ETH_PROTECTION': {
            'min_hold_time_minutes': 20,        # Mínimo 20 min antes de cerrar posición ETH
            'signal_confirmation_required': 2,   # Requiere 2 señales consecutivas para SELL
            'extreme_confidence_threshold': 90.0, # Umbral para bypasses de protección
            'loss_protection_threshold': -4.0,   # Pérdida % que permite cierre inmediato
            'profit_taking_threshold': 5.0       # Ganancia % para cierre con confianza extrema
        },

        # Umbrales de confianza aumentada para cambios de señal
        'MIN_CONFIDENCE_FOR_SIGNAL_CHANGE': {
            'ETHUSDT': 78.0,  # ETH requiere 78% para cambiar señal
            'BTCUSDT': 75.0,  # BTC requiere 75%
            'BNBUSDT': 72.0,  # BNB requiere 72%
            'XRPUSDT': 75.0,  # XRP requiere 75%
            'DOTUSDT': 74.0,  # DOT requiere 74%
            'ADAUSDT': 75.0,  # ADA requiere 75% (nuevo par activado)
            'SOLUSDT': 75.0,  # SOL requiere 75% (nuevo par activado)
            'LINKUSDT': 75.0  # LINK requiere 75% (nuevo par activado)
        },

        # Criterios de cierre de posición por símbolo
        'POSITION_CLOSE_CRITERIA': {
            'ETHUSDT': {
                'extreme_confidence_threshold': 90.0,
                'high_confidence_threshold': 85.0,
                'min_hold_time_minutes': 20,
                'big_loss_threshold': -4.0,
                'medium_loss_threshold': -3.0,
                'big_profit_threshold': 5.0,
                'very_high_profit_threshold': 6.0,
                'reversal_loss_threshold': -2.0
            },
            'DEFAULT': {  # Para BTC, BNB, XRP
                'high_confidence_threshold': 75.0,
                'very_high_confidence_threshold': 85.0,
                'profit_threshold': 2.0,
                'loss_threshold': -1.5,
                'reversal_profit_threshold': 3.0
            }
        }
    }

class ConfigManager:
    """📋 Gestor de configuración centralizada"""

    def __init__(self):
        self._config: Optional[TradingConfig] = None
        self._load_config()

    def _load_config(self):
        """🔄 Cargar configuración desde variables de entorno"""

        # Parámetros de riesgo desde .env
        risk_params = RiskParameters(
            max_position_size_percent=float(os.getenv('MAX_POSITION_SIZE_PERCENT', '15.0')),
            max_total_exposure_percent=float(os.getenv('MAX_TOTAL_EXPOSURE_PERCENT', '40.0')),
            max_daily_loss_percent=float(os.getenv('MAX_DAILY_LOSS_PERCENT', '10.0')),
            max_drawdown_percent=float(os.getenv('MAX_DRAWDOWN_PERCENT', '15.0')),

            stop_loss_percent=float(os.getenv('STOP_LOSS_PERCENT', '3.0')),
            take_profit_percent=float(os.getenv('TAKE_PROFIT_PERCENT', '4.0')),
            trailing_stop_percent=float(os.getenv('TRAILING_STOP_PERCENT', '1.4')),
            min_profit_protection=float(os.getenv('MIN_PROFIT_PROTECTION', '1.5')),  # ✅ NUEVO: Protección mínima de ganancia
            trailing_activation_threshold=float(os.getenv('TRAILING_ACTIVATION_THRESHOLD', '0.5')),  # ✅ NUEVO: Umbral de activación

            max_concurrent_positions=int(os.getenv('MAX_CONCURRENT_POSITIONS', '3')),
            correlation_limit=float(os.getenv('CORRELATION_LIMIT', '0.7')),

            min_confidence_threshold=float(os.getenv('MIN_CONFIDENCE_THRESHOLD', '0.95')),  # ✅ AUMENTADO: 0.70 → 0.95 (95% - NO hará trading)
            signal_reversal_threshold=float(os.getenv('SIGNAL_REVERSAL_THRESHOLD', '0.85')),

            # min_position_value_usdt se obtiene dinámicamente de Binance
            min_position_value_usdt=None
        )

        # Configuración del sistema
        self._config = TradingConfig(
            risk_params=risk_params,

            heartbeat_interval=int(os.getenv('HEARTBEAT_INTERVAL', '30')),
            position_monitor_interval=int(os.getenv('POSITION_MONITOR_INTERVAL', '10')),
            metrics_save_interval=int(os.getenv('METRICS_SAVE_INTERVAL', '300')),

            discord_enabled=os.getenv('DISCORD_ENABLED', 'false').lower() == 'true',
            discord_webhook_url=os.getenv('DISCORD_WEBHOOK_URL'),

            database_url=os.getenv('DATABASE_URL', 'sqlite:///trading.db')
        )

    def get_config(self) -> TradingConfig:
        """📋 Obtener configuración actual"""
        if self._config is None:
            self._load_config()

        if self._config is None:
            raise RuntimeError("Failed to load trading configuration")

        return self._config

    def reload_config(self):
        """🔄 Recargar configuración"""
        print("🔄 Recargando configuración...")
        self._load_config()
        print("✅ Configuración recargada")

    def validate_config(self) -> bool:
        """✅ Validar configuración"""
        config = self.get_config()
        errors = []

        # Validar parámetros de riesgo
        if config.risk_params.stop_loss_percent <= 0:
            errors.append("Stop loss debe ser mayor a 0")

        if config.risk_params.take_profit_percent <= config.risk_params.stop_loss_percent:
            errors.append("Take profit debe ser mayor que stop loss")

        if config.risk_params.max_position_size_percent <= 0 or config.risk_params.max_position_size_percent > 100:
            errors.append("Max position size debe estar entre 0 y 100%")

        if config.risk_params.min_confidence_threshold < 0.5 or config.risk_params.min_confidence_threshold > 1.0:
            errors.append("Min confidence threshold debe estar entre 0.5 y 1.0")

        if config.risk_params.max_concurrent_positions <= 0:
            errors.append("Max concurrent positions debe ser mayor a 0")

        # Mostrar errores si los hay
        if errors:
            print("❌ ERRORES DE CONFIGURACIÓN:")
            for error in errors:
                print(f"   - {error}")
            return False

        print("✅ Configuración válida")
        return True

    def print_config_summary(self):
        """📊 Mostrar resumen de configuración"""
        config = self.get_config()

        print("📋 CONFIGURACIÓN ACTUAL:")
        print("=" * 40)
        print("⚠️ PARÁMETROS DE RIESGO:")
        print(f"   📊 Max posición: {config.risk_params.max_position_size_percent}%")
        print(f"   🚨 Max pérdida diaria: {config.risk_params.max_daily_loss_percent}%")
        print(f"   🛑 Stop Loss: {config.risk_params.stop_loss_percent}%")
        print(f"   🎯 Take Profit: {config.risk_params.take_profit_percent}%")
        print(f"   📈 Trailing Stop: {config.risk_params.trailing_stop_percent}%")
        print(f"   🛡️ Min Profit Protection: {config.risk_params.min_profit_protection}%")
        print(f"   🎯 Trailing Activation: {config.risk_params.trailing_activation_threshold}%")
        print(f"   🔢 Max posiciones: {config.risk_params.max_concurrent_positions}")
        print(f"   🎯 Min confianza: {config.risk_params.min_confidence_threshold:.1%}")
        print(f"   🔄 Umbral reversión: {config.risk_params.signal_reversal_threshold:.1%}")

        print("\n🔧 CONFIGURACIÓN DE SISTEMA:")
        print(f"   💓 Heartbeat: {config.heartbeat_interval}s")
        print(f"   👁️ Monitor posiciones: {config.position_monitor_interval}s")
        print(f"   📊 Guardar métricas: {config.metrics_save_interval}s")
        print(f"   📢 Discord: {'✅' if config.discord_enabled else '❌'}")

# Instancia global del gestor de configuración
config_manager = ConfigManager()

def get_trading_config() -> TradingConfig:
    """🔧 Función helper para obtener configuración"""
    return config_manager.get_config()

def get_risk_params() -> RiskParameters:
    """⚠️ Función helper para obtener parámetros de riesgo"""
    return config_manager.get_config().risk_params

def reload_trading_config():
    """🔄 Función helper para recargar configuración"""
    config_manager.reload_config()

def validate_trading_config() -> bool:
    """✅ Función helper para validar configuración"""
    return config_manager.validate_config()

if __name__ == "__main__":
    # Test de configuración
    print("🧪 TESTING CONFIGURACIÓN...")
    config_manager.print_config_summary()
    print(f"\n✅ Configuración válida: {validate_trading_config()}")
