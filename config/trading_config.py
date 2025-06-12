#!/usr/bin/env python3
"""
⚙️ CONFIGURACIÓN CENTRALIZADA DE TRADING
Sistema de Trading Profesional - Eliminación de Valores Hardcodeados
"""

import os
from typing import Dict, Optional
from dataclasses import dataclass
from decimal import Decimal

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
        'MAX_SYMBOL_CONCENTRATION_PERCENT': 40.0,  # Máximo 40% del portafolio en un símbolo
        'MAX_POSITIONS_PER_SYMBOL': 3,             # Máximo 3 posiciones por símbolo
        'MIN_SYMBOLS_IN_PORTFOLIO': 2,             # Mínimo 2 símbolos diferentes

        # Diversificación por sectores/categorías
        'SYMBOL_CATEGORIES': {
            'BTCUSDT': 'MAJOR_CRYPTO',
            'ETHUSDT': 'MAJOR_CRYPTO',
            'BNBUSDT': 'EXCHANGE_TOKEN',
            'ADAUSDT': 'ALT_CRYPTO',
            'DOTUSDT': 'ALT_CRYPTO',
            'SOLUSDT': 'ALT_CRYPTO'
        },
        'MAX_CATEGORY_CONCENTRATION_PERCENT': 60.0,  # Máximo 60% en una categoría

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
            take_profit_percent=float(os.getenv('TAKE_PROFIT_PERCENT', '6.0')),
            trailing_stop_percent=float(os.getenv('TRAILING_STOP_PERCENT', '2.0')),

            max_concurrent_positions=int(os.getenv('MAX_CONCURRENT_POSITIONS', '2')),
            correlation_limit=float(os.getenv('CORRELATION_LIMIT', '0.7')),

            min_confidence_threshold=float(os.getenv('MIN_CONFIDENCE_THRESHOLD', '0.70')),
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
