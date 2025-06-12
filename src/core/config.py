"""
Sistema de configuración centralizado para el trading bot.

Maneja todas las configuraciones del sistema usando Pydantic BaseSettings
con validación estricta y carga desde variables de entorno.
"""
import os
from typing import List, Optional
from decimal import Decimal
from pydantic import BaseSettings, Field, validator, SecretStr
from pathlib import Path


class TradingBotSettings(BaseSettings):
    """
    Configuración principal del trading bot.

    Carga configuraciones desde variables de entorno y archivos .env
    con validación estricta de tipos y rangos.
    """

    # === CONFIGURACIÓN DE BINANCE ===
    binance_api_key: SecretStr = Field(..., description="Binance API Key")
    binance_secret: SecretStr = Field(..., description="Binance Secret Key")
    binance_testnet: bool = Field(default=True, description="Usar Binance testnet")

    # === CONFIGURACIÓN DE TRADING ===
    symbols: List[str] = Field(
        default=["BTCUSDT"],
        description="Lista de símbolos para trading"
    )
    base_asset: str = Field(default="USDT", description="Asset base para trading")

    # Límites de posición
    max_position_percent: float = Field(
        default=0.02,
        ge=0.001,
        le=0.1,
        description="Máximo % del balance por posición (2%)"
    )
    min_order_amount: Decimal = Field(
        default=Decimal("10"),
        gt=0,
        description="Monto mínimo de orden en USDT"
    )

    # === CONFIGURACIÓN DE RIESGO ===
    max_daily_loss_percent: float = Field(
        default=0.05,
        ge=0.01,
        le=0.2,
        description="Máxima pérdida diaria permitida (5%)"
    )
    stop_loss_percent: float = Field(
        default=0.02,
        ge=0.005,
        le=0.1,
        description="Stop loss automático (2%)"
    )
    take_profit_percent: float = Field(
        default=0.04,
        ge=0.01,
        le=0.2,
        description="Take profit automático (4%)"
    )
    max_open_positions: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Máximo número de posiciones abiertas"
    )

    # === CONFIGURACIÓN DEL MODELO ML ===
    model_path: str = Field(
        default="models/tcn_anti_bias_fixed.h5",
        description="Ruta al modelo entrenado"
    )
    scalers_path: str = Field(
        default="models/feature_scalers_fixed.pkl",
        description="Ruta a los scalers del modelo"
    )
    model_confidence_threshold: float = Field(
        default=0.7,
        ge=0.5,
        le=0.95,
        description="Umbral mínimo de confianza del modelo"
    )
    prediction_interval_minutes: int = Field(
        default=5,
        ge=1,
        le=60,
        description="Intervalo de predicción en minutos"
    )
    data_window_size: int = Field(
        default=100,
        ge=50,
        le=500,
        description="Tamaño de ventana de datos para predicción"
    )

    # === CONFIGURACIÓN DE BASE DE DATOS ===
    database_url: str = Field(
        default="sqlite:///database/trading_bot.db",
        description="URL de conexión a la base de datos"
    )
    db_pool_size: int = Field(default=5, ge=1, le=20)
    db_max_overflow: int = Field(default=10, ge=1, le=50)
    db_pool_timeout: int = Field(default=30, ge=5, le=300)
    db_echo: bool = Field(default=False, description="Logging de SQL queries")

    # === CONFIGURACIÓN DE LOGGING ===
    log_level: str = Field(
        default="INFO",
        description="Nivel de logging"
    )
    log_to_file: bool = Field(default=True, description="Guardar logs en archivo")
    log_file_path: str = Field(
        default="logs/trading_bot.log",
        description="Ruta del archivo de log"
    )
    log_rotation: str = Field(
        default="1 day",
        description="Rotación de archivos de log"
    )
    log_retention: int = Field(
        default=30,
        ge=1,
        le=365,
        description="Días de retención de logs"
    )

    # === CONFIGURACIÓN DE NOTIFICACIONES ===
    enable_notifications: bool = Field(
        default=False,
        description="Habilitar notificaciones"
    )
    webhook_url: Optional[str] = Field(
        default=None,
        description="URL del webhook para notificaciones"
    )
    notify_trades: bool = Field(default=True, description="Notificar trades")
    notify_signals: bool = Field(default=True, description="Notificar señales")
    notify_errors: bool = Field(default=True, description="Notificar errores")

    # === CONFIGURACIÓN DE DESARROLLO ===
    environment: str = Field(
        default="development",
        description="Entorno de ejecución"
    )
    debug_mode: bool = Field(default=False, description="Modo debug")
    dry_run: bool = Field(
        default=True,
        description="Modo dry run (no ejecutar trades reales)"
    )

    # === CONFIGURACIÓN DE RATE LIMITING ===
    api_rate_limit: int = Field(
        default=1200,
        ge=100,
        le=2400,
        description="Límite de requests por minuto"
    )
    order_cooldown_seconds: int = Field(
        default=5,
        ge=1,
        le=300,
        description="Tiempo de espera entre órdenes"
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        validate_assignment = True

    @validator('symbols')
    def validate_symbols(cls, v):
        """Valida que todos los símbolos sean válidos."""
        for symbol in v:
            if not symbol.endswith('USDT'):
                raise ValueError(f'Symbol {symbol} must end with USDT')
            if len(symbol) < 6:
                raise ValueError(f'Symbol {symbol} too short')
            if not symbol.isalnum():
                raise ValueError(f'Symbol {symbol} must be alphanumeric')
        return [s.upper() for s in v]

    @validator('log_level')
    def validate_log_level(cls, v):
        """Valida nivel de logging."""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f'Log level must be one of: {valid_levels}')
        return v.upper()

    @validator('environment')
    def validate_environment(cls, v):
        """Valida entorno."""
        valid_envs = ['development', 'staging', 'production']
        if v.lower() not in valid_envs:
            raise ValueError(f'Environment must be one of: {valid_envs}')
        return v.lower()

    @validator('take_profit_percent')
    def validate_take_profit(cls, v, values):
        """Valida que take profit sea mayor que stop loss."""
        if 'stop_loss_percent' in values:
            if v <= values['stop_loss_percent']:
                raise ValueError('Take profit must be greater than stop loss')
        return v

    @validator('webhook_url')
    def validate_webhook_url(cls, v, values):
        """Valida webhook URL cuando notificaciones están habilitadas."""
        if values.get('enable_notifications') and not v:
            raise ValueError('Webhook URL required when notifications enabled')
        if v and not (v.startswith('http://') or v.startswith('https://')):
            raise ValueError('Invalid webhook URL format')
        return v

    @validator('model_path', 'scalers_path')
    def validate_model_paths(cls, v):
        """Valida que los archivos del modelo existan."""
        path = Path(v)
        if not path.exists():
            raise ValueError(f'Model file not found: {v}')
        return str(path.absolute())

    @validator('database_url')
    def validate_database_url(cls, v):
        """Valida URL de base de datos."""
        if not (v.startswith('sqlite://') or v.startswith('postgresql://')):
            raise ValueError('Only SQLite and PostgreSQL supported')

        # Crear directorio de base de datos si es SQLite
        if v.startswith('sqlite://'):
            db_path = v.replace('sqlite:///', '')
            db_dir = Path(db_path).parent
            db_dir.mkdir(parents=True, exist_ok=True)

        return v

    def get_binance_credentials(self) -> tuple[str, str]:
        """
        Obtiene las credenciales de Binance de forma segura.

        Returns:
            Tuple con (api_key, secret_key)
        """
        return (
            self.binance_api_key.get_secret_value(),
            self.binance_secret.get_secret_value()
        )

    def is_production(self) -> bool:
        """Verifica si está en modo producción."""
        return self.environment == 'production'

    def is_testnet(self) -> bool:
        """Verifica si debe usar testnet."""
        return self.binance_testnet or not self.is_production()

    def should_execute_real_trades(self) -> bool:
        """Verifica si debe ejecutar trades reales."""
        return not self.dry_run and self.is_production()


# Instancia global de configuración
settings = TradingBotSettings()


def get_settings() -> TradingBotSettings:
    """
    Factory function para obtener configuración.

    Returns:
        Instancia de configuración validada
    """
    return settings


def reload_settings() -> TradingBotSettings:
    """
    Recarga la configuración desde las variables de entorno.

    Returns:
        Nueva instancia de configuración
    """
    global settings
    settings = TradingBotSettings()
    return settings
