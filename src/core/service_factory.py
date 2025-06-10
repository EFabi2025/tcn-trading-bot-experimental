"""
🧪 EDUCATIONAL Service Factory - Trading Bot Experimental

Este módulo implementa el patrón Factory educacional que:
- Crea e inyecta todas las dependencias SOLID
- Configura servicios en modo educacional
- Demuestra Dependency Injection patterns
- Facilita testing con mocks

⚠️ EXPERIMENTAL: Solo para fines educacionales
"""

from typing import Optional, Dict, Any
import structlog

from ..core.config import TradingBotSettings
from ..core.logging_config import TradingLogger

# Interfaces
from ..interfaces.trading_interfaces import (
    ITradingClient, IMLPredictor, IRiskManager, 
    IMarketDataProvider, INotificationService
)

# Implementaciones educacionales
from ..services.binance_client import EducationalBinanceClient
from ..services.ml_predictor import EducationalMLPredictor
from ..services.risk_manager import EducationalRiskManager
from ..services.trading_orchestrator import EducationalTradingOrchestrator

logger = structlog.get_logger(__name__)


class EducationalServiceFactory:
    """
    🎓 Factory educacional para servicios del trading bot
    
    Características educacionales:
    - Implementa Dependency Injection pattern
    - Crea servicios en modo seguro (dry-run)
    - Facilita testing con interfaces
    - Demuestra arquitectura SOLID limpia
    """
    
    def __init__(self, settings: TradingBotSettings):
        """
        Inicializa la factory educacional
        
        Args:
            settings: Configuración del bot (debe ser modo educacional)
        """
        self.settings = settings
        self.trading_logger = TradingLogger()
        
        # Validación educacional de seguridad
        self._validate_educational_settings()
        
        # Cache de servicios (singleton pattern)
        self._services_cache: Dict[str, Any] = {}
        
        logger.info(
            "🎓 Factory educacional inicializada",
            dry_run=settings.dry_run,
            testnet=settings.binance_testnet,
            educational_note="Factory lista para crear servicios experimentales"
        )
    
    def _validate_educational_settings(self) -> None:
        """Valida que la configuración sea segura para educación"""
        if not self.settings.dry_run:
            raise ValueError(
                "🚨 EDUCATIONAL: Factory requiere dry_run=True para seguridad"
            )
        
        if not self.settings.binance_testnet:
            raise ValueError(
                "🚨 EDUCATIONAL: Factory requiere testnet=True para educación"
            )
        
        if self.settings.environment == "production":
            raise ValueError(
                "🚨 EDUCATIONAL: Factory no puede usar environment=production"
            )
    
    def create_trading_client(self) -> ITradingClient:
        """
        🎓 Crea cliente de trading educacional
        
        Returns:
            Cliente que implementa ITradingClient con Binance testnet
        """
        if "trading_client" not in self._services_cache:
            self._services_cache["trading_client"] = EducationalBinanceClient(
                settings=self.settings,
                trading_logger=self.trading_logger
            )
            
            logger.info(
                "🎓 Cliente de trading educacional creado",
                educational_note="Cliente Binance en modo testnet/dry-run"
            )
        
        return self._services_cache["trading_client"]
    
    def create_market_data_provider(self) -> IMarketDataProvider:
        """
        🎓 Crea proveedor de datos de mercado educacional
        
        Returns:
            Proveedor que implementa IMarketDataProvider
        """
        # En este caso, reutilizamos el mismo cliente de Binance
        # que también implementa IMarketDataProvider
        return self.create_trading_client()
    
    def create_ml_predictor(self) -> IMLPredictor:
        """
        🎓 Crea predictor ML educacional
        
        Returns:
            Predictor que implementa IMLPredictor con modelo TCN
        """
        if "ml_predictor" not in self._services_cache:
            self._services_cache["ml_predictor"] = EducationalMLPredictor(
                settings=self.settings,
                trading_logger=self.trading_logger
            )
            
            logger.info(
                "🎓 Predictor ML educacional creado",
                educational_note="Predictor TCN para experimentación"
            )
        
        return self._services_cache["ml_predictor"]
    
    def create_risk_manager(self) -> IRiskManager:
        """
        🎓 Crea gestor de riesgos educacional
        
        Returns:
            Gestor que implementa IRiskManager con validaciones
        """
        if "risk_manager" not in self._services_cache:
            self._services_cache["risk_manager"] = EducationalRiskManager(
                settings=self.settings,
                trading_logger=self.trading_logger
            )
            
            logger.info(
                "🎓 Gestor de riesgos educacional creado",
                educational_note="Risk manager con validaciones experimentales"
            )
        
        return self._services_cache["risk_manager"]
    
    def create_notification_service(self) -> Optional[INotificationService]:
        """
        🎓 Crea servicio de notificaciones educacional (opcional)
        
        Returns:
            Servicio que implementa INotificationService o None
        """
        # Por ahora, retornar None - se puede implementar después
        # como servicio de logging o email educacional
        logger.info(
            "🎓 Servicio de notificaciones no implementado",
            educational_note="Se puede agregar Discord/Slack/Email educacional"
        )
        return None
    
    def create_trading_orchestrator(self) -> EducationalTradingOrchestrator:
        """
        🎓 Crea orquestador principal educacional
        
        Returns:
            Orquestador completamente configurado con todas las dependencias
        """
        if "orchestrator" not in self._services_cache:
            # Crear todas las dependencias
            trading_client = self.create_trading_client()
            market_data_provider = self.create_market_data_provider()
            ml_predictor = self.create_ml_predictor()
            risk_manager = self.create_risk_manager()
            notification_service = self.create_notification_service()
            
            # Crear orquestador con inyección de dependencias
            self._services_cache["orchestrator"] = EducationalTradingOrchestrator(
                settings=self.settings,
                trading_logger=self.trading_logger,
                trading_client=trading_client,
                market_data_provider=market_data_provider,
                ml_predictor=ml_predictor,
                risk_manager=risk_manager,
                notification_service=notification_service,
                trading_strategy=None  # Se puede agregar después
            )
            
            logger.info(
                "🎓 Orquestador educacional creado",
                educational_note="Orquestador con todas las dependencias inyectadas"
            )
        
        return self._services_cache["orchestrator"]
    
    def create_all_services(self) -> Dict[str, Any]:
        """
        🎓 Crea todos los servicios educacionales
        
        Returns:
            Diccionario con todos los servicios creados
        """
        services = {
            "trading_client": self.create_trading_client(),
            "market_data_provider": self.create_market_data_provider(),
            "ml_predictor": self.create_ml_predictor(),
            "risk_manager": self.create_risk_manager(),
            "notification_service": self.create_notification_service(),
            "orchestrator": self.create_trading_orchestrator(),
            "trading_logger": self.trading_logger,
            "settings": self.settings
        }
        
        logger.info(
            "🎓 Todos los servicios educacionales creados",
            services_count=len([s for s in services.values() if s is not None]),
            educational_note="Sistema completo listo para experimentación"
        )
        
        return services
    
    async def close_all_services(self) -> None:
        """🎓 Cierra todos los servicios creados"""
        try:
            for service_name, service in self._services_cache.items():
                if service and hasattr(service, 'close'):
                    await service.close()
                    logger.info(
                        f"🎓 Servicio {service_name} cerrado",
                        educational_note="Recurso liberado correctamente"
                    )
            
            self._services_cache.clear()
            
        except Exception as e:
            logger.error(
                "🎓 Error cerrando servicios educacionales",
                error=str(e),
                educational_tip="Algunos recursos pueden no haberse liberado"
            )
    
    def get_service_status(self) -> Dict[str, Any]:
        """🎓 Obtiene estado de todos los servicios"""
        status = {}
        
        for service_name, service in self._services_cache.items():
            if service:
                # Intentar obtener status específico del servicio
                if hasattr(service, 'is_connected'):
                    status[service_name] = {
                        "created": True,
                        "connected": service.is_connected()
                    }
                elif hasattr(service, 'is_model_loaded'):
                    status[service_name] = {
                        "created": True,
                        "model_loaded": service.is_model_loaded
                    }
                elif hasattr(service, 'is_running'):
                    status[service_name] = {
                        "created": True,
                        "running": service.is_running
                    }
                else:
                    status[service_name] = {"created": True}
            else:
                status[service_name] = {"created": False}
        
        status["educational_note"] = "Estado de servicios experimentales"
        return status


class EducationalServiceFactoryBuilder:
    """
    🎓 Builder para configurar la factory educacional
    
    Permite configurar la factory paso a paso de manera fluida
    """
    
    def __init__(self):
        self._settings: Optional[TradingBotSettings] = None
        self._custom_config: Dict[str, Any] = {}
    
    def with_settings(self, settings: TradingBotSettings) -> 'EducationalServiceFactoryBuilder':
        """Configura settings base"""
        self._settings = settings
        return self
    
    def with_custom_config(self, **config) -> 'EducationalServiceFactoryBuilder':
        """Agrega configuración personalizada"""
        self._custom_config.update(config)
        return self
    
    def build(self) -> EducationalServiceFactory:
        """Construye la factory educacional"""
        if not self._settings:
            raise ValueError("🎓 Settings requeridos para construir factory")
        
        # Aplicar configuración personalizada si existe
        if self._custom_config:
            # Crear nueva configuración con overrides
            config_dict = self._settings.dict()
            config_dict.update(self._custom_config)
            # Recrear settings con nueva configuración
            self._settings = TradingBotSettings(**config_dict)
        
        return EducationalServiceFactory(self._settings)


# Funciones de conveniencia educacionales

def create_educational_trading_system(settings: TradingBotSettings) -> Dict[str, Any]:
    """
    🎓 Función de conveniencia para crear sistema completo educacional
    
    Args:
        settings: Configuración educacional del bot
        
    Returns:
        Sistema completo de trading educacional
    """
    factory = EducationalServiceFactory(settings)
    return factory.create_all_services()


def create_educational_factory_with_overrides(**overrides) -> EducationalServiceFactory:
    """
    🎓 Crea factory con configuración por defecto y overrides educacionales
    
    Args:
        **overrides: Parámetros a sobrescribir en la configuración
        
    Returns:
        Factory configurada para educación
    """
    # Configuración por defecto educacional
    default_config = {
        "dry_run": True,
        "binance_testnet": True,
        "environment": "development",
        "trading_symbols": ["BTCUSDT"],
        "trading_interval_seconds": 60,
        "max_position_percent": 0.01,
        "max_daily_loss_percent": 0.02
    }
    
    # Aplicar overrides
    default_config.update(overrides)
    
    # Crear settings
    settings = TradingBotSettings(**default_config)
    
    return EducationalServiceFactory(settings) 