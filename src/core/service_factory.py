"""
🧪 EXPERIMENTAL Service Factory - Trading Bot Research

Este módulo implementa el patrón Factory para crear servicios experimentales:
- Dependency injection para servicios intercambiables
- Configuración centralizada para investigación
- Validaciones de seguridad configurables
- Soporte para modo dry-run Y trading real

⚠️ EXPERIMENTAL: Para investigación en trading algorítmico
"""

from typing import Dict, Any, Optional

import structlog
from ..services.binance_client import ExperimentalBinanceClient
from ..services.ml_predictor import ExperimentalMLPredictor
from ..services.trading_orchestrator import ExperimentalTradingOrchestrator
from ..interfaces.trading_interfaces import ITradingClient, IMLPredictor, ITradingOrchestrator
from ..core.config import TradingBotSettings
from ..core.logging_config import TradingLogger

logger = structlog.get_logger(__name__)


class ExperimentalServiceFactory:
    """
    🧪 Factory experimental para servicios de trading research

    Implementa:
    - SOLID Dependency Injection
    - Configuración flexible para investigación
    - Validaciones de seguridad configurables
    - Modo dry-run Y trading real
    """

    def __init__(self, settings: TradingBotSettings, trading_logger: TradingLogger):
        """
        Inicializa el factory experimental

        Args:
            settings: Configuración del sistema (soporta dry_run y production)
            trading_logger: Logger estructurado para investigación
        """
        self.settings = settings
        self.logger = trading_logger
        self._services: Dict[str, Any] = {}

        # Log configuración experimental
        self.logger.log_system_event(
            "experimental_factory_initialized",
            dry_run=settings.dry_run,
            testnet=settings.binance_testnet,
            environment=settings.environment,
            research_note="Factory configurado para investigación"
        )

    def create_trading_client(self) -> ITradingClient:
        """
        🧪 Crea cliente experimental de Binance

        Configuración según settings:
        - dry_run=True: Solo simulación
        - dry_run=False + testnet=True: Trading real en testnet
        - dry_run=False + testnet=False: Trading real en producción
        """
        if 'trading_client' not in self._services:
            self.logger.log_system_event(
                "creating_experimental_trading_client",
                dry_run=self.settings.dry_run,
                testnet=self.settings.binance_testnet,
                research_note="Creando cliente experimental Binance"
            )

            try:
                # 🧪 EXPERIMENTAL: Cliente configurable
                client = ExperimentalBinanceClient(
                    settings=self.settings,
                    trading_logger=self.logger
                )

                self._services['trading_client'] = client

                self.logger.log_system_event(
                    "experimental_trading_client_created",
                    dry_run=self.settings.dry_run,
                    testnet=self.settings.binance_testnet,
                    research_note="Cliente experimental listo para investigación"
                )

            except Exception as e:
                self.logger.log_error(
                    "experimental_trading_client_creation_failed",
                    error=str(e),
                    research_tip="Verificar configuración de API keys"
                )
                raise

        return self._services['trading_client']

    def create_ml_predictor(self) -> IMLPredictor:
        """
        🧪 Crea predictor ML experimental

        Carga modelo TCN para investigación algorítmica
        """
        if 'ml_predictor' not in self._services:
            self.logger.log_system_event(
                "creating_experimental_ml_predictor",
                research_note="Creando predictor ML para investigación"
            )

            try:
                # 🧪 EXPERIMENTAL: Predictor ML configurable
                predictor = ExperimentalMLPredictor(
                    settings=self.settings,
                    trading_logger=self.logger
                )

                self._services['ml_predictor'] = predictor

                self.logger.log_system_event(
                    "experimental_ml_predictor_created",
                    research_note="Predictor ML experimental listo"
                )

            except Exception as e:
                self.logger.log_error(
                    "experimental_ml_predictor_creation_failed",
                    error=str(e),
                    research_tip="Verificar modelo TCN disponible"
                )
                raise

        return self._services['ml_predictor']

    def create_trading_orchestrator(self) -> ITradingOrchestrator:
        """
        🧪 Crea orquestador experimental de trading

        Configura el motor de trading principal para investigación
        """
        if 'trading_orchestrator' not in self._services:
            self.logger.log_system_event(
                "creating_experimental_trading_orchestrator",
                dry_run=self.settings.dry_run,
                research_note="Creando orquestador experimental"
            )

            try:
                # Inyectar dependencias experimentales
                trading_client = self.create_trading_client()
                ml_predictor = self.create_ml_predictor()

                # 🧪 EXPERIMENTAL: Orquestador configurable
                orchestrator = ExperimentalTradingOrchestrator(
                    trading_client=trading_client,
                    ml_predictor=ml_predictor,
                    settings=self.settings,
                    trading_logger=self.logger
                )

                self._services['trading_orchestrator'] = orchestrator

                self.logger.log_system_event(
                    "experimental_trading_orchestrator_created",
                    dry_run=self.settings.dry_run,
                    research_note="Orquestador experimental configurado"
                )

            except Exception as e:
                self.logger.log_error(
                    "experimental_trading_orchestrator_creation_failed",
                    error=str(e),
                    research_tip="Verificar dependencias del orquestador"
                )
                raise

        return self._services['trading_orchestrator']

    def create_all_services(self) -> Dict[str, Any]:
        """
        🧪 Crea todos los servicios experimentales

        Returns:
            Dict con todos los servicios configurados para investigación
        """
        self.logger.log_system_event(
            "creating_all_experimental_services",
            dry_run=self.settings.dry_run,
            testnet=self.settings.binance_testnet,
            research_note="Inicializando stack completo experimental"
        )

        try:
            # Crear servicios con inyección de dependencias
            services = {
                'trading_client': self.create_trading_client(),
                'ml_predictor': self.create_ml_predictor(),
                'trading_orchestrator': self.create_trading_orchestrator()
            }

            self.logger.log_system_event(
                "all_experimental_services_created",
                services_count=len(services),
                dry_run=self.settings.dry_run,
                research_note="Stack experimental completo listo"
            )

            return services

        except Exception as e:
            self.logger.log_error(
                "experimental_services_creation_failed",
                error=str(e),
                research_tip="Verificar configuración completa del sistema"
            )
            raise

    def validate_experimental_config(self) -> Dict[str, Any]:
        """
        🧪 Valida configuración experimental

        Returns:
            Dict con status de validación para investigación
        """
        validation_results = {
            'config_valid': True,
            'warnings': [],
            'research_notes': []
        }

        # Validar configuración de trading
        if not self.settings.dry_run and not self.settings.binance_testnet:
            validation_results['warnings'].append(
                "🚨 MODO PRODUCCIÓN: Trading real activado"
            )
            validation_results['research_notes'].append(
                "Configurado para trading real en Binance mainnet"
            )
        elif not self.settings.dry_run and self.settings.binance_testnet:
            validation_results['research_notes'].append(
                "Configurado para trading real en Binance testnet"
            )
        else:
            validation_results['research_notes'].append(
                "Configurado para simulación de trading (dry-run)"
            )

        # Validar API keys
        try:
            api_key = self.settings.binance_api_key.get_secret_value()
            secret = self.settings.binance_secret.get_secret_value()

            if len(api_key) < 10 or len(secret) < 10:
                validation_results['config_valid'] = False
                validation_results['warnings'].append(
                    "🚨 API keys parecen inválidas"
                )
        except Exception:
            validation_results['config_valid'] = False
            validation_results['warnings'].append(
                "🚨 Error accediendo a API keys"
            )

        # Log validación
        self.logger.log_system_event(
            "experimental_config_validated",
            **validation_results,
            research_note="Configuración validada para investigación"
        )

        return validation_results

    def get_service_status(self) -> Dict[str, bool]:
        """
        🧪 Obtiene status de servicios experimentales

        Returns:
            Dict con status de cada servicio
        """
        status = {}

        for service_name, service in self._services.items():
            try:
                if hasattr(service, 'is_connected'):
                    status[service_name] = service.is_connected()
                else:
                    status[service_name] = service is not None
            except Exception:
                status[service_name] = False

        self.logger.log_system_event(
            "experimental_services_status_check",
            **status,
            research_note="Status de servicios para monitoreo"
        )

        return status

    async def cleanup_services(self) -> None:
        """
        🧪 Limpia recursos de servicios experimentales
        """
        self.logger.log_system_event(
            "cleaning_experimental_services",
            services_count=len(self._services),
            research_note="Limpiando recursos experimentales"
        )

        # Cerrar servicios con recursos
        for service_name, service in self._services.items():
            try:
                if hasattr(service, 'close'):
                    await service.close()
                    self.logger.log_system_event(
                        f"experimental_{service_name}_closed",
                        research_note=f"Servicio {service_name} cerrado"
                    )
            except Exception as e:
                self.logger.log_error(
                    f"experimental_{service_name}_cleanup_failed",
                    error=str(e),
                    research_note=f"Error cerrando {service_name}"
                )

        # Limpiar referencias
        self._services.clear()

        self.logger.log_system_event(
            "experimental_services_cleaned",
            research_note="Recursos experimentales liberados"
        )
