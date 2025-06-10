#!/usr/bin/env python3
"""
🧪 EDUCATIONAL Trading Bot - Main Script

Este script demuestra el uso completo del trading bot educacional:
- Carga configuración segura (dry-run + testnet)
- Inicializa todos los servicios SOLID
- Ejecuta ciclo de trading educacional
- Muestra logging estructurado en tiempo real

⚠️ EXPERIMENTAL: Solo para fines educacionales
⚠️ NO ejecuta trades reales - Solo simulación
⚠️ Solo opera en Binance testnet
"""

import asyncio
import signal
import sys
from pathlib import Path
from typing import Optional

import structlog

# Agregar src al path para imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.core.config import TradingBotSettings
from src.core.service_factory import (
    EducationalServiceFactory,
    create_educational_factory_with_overrides
)

# Configurar logging educacional
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


class EducationalTradingBotMain:
    """
    🎓 Aplicación principal del trading bot educacional
    
    Demuestra:
    - Configuración segura educacional
    - Inicialización de servicios SOLID
    - Ejecución de trading algorítmico
    - Manejo graceful de shutdown
    """
    
    def __init__(self):
        self.factory: Optional[EducationalServiceFactory] = None
        self.services: Optional[dict] = None
        self.is_running = False
    
    async def initialize(self):
        """🎓 Inicializa el sistema educacional"""
        try:
            logger.info(
                "🎓 Iniciando Trading Bot Educacional",
                educational_note="Sistema experimental - NO ejecuta trades reales"
            )
            
            # Crear factory con configuración educacional segura
            self.factory = create_educational_factory_with_overrides(
                # Configuración obligatoria de seguridad
                dry_run=True,
                binance_testnet=True,
                environment="development",
                
                # Configuración experimental
                trading_symbols=["BTCUSDT", "ETHUSDT"],
                trading_interval_seconds=30,  # 30 segundos para demo
                max_position_percent=0.001,   # 0.1% experimental
                max_daily_loss_percent=0.005,  # 0.5% experimental
                
                # ML y Risk educacional
                ml_confidence_threshold=0.7,
                circuit_breaker_threshold=0.01,
                
                # Credenciales (deben estar en .env)
                binance_api_key="dummy_key_for_educational_demo",
                binance_secret="dummy_secret_for_educational_demo"
            )
            
            # Crear todos los servicios
            self.services = self.factory.create_all_services()
            
            logger.info(
                "🎓 Sistema educacional inicializado",
                servicios_creados=len([s for s in self.services.values() if s is not None]),
                educational_note="Todos los servicios listos para experimentación"
            )
            
        except Exception as e:
            logger.error(
                "🎓 Error inicializando sistema educacional",
                error=str(e),
                educational_tip="Verificar configuración y dependencias"
            )
            raise
    
    async def run_demo_cycle(self, duration_seconds: int = 300):
        """
        🎓 Ejecuta ciclo de demostración educacional
        
        Args:
            duration_seconds: Duración de la demo en segundos (5 min default)
        """
        if not self.services:
            raise ValueError("🎓 Sistema no inicializado")
        
        orchestrator = self.services["orchestrator"]
        
        try:
            logger.info(
                "🎓 Iniciando demostración de trading educacional",
                duracion_segundos=duration_seconds,
                educational_note="Demo de trading algorítmico experimental"
            )
            
            # Usar context manager para sesión educacional
            async with orchestrator.trading_session():
                self.is_running = True
                
                # Ejecutar por tiempo limitado para demo
                await asyncio.sleep(duration_seconds)
                
            logger.info(
                "🎓 Demostración completada",
                educational_note="Demo educacional finalizada exitosamente"
            )
            
        except Exception as e:
            logger.error(
                "🎓 Error en demostración educacional",
                error=str(e),
                educational_tip="Error durante ejecución del ciclo demo"
            )
            raise
    
    async def run_interactive_mode(self):
        """🎓 Ejecuta modo interactivo educacional"""
        if not self.services:
            raise ValueError("🎓 Sistema no inicializado")
        
        orchestrator = self.services["orchestrator"]
        
        try:
            logger.info(
                "🎓 Iniciando modo interactivo educacional",
                educational_note="Presiona Ctrl+C para detener"
            )
            
            # Configurar handler para Ctrl+C
            def signal_handler(signum, frame):
                logger.info("🎓 Señal de interrupción recibida, deteniendo...")
                self.is_running = False
            
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
            
            # Iniciar trading
            await orchestrator.start_trading()
            self.is_running = True
            
            # Mostrar estadísticas periódicamente
            while self.is_running:
                await asyncio.sleep(60)  # Cada minuto
                
                if self.is_running:
                    stats = orchestrator.get_trading_stats()
                    logger.info(
                        "🎓 Estadísticas educacionales",
                        **stats
                    )
            
            # Detener trading
            await orchestrator.stop_trading()
            
        except KeyboardInterrupt:
            logger.info("🎓 Interrupción manual del usuario")
        except Exception as e:
            logger.error(
                "🎓 Error en modo interactivo",
                error=str(e),
                educational_tip="Error durante ejecución interactiva"
            )
            raise
    
    async def show_system_status(self):
        """🎓 Muestra estado completo del sistema educacional"""
        if not self.factory:
            print("🎓 Sistema no inicializado")
            return
        
        print("\n" + "="*60)
        print("🎓 ESTADO DEL SISTEMA EDUCACIONAL")
        print("="*60)
        
        # Estado de servicios
        status = self.factory.get_service_status()
        for service_name, service_status in status.items():
            if service_name != "educational_note":
                print(f"📋 {service_name}: {service_status}")
        
        # Estadísticas del orquestador
        if self.services and "orchestrator" in self.services:
            stats = self.services["orchestrator"].get_trading_stats()
            print(f"\n📊 ESTADÍSTICAS DE TRADING:")
            for key, value in stats.items():
                if key != "educational_note":
                    print(f"   {key}: {value}")
        
        # Performance del modelo ML
        if self.services and "ml_predictor" in self.services:
            ml_performance = self.services["ml_predictor"].get_model_performance()
            print(f"\n🧠 MODELO ML:")
            for key, value in ml_performance.items():
                if key != "educational_note":
                    print(f"   {key}: {value}")
        
        print("\n" + "="*60)
    
    async def cleanup(self):
        """🎓 Limpia recursos del sistema educacional"""
        try:
            if self.factory:
                await self.factory.close_all_services()
                logger.info(
                    "🎓 Limpieza completada",
                    educational_note="Todos los recursos liberados"
                )
        except Exception as e:
            logger.error(
                "🎓 Error en limpieza",
                error=str(e),
                educational_tip="Algunos recursos pueden no haberse liberado"
            )


async def main():
    """🎓 Función principal educacional"""
    print("🧪 TRADING BOT EDUCACIONAL - EXPERIMENTAL")
    print("⚠️  SOLO PARA FINES EDUCACIONALES")
    print("⚠️  NO EJECUTA TRADES REALES")
    print("⚠️  SOLO TESTNET DE BINANCE")
    print("="*50)
    
    # Crear aplicación educacional
    app = EducationalTradingBotMain()
    
    try:
        # Inicializar sistema
        await app.initialize()
        
        # Mostrar estado inicial
        await app.show_system_status()
        
        # Preguntar modo de ejecución
        print("\n🎓 Modos de ejecución disponibles:")
        print("1. Demo de 5 minutos (automático)")
        print("2. Modo interactivo (manual)")
        print("3. Solo mostrar estado y salir")
        
        try:
            choice = input("\nSelecciona modo (1-3): ").strip()
        except (EOFError, KeyboardInterrupt):
            choice = "3"
        
        if choice == "1":
            print("\n🎓 Iniciando demo automática de 5 minutos...")
            await app.run_demo_cycle(300)  # 5 minutos
            
        elif choice == "2":
            print("\n🎓 Iniciando modo interactivo...")
            await app.run_interactive_mode()
            
        else:
            print("\n🎓 Mostrando solo estado del sistema...")
            await app.show_system_status()
        
        print("\n🎓 Ejecución completada exitosamente")
        
    except Exception as e:
        logger.error(
            "🎓 Error en ejecución principal",
            error=str(e),
            educational_tip="Verificar logs para más detalles"
        )
        sys.exit(1)
        
    finally:
        # Limpiar recursos
        await app.cleanup()
        print("🎓 Trading Bot Educacional finalizado")


if __name__ == "__main__":
    # Ejecutar aplicación educacional
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🎓 Ejecución interrumpida por el usuario")
    except Exception as e:
        print(f"\n🎓 Error fatal: {e}")
        sys.exit(1) 