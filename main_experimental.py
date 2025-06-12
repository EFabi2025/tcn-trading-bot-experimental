#!/usr/bin/env python3
"""
🧪 EXPERIMENTAL TCN Trading Bot - Research Tool

Bot de trading experimental para investigación algorítmica que:
- Usa modelo TCN (Temporal Convolutional Network) real
- Puede operar en modo simulación O trading real
- Integra con Binance API (testnet y mainnet)
- Implementa estrategias de ML para análisis de mercado

🚨 EXPERIMENTAL: Para investigación en trading algorítmico
Configure cuidadosamente el modo de operación en .env

Modos disponibles:
1. SIMULACIÓN: dry_run=true - Solo análisis, sin trades reales
2. TESTNET: dry_run=false + testnet=true - Trading real en testnet
3. PRODUCCIÓN: dry_run=false + testnet=false - Trading real en mainnet

⚠️ ADVERTENCIA: Modo producción usa dinero real
"""

import asyncio
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional

import structlog
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.live import Live
from rich.layout import Layout

# Configurar el path para imports
sys.path.append(str(Path(__file__).parent))

from src.core.config import TradingBotSettings
from src.core.logging_config import TradingLogger
from src.core.service_factory import ExperimentalServiceFactory
from src.interfaces.trading_interfaces import ITradingOrchestrator

console = Console()
logger = structlog.get_logger(__name__)


class ExperimentalTradingBot:
    """
    🧪 Bot de Trading Experimental
    
    Sistema completo para investigación en trading algorítmico
    con modelo TCN y configuración flexible de seguridad
    """
    
    def __init__(self):
        self.settings: Optional[TradingBotSettings] = None
        self.trading_logger: Optional[TradingLogger] = None
        self.service_factory: Optional[ExperimentalServiceFactory] = None
        self.orchestrator: Optional[ITradingOrchestrator] = None
        self.is_running = False
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        console.print("\n🛑 Señal de interrupción recibida. Cerrando sistema...", style="bold yellow")
        self.is_running = False
    
    async def initialize(self) -> None:
        """
        🧪 Inicializa el sistema experimental
        """
        try:
            # 1. Cargar configuración
            self.settings = TradingBotSettings()
            
            # 2. Configurar logging
            self.trading_logger = TradingLogger()
            
            # 3. Mostrar configuración
            self._display_configuration()
            
            # 4. Validar y confirmar configuración
            if not await self._confirm_configuration():
                console.print("❌ Configuración cancelada por el usuario", style="bold red")
                return
            
            # 5. Crear servicios
            self.service_factory = ExperimentalServiceFactory(
                settings=self.settings,
                trading_logger=self.trading_logger
            )
            
            # 6. Validar configuración experimental
            validation = self.service_factory.validate_experimental_config()
            self._display_validation_results(validation)
            
            if not validation['config_valid']:
                console.print("❌ Configuración inválida. Revise los errores.", style="bold red")
                return
            
            # 7. Crear orquestador
            self.orchestrator = self.service_factory.create_trading_orchestrator()
            
            console.print("✅ Sistema experimental inicializado correctamente", style="bold green")
            
        except Exception as e:
            console.print(f"❌ Error inicializando sistema: {e}", style="bold red")
            logger.error("initialization_failed", error=str(e))
            raise
    
    def _display_configuration(self) -> None:
        """Muestra la configuración actual"""
        # Determinar el modo de operación
        if self.settings.dry_run:
            mode = "🧪 SIMULACIÓN"
            mode_style = "green"
            description = "Solo análisis - No ejecuta trades reales"
        elif self.settings.binance_testnet:
            mode = "🔬 TESTNET"
            mode_style = "yellow"
            description = "Trading real en testnet de Binance"
        else:
            mode = "🚨 PRODUCCIÓN"
            mode_style = "red"
            description = "Trading REAL con dinero REAL en Binance"
        
        # Crear tabla de configuración
        table = Table(title="🧪 Configuración Experimental del Bot")
        table.add_column("Parámetro", style="cyan", no_wrap=True)
        table.add_column("Valor", style="magenta")
        table.add_column("Descripción", style="dim")
        
        table.add_row("Modo", f"[{mode_style}]{mode}[/{mode_style}]", description)
        table.add_row("Entorno", self.settings.environment, "Entorno de ejecución")
        table.add_row("Símbolo", ", ".join(self.settings.trading_symbols), "Pares de trading")
        table.add_row("Intervalo", f"{self.settings.trading_interval_seconds}s", "Frecuencia de análisis")
        table.add_row("Max Posición", f"{self.settings.max_position_percent}%", "% máximo del portafolio")
        table.add_row("Max Pérdida Diaria", f"{self.settings.max_daily_loss_percent}%", "% máximo de pérdida diaria")
        
        console.print(table)
    
    async def _confirm_configuration(self) -> bool:
        """Confirma la configuración con el usuario"""
        if self.settings.dry_run:
            # En modo simulación, confirmar automáticamente
            console.print("✅ Modo simulación - Continúo automáticamente", style="bold green")
            return True
        
        # En modos con trading real, requerir confirmación explícita
        console.print("\n⚠️  CONFIRMACIÓN REQUERIDA", style="bold yellow")
        
        if self.settings.binance_testnet:
            message = "¿Confirma ejecutar trading REAL en TESTNET de Binance?"
        else:
            message = "🚨 ¿Confirma ejecutar trading REAL en PRODUCCIÓN con DINERO REAL?"
            console.print("💰 ADVERTENCIA: Esto usará dinero real de su cuenta de Binance", style="bold red")
        
        while True:
            response = input(f"\n{message} (sí/no): ").lower().strip()
            if response in ['sí', 'si', 'yes', 'y', 's']:
                return True
            elif response in ['no', 'n']:
                return False
            else:
                console.print("Por favor responda 'sí' o 'no'", style="yellow")
    
    def _display_validation_results(self, validation: Dict[str, Any]) -> None:
        """Muestra resultados de validación"""
        if validation['config_valid']:
            console.print("✅ Configuración válida", style="bold green")
        else:
            console.print("❌ Configuración inválida", style="bold red")
        
        if validation['warnings']:
            console.print("\n⚠️  Advertencias:", style="bold yellow")
            for warning in validation['warnings']:
                console.print(f"  • {warning}", style="yellow")
        
        if validation['research_notes']:
            console.print("\n📋 Notas de configuración:", style="bold blue")
            for note in validation['research_notes']:
                console.print(f"  • {note}", style="blue")
    
    async def run_continuous_trading(self) -> None:
        """
        🔄 Ejecuta trading continuo experimental
        """
        if not self.orchestrator:
            raise RuntimeError("Sistema no inicializado")
        
        console.print("🚀 Iniciando trading continuo experimental...", style="bold green")
        self.is_running = True
        
        try:
            # Layout para display en vivo
            layout = Layout()
            layout.split_column(
                Layout(Panel(Text("📊 TCN Trading Bot Experimental", justify="center", style="bold blue"))),
                Layout(name="main"),
                Layout(Panel(Text("Presione Ctrl+C para detener", justify="center", style="dim")), size=3)
            )
            
            with Live(layout, refresh_per_second=1) as live:
                cycle_count = 0
                
                while self.is_running:
                    cycle_count += 1
                    
                    try:
                        # Actualizar display
                        self._update_live_display(layout, cycle_count)
                        
                        # Ejecutar ciclo de trading
                        results = await self.orchestrator.execute_trading_cycle()
                        
                        # Log resultados
                        self.trading_logger.log_trading_cycle_completed(
                            results.dict(),
                            research_note=f"Ciclo {cycle_count} completado"
                        )
                        
                        # Esperar hasta el siguiente ciclo
                        await asyncio.sleep(self.settings.trading_interval_seconds)
                        
                    except KeyboardInterrupt:
                        break
                    except Exception as e:
                        logger.error("trading_cycle_error", cycle=cycle_count, error=str(e))
                        console.print(f"❌ Error en ciclo {cycle_count}: {e}", style="red")
                        await asyncio.sleep(5)  # Esperar antes de reintentar
        
        except Exception as e:
            logger.error("continuous_trading_failed", error=str(e))
            console.print(f"❌ Error en trading continuo: {e}", style="bold red")
        
        finally:
            console.print("🛑 Trading continuo detenido", style="bold yellow")
    
    def _update_live_display(self, layout: Layout, cycle_count: int) -> None:
        """Actualiza el display en vivo"""
        # Obtener status de servicios
        status = self.service_factory.get_service_status()
        
        # Crear tabla de status
        status_table = Table(title=f"Status del Sistema - Ciclo {cycle_count}")
        status_table.add_column("Servicio", style="cyan")
        status_table.add_column("Estado", style="green")
        status_table.add_column("Tiempo", style="dim")
        
        for service, is_ok in status.items():
            estado = "✅ OK" if is_ok else "❌ Error"
            status_table.add_row(service, estado, datetime.now(timezone.utc).strftime("%H:%M:%S"))
        
        layout["main"].update(Panel(status_table))
    
    async def run_manual_analysis(self) -> None:
        """
        🔍 Ejecuta análisis manual experimental
        """
        if not self.orchestrator:
            raise RuntimeError("Sistema no inicializado")
        
        console.print("🔍 Ejecutando análisis manual experimental...", style="bold blue")
        
        try:
            for symbol in self.settings.trading_symbols:
                console.print(f"\n📊 Analizando {symbol}...", style="cyan")
                
                # Ejecutar análisis
                results = await self.orchestrator.analyze_symbol(symbol)
                
                # Mostrar resultados
                self._display_analysis_results(symbol, results)
                
                # Log resultados
                self.trading_logger.log_analysis_completed(
                    symbol,
                    results.dict(),
                    research_note="Análisis manual experimental"
                )
        
        except Exception as e:
            logger.error("manual_analysis_failed", error=str(e))
            console.print(f"❌ Error en análisis manual: {e}", style="bold red")
    
    def _display_analysis_results(self, symbol: str, results: Dict[str, Any]) -> None:
        """Muestra resultados de análisis"""
        table = Table(title=f"Análisis de {symbol}")
        table.add_column("Métrica", style="cyan")
        table.add_column("Valor", style="magenta")
        
        # Mostrar datos básicos del análisis
        for key, value in results.items():
            if isinstance(value, (int, float, str)):
                table.add_row(str(key), str(value))
        
        console.print(table)
    
    async def shutdown(self) -> None:
        """
        🛑 Cierra el sistema experimental ordenadamente
        """
        console.print("🛑 Cerrando sistema experimental...", style="bold yellow")
        
        try:
            if self.service_factory:
                await self.service_factory.cleanup_services()
            
            if self.trading_logger:
                self.trading_logger.log_system_event(
                    "experimental_bot_shutdown",
                    research_note="Sistema experimental cerrado correctamente"
                )
            
            console.print("✅ Sistema cerrado correctamente", style="bold green")
        
        except Exception as e:
            logger.error("shutdown_error", error=str(e))
            console.print(f"❌ Error durante cierre: {e}", style="red")


async def main():
    """
    🚀 Función principal experimental
    """
    console.print("🧪 TCN Trading Bot Experimental", style="bold blue")
    console.print("Para investigación en trading algorítmico\n", style="dim")
    
    bot = ExperimentalTradingBot()
    
    try:
        # Inicializar sistema
        await bot.initialize()
        
        if not bot.orchestrator:
            console.print("❌ Sistema no se pudo inicializar", style="bold red")
            return
        
        # Menú principal
        while True:
            console.print("\n🧪 Menú Experimental:")
            console.print("1. 🔄 Trading continuo (automático)")
            console.print("2. 🔍 Análisis manual")
            console.print("3. 📊 Ver estado del sistema")
            console.print("4. 🛑 Salir")
            
            try:
                choice = input("\nSeleccione una opción: ").strip()
                
                if choice == "1":
                    await bot.run_continuous_trading()
                elif choice == "2":
                    await bot.run_manual_analysis()
                elif choice == "3":
                    status = bot.service_factory.get_service_status()
                    console.print(f"\n📊 Estado del sistema: {status}")
                elif choice == "4":
                    break
                else:
                    console.print("❌ Opción inválida", style="red")
            
            except KeyboardInterrupt:
                break
            except Exception as e:
                console.print(f"❌ Error: {e}", style="red")
                logger.error("menu_error", error=str(e))
    
    except Exception as e:
        console.print(f"❌ Error crítico: {e}", style="bold red")
        logger.error("critical_error", error=str(e))
    
    finally:
        await bot.shutdown()


if __name__ == "__main__":
    # Configurar logging
    import logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n👋 Bot experimental detenido por el usuario", style="bold yellow")
    except Exception as e:
        console.print(f"\n💥 Error crítico: {e}", style="bold red")
        sys.exit(1) 