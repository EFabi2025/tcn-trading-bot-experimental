#!/usr/bin/env python3
"""
INTEGRATED TRADING SUITE - Suite completa de trading
Combina monitoreo continuo, backtesting, trading automatizado y Discord
"""

import asyncio
import os
from datetime import datetime
from typing import Dict, List
from continuous_monitor_discord import ContinuousMarketMonitor, DiscordNotifier
from backtesting_system import BacktestingManager
from automated_trading_system import AutomatedTradingBot
from final_real_binance_predictor import OptimizedTCNPredictor
import warnings
warnings.filterwarnings('ignore')

class TradingSuite:
    """Suite completa de trading con todos los sistemas"""
    
    def __init__(self):
        self.discord = DiscordNotifier()
        self.monitor = None
        self.backtester = None
        self.trading_bot = None
        self.predictor = OptimizedTCNPredictor()
    
    async def start_interactive_suite(self):
        """Inicio interactivo de la suite completa"""
        
        print("🚀 INTEGRATED TRADING SUITE")
        print("="*60)
        print("🎯 Suite completa de trading automatizado")
        print("📊 Incluye: Monitoreo, Backtesting, Trading y Discord")
        print("="*60)
        print()
        
        # Configurar Discord
        webhook_url = self._setup_discord()
        
        # Menú principal
        while True:
            print("\\n📋 MENÚ PRINCIPAL")
            print("1. 🔍 Monitoreo Continuo con Discord")
            print("2. 📈 Backtesting Histórico")  
            print("3. 🤖 Trading Automatizado")
            print("4. 🎯 Demo Completo (Todos los sistemas)")
            print("5. 📊 Estado de los Modelos")
            print("6. ⚙️  Configuración")
            print("0. ❌ Salir")
            
            try:
                choice = input("\\nSelecciona una opción: ").strip()
                
                if choice == "1":
                    await self.start_monitoring_system()
                elif choice == "2":
                    await self.start_backtesting_system()
                elif choice == "3":
                    await self.start_trading_system()
                elif choice == "4":
                    await self.start_complete_demo()
                elif choice == "5":
                    await self.show_model_status()
                elif choice == "6":
                    self.show_configuration()
                elif choice == "0":
                    print("\\n👋 Saliendo de la suite...")
                    break
                else:
                    print("❌ Opción inválida")
                    
            except KeyboardInterrupt:
                print("\\n\\n⏹️  Operación cancelada")
                break
    
    async def start_monitoring_system(self):
        """Sistema de monitoreo continuo"""
        
        print("\\n🔍 SISTEMA DE MONITOREO CONTINUO")
        print("-" * 40)
        
        webhook_url = os.getenv('DISCORD_WEBHOOK_URL')
        self.monitor = ContinuousMarketMonitor(webhook_url)
        
        print("Duración del monitoreo:")
        print("1. 30 minutos (demo)")
        print("2. 1 hora")
        print("3. 4 horas")
        print("4. 8 horas")
        print("5. 24 horas")
        
        try:
            duration_choice = input("Selecciona duración: ").strip()
            durations = {"1": 0.5, "2": 1, "3": 4, "4": 8, "5": 24}
            duration = durations.get(duration_choice, 1)
            
            print(f"\\n🚀 Iniciando monitoreo por {duration} hora(s)...")
            await self.monitor.start_monitoring(duration)
            
        except KeyboardInterrupt:
            if self.monitor:
                await self.monitor.stop_monitoring()
    
    async def start_backtesting_system(self):
        """Sistema de backtesting"""
        
        print("\\n📈 SISTEMA DE BACKTESTING HISTÓRICO")
        print("-" * 40)
        
        self.backtester = BacktestingManager()
        
        print("Período de backtesting:")
        print("1. 3 días (demo rápido)")
        print("2. 7 días") 
        print("3. 15 días")
        print("4. 30 días")
        print("5. 60 días")
        
        try:
            period_choice = input("Selecciona período: ").strip()
            periods = {"1": 3, "2": 7, "3": 15, "4": 30, "5": 60}
            days = periods.get(period_choice, 7)
            
            print(f"\\n📊 Iniciando backtesting de {days} días...")
            
            # Enviar notificación Discord
            await self.discord.send_alert(
                "📈 Backtesting Iniciado",
                f"Análisis histórico de {days} días iniciado\\n"
                f"Pares: BTCUSDT, ETHUSDT, BNBUSDT\\n"
                f"Validando rendimiento del modelo TCN",
                0x3498db
            )
            
            await self.backtester.run_comprehensive_backtest(days)
            
        except KeyboardInterrupt:
            print("\\n⏹️  Backtesting cancelado")
    
    async def start_trading_system(self):
        """Sistema de trading automatizado"""
        
        print("\\n🤖 SISTEMA DE TRADING AUTOMATIZADO")
        print("-" * 40)
        print("⚠️  IMPORTANTE: Solo en modo simulación por seguridad")
        
        webhook_url = os.getenv('DISCORD_WEBHOOK_URL')
        self.trading_bot = AutomatedTradingBot(testnet=True, discord_webhook=webhook_url)
        
        print("\\nDuración del trading:")
        print("1. 30 minutos (demo)")
        print("2. 1 hora")
        print("3. 4 horas")
        print("4. 8 horas")
        
        try:
            duration_choice = input("Selecciona duración: ").strip()
            durations = {"1": 0.5, "2": 1, "3": 4, "4": 8}
            duration = durations.get(duration_choice, 1)
            
            print(f"\\n🚀 Iniciando trading simulado por {duration} hora(s)...")
            await self.trading_bot.start_automated_trading(duration, dry_run=True)
            
        except KeyboardInterrupt:
            if self.trading_bot:
                await self.trading_bot.stop_trading()
    
    async def start_complete_demo(self):
        """Demo completo de todos los sistemas"""
        
        print("\\n🎯 DEMO COMPLETO - TODOS LOS SISTEMAS")
        print("="*50)
        print("Este demo ejecutará secuencialmente:")
        print("1. 📊 Estado de modelos")
        print("2. 📈 Backtesting (3 días)")
        print("3. 🔍 Monitoreo (30 min)")
        print("4. 🤖 Trading simulado (30 min)")
        print()
        
        confirm = input("¿Continuar con el demo completo? (s/N): ").strip().lower()
        if confirm != 's':
            return
        
        # Notificación de inicio
        await self.discord.send_alert(
            "🎯 Demo Completo Iniciado",
            "Ejecutando suite completa de trading\\n"
            "• Verificación de modelos\\n"
            "• Backtesting histórico\\n"
            "• Monitoreo en tiempo real\\n"
            "• Trading automatizado\\n"
            "Duración estimada: 1.5 horas",
            0xff9900,
            urgent=True
        )
        
        try:
            # 1. Estado de modelos
            print("\\n" + "="*50)
            print("📊 FASE 1: VERIFICACIÓN DE MODELOS")
            print("="*50)
            await self.show_model_status()
            
            # 2. Backtesting
            print("\\n" + "="*50)
            print("📈 FASE 2: BACKTESTING HISTÓRICO")
            print("="*50)
            self.backtester = BacktestingManager()
            await self.backtester.run_comprehensive_backtest(3)  # 3 días
            
            # 3. Monitoreo
            print("\\n" + "="*50)
            print("🔍 FASE 3: MONITOREO EN TIEMPO REAL")
            print("="*50)
            webhook_url = os.getenv('DISCORD_WEBHOOK_URL')
            self.monitor = ContinuousMarketMonitor(webhook_url)
            
            # Ejecutar monitoreo y trading en paralelo
            monitor_task = asyncio.create_task(
                self.monitor.start_monitoring(0.5)  # 30 minutos
            )
            
            # 4. Trading automatizado
            print("\\n" + "="*50)
            print("🤖 FASE 4: TRADING AUTOMATIZADO")
            print("="*50)
            self.trading_bot = AutomatedTradingBot(testnet=True, discord_webhook=webhook_url)
            
            trading_task = asyncio.create_task(
                self.trading_bot.start_automated_trading(0.5, dry_run=True)  # 30 minutos
            )
            
            # Esperar que terminen ambos
            await asyncio.gather(monitor_task, trading_task)
            
            # Reporte final
            await self.discord.send_alert(
                "✅ Demo Completo Finalizado",
                "Suite de trading completada exitosamente\\n"
                "• Modelos verificados ✅\\n"
                "• Backtesting completado ✅\\n"
                "• Monitoreo ejecutado ✅\\n"
                "• Trading simulado ✅\\n"
                "Todos los sistemas operativos",
                0x00ff00
            )
            
            print("\\n" + "="*50)
            print("✅ DEMO COMPLETO FINALIZADO")
            print("Todos los sistemas han sido probados exitosamente")
            print("="*50)
            
        except KeyboardInterrupt:
            print("\\n⏹️  Demo cancelado por usuario")
            
            # Limpiar tareas
            if self.monitor:
                await self.monitor.stop_monitoring()
            if self.trading_bot:
                await self.trading_bot.stop_trading()
    
    async def show_model_status(self):
        """Mostrar estado de los modelos"""
        
        print("\\n📊 ESTADO DE LOS MODELOS TCN")
        print("-" * 40)
        
        try:
            # Verificar carga de modelos
            pairs = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
            model_status = {}
            
            for pair in pairs:
                try:
                    if pair in self.predictor.models:
                        model = self.predictor.models[pair]
                        input_shape = model.input_shape
                        model_status[pair] = {
                            'loaded': True,
                            'input_shape': input_shape,
                            'params': model.count_params()
                        }
                        print(f"  ✅ {pair}: Cargado - Shape {input_shape} - {model.count_params():,} params")
                    else:
                        model_status[pair] = {'loaded': False}
                        print(f"  ❌ {pair}: No cargado")
                except Exception as e:
                    model_status[pair] = {'loaded': False, 'error': str(e)}
                    print(f"  ❌ {pair}: Error - {e}")
            
            # Estadísticas generales
            loaded_models = sum(1 for status in model_status.values() if status.get('loaded', False))
            print(f"\\n📈 Resumen:")
            print(f"  Modelos cargados: {loaded_models}/{len(pairs)}")
            print(f"  Estado general: {'✅ OPERATIVO' if loaded_models == len(pairs) else '⚠️ PARCIAL'}")
            
            # Notificación Discord
            await self.discord.send_alert(
                "📊 Estado de Modelos",
                f"Verificación de modelos TCN\\n"
                f"Modelos cargados: {loaded_models}/{len(pairs)}\\n"
                f"Estado: {'✅ OPERATIVO' if loaded_models == len(pairs) else '⚠️ PARCIAL'}\\n"
                f"Pares disponibles: {', '.join(pairs)}",
                0x00ff00 if loaded_models == len(pairs) else 0xffff00
            )
            
        except Exception as e:
            print(f"❌ Error verificando modelos: {e}")
    
    def show_configuration(self):
        """Mostrar configuración actual"""
        
        print("\\n⚙️  CONFIGURACIÓN ACTUAL")
        print("-" * 40)
        
        # Variables de entorno importantes
        env_vars = {
            'DISCORD_WEBHOOK_URL': 'Discord Webhook',
            'BINANCE_API_KEY': 'Binance API Key',
            'BINANCE_API_SECRET': 'Binance API Secret'
        }
        
        print("Variables de entorno:")
        for var, description in env_vars.items():
            value = os.getenv(var)
            status = "✅ Configurado" if value else "❌ No configurado"
            if var == 'DISCORD_WEBHOOK_URL' and value:
                # Mostrar solo parte del webhook por seguridad
                display_value = f"{value[:30]}..." if len(value) > 30 else value
                print(f"  {description}: {status} ({display_value})")
            elif value and 'API' in var:
                print(f"  {description}: {status} (***)")
            else:
                print(f"  {description}: {status}")
        
        print("\\nArquivos principales:")
        files = [
            'final_real_binance_predictor.py',
            'continuous_monitor_discord.py', 
            'backtesting_system.py',
            'automated_trading_system.py'
        ]
        
        for file in files:
            exists = os.path.exists(file)
            print(f"  {file}: {'✅' if exists else '❌'}")
        
        print("\\nConfiguración recomendada:")
        print("1. Configurar Discord Webhook para notificaciones")
        print("2. Para trading real: configurar APIs de Binance")
        print("3. Usar modo testnet para pruebas")
        print("4. Mantener dry_run=True hasta validar completamente")
    
    def _setup_discord(self) -> str:
        """Configurar Discord webhook"""
        
        webhook_url = os.getenv('DISCORD_WEBHOOK_URL')
        
        if webhook_url:
            print(f"✅ Discord configurado")
            return webhook_url
        
        print("⚠️  Discord no configurado")
        print("Para habilitar notificaciones Discord:")
        print("1. Crea un webhook en tu servidor Discord")
        print("2. Establece: export DISCORD_WEBHOOK_URL='tu_webhook_url'")
        print("3. Reinicia la aplicación")
        
        return None

async def main():
    """Función principal de la suite"""
    
    print("🚀 Inicializando Integrated Trading Suite...")
    
    try:
        suite = TradingSuite()
        await suite.start_interactive_suite()
    except KeyboardInterrupt:
        print("\\n\\n👋 Suite cerrada por usuario")
    except Exception as e:
        print(f"\\n❌ Error en suite: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 