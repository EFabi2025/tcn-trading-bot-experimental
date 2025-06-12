#!/usr/bin/env python3
"""
CONTINUOUS MONITOR DISCORD - Sistema de monitoreo continuo con alertas Discord
Monitoreo 24/7 del mercado con notificaciones Discord automáticas
"""

import asyncio
import aiohttp
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import tensorflow as tf
from final_real_binance_predictor import OptimizedBinanceData, OptimizedTCNPredictor
import warnings
warnings.filterwarnings('ignore')

class DiscordNotifier:
    """Sistema de notificaciones Discord"""
    
    def __init__(self, webhook_url: str = None):
        self.webhook_url = webhook_url or os.getenv('DISCORD_WEBHOOK_URL')
        if not self.webhook_url:
            print("⚠️  DISCORD_WEBHOOK_URL no configurado. Usando logs únicamente.")
    
    async def send_alert(self, title: str, message: str, color: int = 0x00ff00, 
                        fields: List[Dict] = None, urgent: bool = False):
        """Enviar alerta a Discord"""
        
        if not self.webhook_url:
            print(f"📢 {title}: {message}")
            return
        
        # Crear embed para Discord
        embed = {
            "title": title,
            "description": message,
            "color": color,
            "timestamp": datetime.utcnow().isoformat(),
            "footer": {
                "text": "Binance TCN Trading Bot",
                "icon_url": "https://cdn.discordapp.com/attachments/placeholder.png"
            }
        }
        
        if fields:
            embed["fields"] = fields
        
        payload = {
            "embeds": [embed],
            "username": "TCN Trading Bot"
        }
        
        if urgent:
            payload["content"] = "@everyone 🚨 ALERTA URGENTE 🚨"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.webhook_url, json=payload) as response:
                    if response.status == 204:
                        print(f"✅ Alerta Discord enviada: {title}")
                    else:
                        print(f"❌ Error Discord: {response.status}")
        except Exception as e:
            print(f"❌ Error enviando a Discord: {e}")
    
    async def send_market_update(self, market_data: Dict):
        """Enviar actualización de mercado"""
        
        title = "📊 Actualización de Mercado"
        description = f"Análisis TCN - {datetime.now().strftime('%H:%M:%S')}"
        
        fields = []
        color = 0x3498db  # Azul por defecto
        
        high_confidence_signals = []
        
        for pair, data in market_data.items():
            if data and data.get('prediction'):
                pred = data['prediction']
                signal = pred['signal']
                confidence = pred['confidence']
                price = data.get('price', 0)
                
                # Determinar color según señal
                signal_color = {
                    'BUY': '🟢',
                    'SELL': '🔴', 
                    'HOLD': '🟡'
                }.get(signal, '⚪')
                
                field_value = f"{signal_color} **{signal}**\n"
                field_value += f"💰 ${price:,.4f}\n"
                field_value += f"🎯 {confidence:.1%}\n"
                field_value += f"📊 RSI: {data.get('rsi', 0):.1f}"
                
                fields.append({
                    "name": f"📈 {pair}",
                    "value": field_value,
                    "inline": True
                })
                
                # Detectar señales de alta confianza
                if confidence >= 0.75:
                    high_confidence_signals.append(f"{pair}: {signal} ({confidence:.1%})")
        
        # Cambiar color si hay señales importantes
        if high_confidence_signals:
            color = 0xff9900  # Naranja para alta confianza
            description += f"\n🔥 **{len(high_confidence_signals)} señales de alta confianza**"
        
        await self.send_alert(title, description, color, fields)
    
    async def send_trade_alert(self, pair: str, signal: str, confidence: float, 
                              price: float, analysis: Dict, urgent: bool = True):
        """Enviar alerta específica de trading"""
        
        signal_emoji = {
            'BUY': '🟢',
            'SELL': '🔴',
            'HOLD': '🟡'
        }.get(signal, '⚪')
        
        title = f"{signal_emoji} SEÑAL {signal} - {pair}"
        
        description = f"**Confianza: {confidence:.1%}**\n"
        description += f"💰 Precio: ${price:,.4f}"
        
        fields = [
            {
                "name": "📊 Análisis Técnico",
                "value": f"RSI: {analysis.get('rsi', 0):.1f}\n"
                        f"Precio vs SMA20: {analysis.get('price_vs_sma20', 0):+.2f}%\n"
                        f"Volatilidad: {analysis.get('volatility_ratio', 1):.2f}x\n"
                        f"Momentum: {analysis.get('momentum_10', 0):+.2f}%",
                "inline": True
            },
            {
                "name": "🎯 Distribución",
                "value": f"SELL: {analysis.get('sell_prob', 0):.1%}\n"
                        f"HOLD: {analysis.get('hold_prob', 0):.1%}\n" 
                        f"BUY: {analysis.get('buy_prob', 0):.1%}",
                "inline": True
            }
        ]
        
        # Color según señal
        color = {
            'BUY': 0x00ff00,    # Verde
            'SELL': 0xff0000,   # Rojo
            'HOLD': 0xffff00    # Amarillo
        }.get(signal, 0x808080)
        
        await self.send_alert(title, description, color, fields, urgent and confidence >= 0.75)

class ContinuousMarketMonitor:
    """Monitor continuo del mercado"""
    
    def __init__(self, discord_webhook: str = None):
        self.discord = DiscordNotifier(discord_webhook)
        self.predictor = OptimizedTCNPredictor()
        self.data_provider = None
        
        # Configuración
        self.pairs = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
        self.check_interval = 60  # segundos
        self.alert_cooldown = 300  # 5 minutos entre alertas del mismo par
        
        # Estado
        self.last_alerts = {}  # Cooldown de alertas
        self.last_signals = {}  # Última señal por par
        self.monitoring = False
        
        # Estadísticas
        self.stats = {
            'total_checks': 0,
            'signals_detected': 0,
            'alerts_sent': 0,
            'start_time': None
        }
    
    async def start_monitoring(self, duration_hours: int = 24):
        """Iniciar monitoreo continuo"""
        
        print("🚀 INICIANDO MONITOREO CONTINUO")
        print("="*60)
        print(f"🕐 Duración: {duration_hours} horas")
        print(f"📊 Pares: {', '.join(self.pairs)}")
        print(f"⏱️  Intervalo: {self.check_interval} segundos")
        print("="*60)
        
        self.monitoring = True
        self.stats['start_time'] = datetime.now()
        
        # Alerta de inicio
        await self.discord.send_alert(
            "🚀 Monitor Iniciado",
            f"Monitoreo continuo activo por {duration_hours}h\n"
            f"Pares: {', '.join(self.pairs)}\n"
            f"Intervalo: {self.check_interval}s",
            0x00ff00
        )
        
        end_time = datetime.now() + timedelta(hours=duration_hours)
        cycle = 0
        
        try:
            async with OptimizedBinanceData() as provider:
                self.data_provider = provider
                
                while datetime.now() < end_time and self.monitoring:
                    cycle += 1
                    print(f"\\n🔄 CICLO {cycle} - {datetime.now().strftime('%H:%M:%S')}")
                    
                    # Análisis del mercado
                    market_data = await self.analyze_all_pairs()
                    
                    # Procesar señales
                    await self.process_signals(market_data)
                    
                    # Estadísticas
                    self.stats['total_checks'] += len(self.pairs)
                    
                    # Enviar actualización cada 10 ciclos (10 minutos)
                    if cycle % 10 == 0:
                        await self.discord.send_market_update(market_data)
                    
                    # Esperar siguiente ciclo
                    if datetime.now() < end_time and self.monitoring:
                        await asyncio.sleep(self.check_interval)
                
        except KeyboardInterrupt:
            print("\\n⏹️  Monitoreo detenido por usuario")
        except Exception as e:
            print(f"\\n❌ Error en monitoreo: {e}")
            await self.discord.send_alert(
                "❌ Error de Sistema",
                f"Error en monitoreo: {str(e)}",
                0xff0000,
                urgent=True
            )
        finally:
            await self.stop_monitoring()
    
    async def analyze_all_pairs(self) -> Dict:
        """Analizar todos los pares"""
        
        market_data = {}
        
        for pair in self.pairs:
            try:
                # Obtener datos
                data = await self.data_provider.get_market_data(pair)
                
                if data['klines']:
                    # Predicción
                    prediction = await self.predictor.predict_real_market(pair, data)
                    
                    if prediction:
                        current_candle = data['klines'][-1]
                        ticker_24h = data.get('ticker_24h', {})
                        
                        market_data[pair] = {
                            'prediction': prediction,
                            'price': current_candle['close'],
                            'change_24h': float(ticker_24h.get('priceChangePercent', 0)),
                            'volume_24h': float(ticker_24h.get('volume', 0)),
                            'rsi': prediction['market_analysis'].get('rsi', 50),
                            'timestamp': datetime.now()
                        }
                        
                        print(f"  ✅ {pair}: {prediction['signal']} ({prediction['confidence']:.1%})")
                    else:
                        print(f"  ❌ {pair}: Sin predicción")
                else:
                    print(f"  ⚠️  {pair}: Sin datos")
                    
            except Exception as e:
                print(f"  ❌ {pair}: Error - {e}")
        
        return market_data
    
    async def process_signals(self, market_data: Dict):
        """Procesar señales y enviar alertas"""
        
        for pair, data in market_data.items():
            if not data or 'prediction' not in data:
                continue
            
            prediction = data['prediction']
            signal = prediction['signal']
            confidence = prediction['confidence']
            
            # Verificar si es una señal nueva o cambio significativo
            should_alert = self._should_send_alert(pair, signal, confidence)
            
            if should_alert:
                # Preparar análisis para alerta
                analysis = {
                    'rsi': data.get('rsi', 50),
                    'price_vs_sma20': prediction['market_analysis'].get('price_vs_sma20', 0),
                    'volatility_ratio': prediction['market_analysis'].get('volatility_ratio', 1),
                    'momentum_10': prediction['market_analysis'].get('momentum_10', 0),
                    'sell_prob': prediction['probabilities']['SELL'],
                    'hold_prob': prediction['probabilities']['HOLD'],
                    'buy_prob': prediction['probabilities']['BUY']
                }
                
                # Enviar alerta
                await self.discord.send_trade_alert(
                    pair, signal, confidence, data['price'], analysis,
                    urgent=(confidence >= 0.75)
                )
                
                # Actualizar estado
                self.last_alerts[pair] = datetime.now()
                self.last_signals[pair] = (signal, confidence)
                self.stats['alerts_sent'] += 1
                
                print(f"  🚨 ALERTA: {pair} {signal} ({confidence:.1%})")
            
            self.stats['signals_detected'] += 1
    
    def _should_send_alert(self, pair: str, signal: str, confidence: float) -> bool:
        """Determinar si enviar alerta"""
        
        now = datetime.now()
        
        # Verificar cooldown
        if pair in self.last_alerts:
            time_since_last = (now - self.last_alerts[pair]).total_seconds()
            if time_since_last < self.alert_cooldown:
                return False
        
        # Verificar si es señal nueva o cambio significativo
        if pair in self.last_signals:
            last_signal, last_confidence = self.last_signals[pair]
            
            # Cambio de señal
            if signal != last_signal:
                return True
            
            # Cambio significativo en confianza
            if abs(confidence - last_confidence) >= 0.15:
                return True
        else:
            # Primera señal del par
            return True
        
        # Señal de muy alta confianza (siempre alertar)
        if confidence >= 0.85:
            return True
        
        return False
    
    async def stop_monitoring(self):
        """Detener monitoreo"""
        
        self.monitoring = False
        
        # Estadísticas finales
        runtime = datetime.now() - self.stats['start_time']
        
        await self.discord.send_alert(
            "⏹️ Monitor Detenido",
            f"Monitoreo finalizado\n"
            f"⏱️ Tiempo activo: {runtime}\n"
            f"📊 Checks realizados: {self.stats['total_checks']}\n"
            f"🎯 Señales detectadas: {self.stats['signals_detected']}\n"
            f"🚨 Alertas enviadas: {self.stats['alerts_sent']}",
            0xff9900
        )
        
        print("\\n✅ Monitoreo detenido")
        print(f"📊 Estadísticas: {self.stats}")

class MonitoringManager:
    """Gestor de monitoreo con múltiples opciones"""
    
    def __init__(self):
        self.monitor = None
    
    async def start_interactive_monitoring(self):
        """Inicio interactivo del monitoreo"""
        
        print("🎯 CONTINUOUS MARKET MONITOR")
        print("Sistema de monitoreo continuo con Discord")
        print()
        
        # Configurar webhook de Discord
        webhook_url = self._get_discord_webhook()
        
        self.monitor = ContinuousMarketMonitor(webhook_url)
        
        print("Selecciona duración del monitoreo:")
        print("1. 1 hora (demo)")
        print("2. 4 horas")
        print("3. 8 horas")
        print("4. 24 horas (día completo)")
        print("5. Personalizado")
        
        # Para demo automático, usar 1 hora
        duration = 1
        print(f"\\n🚀 Iniciando monitoreo por {duration} hora(s)...")
        
        try:
            await self.monitor.start_monitoring(duration)
        except KeyboardInterrupt:
            if self.monitor:
                await self.monitor.stop_monitoring()
    
    def _get_discord_webhook(self) -> str:
        """Obtener webhook de Discord"""
        
        # Buscar en variables de entorno
        webhook = os.getenv('DISCORD_WEBHOOK_URL')
        
        if webhook:
            print(f"✅ Discord webhook configurado")
            return webhook
        
        print("⚠️  Discord webhook no configurado")
        print("Para configurar, establece la variable DISCORD_WEBHOOK_URL")
        print("Ejemplo: export DISCORD_WEBHOOK_URL='https://discord.com/api/webhooks/...'")
        print("El sistema funcionará sin notificaciones Discord")
        
        return None

async def main():
    """Función principal"""
    manager = MonitoringManager()
    await manager.start_interactive_monitoring()

if __name__ == "__main__":
    asyncio.run(main()) 