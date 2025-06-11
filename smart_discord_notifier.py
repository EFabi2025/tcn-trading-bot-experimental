#!/usr/bin/env python3
"""
🤖 SMART DISCORD NOTIFIER
Sistema inteligente de notificaciones Discord con filtros de calidad
- Evita spam
- Solo notificaciones importantes
- Agrupa mensajes similares
- Rate limiting inteligente
"""

import asyncio
import aiohttp
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

class NotificationPriority(Enum):
    """🚨 Prioridades de notificación"""
    LOW = "LOW"           # Info general, métricas
    MEDIUM = "MEDIUM"     # Trades normales
    HIGH = "HIGH"         # Ganancias/pérdidas importantes
    CRITICAL = "CRITICAL" # Errores, emergencias

@dataclass
class NotificationFilter:
    """🔍 Configuración de filtros"""
    min_trade_value_usd: float = 12.0          # Mínimo valor para notificar trade
    min_pnl_percent_notify: float = 2.0        # Mínimo PnL% para notificar
    max_notifications_per_hour: int = 10       # Máximo notificaciones/hora
    max_notifications_per_day: int = 50        # Máximo notificaciones/día
    suppress_similar_minutes: int = 15         # Suprimir similares por X minutos
    only_profitable_trades: bool = False       # Solo notificar trades rentables
    emergency_only_mode: bool = False          # Solo emergencias
    
class SmartDiscordNotifier:
    """🤖 Notificador Discord inteligente"""
    
    def __init__(self, webhook_url: str = None):
        self.webhook_url = webhook_url or os.getenv('DISCORD_WEBHOOK_URL')
        self.filters = NotificationFilter()
        
        # Control de rate limiting
        self.notification_history: List[Dict] = []
        self.last_similar_messages: Dict[str, datetime] = {}
        
        # Estadísticas
        self.stats = {
            'total_sent': 0,
            'filtered_out': 0,
            'rate_limited': 0,
            'errors': 0
        }
        
        print(f"🤖 Smart Discord Notifier {'habilitado' if self.webhook_url else 'deshabilitado'}")
    
    def configure_filters(self, **kwargs):
        """⚙️ Configurar filtros personalizados"""
        for key, value in kwargs.items():
            if hasattr(self.filters, key):
                setattr(self.filters, key, value)
                print(f"🔧 Filtro actualizado: {key} = {value}")
    
    async def send_trade_notification(self, trade_data: Dict) -> bool:
        """📈 Enviar notificación de trade con filtros inteligentes"""
        
        # Extraer datos del trade
        symbol = trade_data.get('symbol', 'UNKNOWN')
        side = trade_data.get('side', 'UNKNOWN')
        value_usd = trade_data.get('value_usd', 0)
        pnl_percent = trade_data.get('pnl_percent', 0)
        pnl_usd = trade_data.get('pnl_usd', 0)
        confidence = trade_data.get('confidence', 0)
        
        # 🔍 FILTRO 1: Valor mínimo
        if value_usd < self.filters.min_trade_value_usd:
            self._log_filtered(f"Trade muy pequeño: ${value_usd:.2f}")
            return False
        
        # 🔍 FILTRO 2: Solo trades rentables (si está activado)
        if self.filters.only_profitable_trades and pnl_usd <= 0:
            self._log_filtered(f"Trade no rentable: {pnl_percent:.2f}%")
            return False
        
        # 🔍 FILTRO 3: PnL mínimo
        if abs(pnl_percent) < self.filters.min_pnl_percent_notify and pnl_usd != 0:
            self._log_filtered(f"PnL muy bajo: {pnl_percent:.2f}%")
            return False
        
        # 🔍 FILTRO 4: Rate limiting
        if not self._check_rate_limits():
            self.stats['rate_limited'] += 1
            return False
        
        # 🔍 FILTRO 5: Mensajes similares recientes
        message_key = f"{symbol}_{side}"
        if self._is_similar_recent(message_key):
            self._log_filtered(f"Mensaje similar reciente: {message_key}")
            return False
        
        # Determinar prioridad
        priority = self._calculate_priority(trade_data)
        
        # Crear mensaje
        if pnl_usd == 0:  # Nueva posición
            message = self._format_new_position_message(trade_data, priority)
        else:  # Posición cerrada
            message = self._format_closed_position_message(trade_data, priority)
        
        # Enviar notificación
        success = await self._send_notification(message, priority)
        
        if success:
            self._record_notification(message_key)
        
        return success
    
    async def send_system_notification(self, message: str, priority: NotificationPriority = NotificationPriority.MEDIUM) -> bool:
        """🖥️ Enviar notificación del sistema"""
        
        # 🔍 FILTRO: Solo críticas en modo emergencia
        if self.filters.emergency_only_mode and priority != NotificationPriority.CRITICAL:
            return False
        
        # 🔍 FILTRO: Rate limiting
        if not self._check_rate_limits():
            return False
        
        formatted_message = f"🖥️ **SISTEMA** | {priority.value}\n{message}"
        
        return await self._send_notification(formatted_message, priority)
    
    async def send_daily_summary(self, summary_data: Dict) -> bool:
        """📊 Enviar resumen diario (solo si hay datos importantes)"""
        
        total_trades = summary_data.get('total_trades', 0)
        total_pnl = summary_data.get('total_pnl', 0)
        win_rate = summary_data.get('win_rate', 0)
        
        # 🔍 FILTRO: Solo enviar si hubo actividad significativa
        if total_trades == 0 and abs(total_pnl) < 1.0:
            return False
        
        emoji = "📈" if total_pnl > 0 else "📉" if total_pnl < 0 else "📊"
        
        message = f"{emoji} **RESUMEN DIARIO**\n"
        message += f"💼 Trades: {total_trades}\n"
        message += f"💰 PnL: ${total_pnl:+.2f}\n"
        message += f"🎯 Win Rate: {win_rate:.1f}%\n"
        message += f"⏰ {datetime.now().strftime('%Y-%m-%d')}"
        
        return await self._send_notification(message, NotificationPriority.MEDIUM)
    
    def _calculate_priority(self, trade_data: Dict) -> NotificationPriority:
        """🎯 Calcular prioridad de notificación"""
        
        pnl_percent = abs(trade_data.get('pnl_percent', 0))
        pnl_usd = abs(trade_data.get('pnl_usd', 0))
        value_usd = trade_data.get('value_usd', 0)
        
        # Prioridad CRITICAL: Pérdidas grandes
        if pnl_percent > 10 or pnl_usd > 20:
            return NotificationPriority.CRITICAL
        
        # Prioridad HIGH: Ganancias/pérdidas importantes
        if pnl_percent > 5 or pnl_usd > 10 or value_usd > 50:
            return NotificationPriority.HIGH
        
        # Prioridad MEDIUM: Trades normales
        if pnl_percent > 2 or value_usd > 15:
            return NotificationPriority.MEDIUM
        
        return NotificationPriority.LOW
    
    def _format_new_position_message(self, data: Dict, priority: NotificationPriority) -> str:
        """📝 Formatear mensaje de nueva posición"""
        
        priority_emoji = {
            NotificationPriority.LOW: "🟦",
            NotificationPriority.MEDIUM: "🟩", 
            NotificationPriority.HIGH: "🟨",
            NotificationPriority.CRITICAL: "🟥"
        }
        
        emoji = priority_emoji.get(priority, "🟩")
        
        message = f"{emoji} **NUEVA POSICIÓN**\n"
        message += f"📊 {data.get('symbol', 'UNKNOWN')}: {data.get('side', 'UNKNOWN')}\n"
        message += f"💰 Valor: ${data.get('value_usd', 0):.2f}\n"
        message += f"💲 Precio: ${data.get('price', 0):.4f}\n"
        message += f"🎯 Confianza: {data.get('confidence', 0):.1%}\n"
        message += f"⏰ {datetime.now().strftime('%H:%M:%S')}"
        
        return message
    
    def _format_closed_position_message(self, data: Dict, priority: NotificationPriority) -> str:
        """📝 Formatear mensaje de posición cerrada"""
        
        pnl_usd = data.get('pnl_usd', 0)
        emoji = "🟢" if pnl_usd > 0 else "🔴"
        
        message = f"{emoji} **POSICIÓN CERRADA**\n"
        message += f"📊 {data.get('symbol', 'UNKNOWN')}: {data.get('side', 'UNKNOWN')}\n"
        message += f"📈 PnL: {data.get('pnl_percent', 0):+.2f}% (${pnl_usd:+.2f})\n"
        message += f"🔄 Razón: {data.get('reason', 'UNKNOWN')}\n"
        message += f"⏰ {datetime.now().strftime('%H:%M:%S')}"
        
        return message
    
    def _check_rate_limits(self) -> bool:
        """⏱️ Verificar límites de rate"""
        
        now = datetime.now()
        
        # Limpiar historial antiguo
        self.notification_history = [
            n for n in self.notification_history 
            if now - n['timestamp'] < timedelta(days=1)
        ]
        
        # Contar notificaciones recientes
        hour_ago = now - timedelta(hours=1)
        recent_hour = sum(1 for n in self.notification_history if n['timestamp'] > hour_ago)
        
        daily_count = len(self.notification_history)
        
        # Verificar límites
        if recent_hour >= self.filters.max_notifications_per_hour:
            print(f"⏱️ Rate limit hora: {recent_hour}/{self.filters.max_notifications_per_hour}")
            return False
        
        if daily_count >= self.filters.max_notifications_per_day:
            print(f"⏱️ Rate limit día: {daily_count}/{self.filters.max_notifications_per_day}")
            return False
        
        return True
    
    def _is_similar_recent(self, message_key: str) -> bool:
        """🔄 Verificar si hay mensaje similar reciente"""
        
        if message_key not in self.last_similar_messages:
            return False
        
        last_time = self.last_similar_messages[message_key]
        time_diff = datetime.now() - last_time
        
        return time_diff < timedelta(minutes=self.filters.suppress_similar_minutes)
    
    def _record_notification(self, message_key: str):
        """📝 Registrar notificación enviada"""
        
        now = datetime.now()
        
        self.notification_history.append({
            'timestamp': now,
            'message_key': message_key
        })
        
        self.last_similar_messages[message_key] = now
        self.stats['total_sent'] += 1
    
    def _log_filtered(self, reason: str):
        """📊 Log de mensajes filtrados"""
        self.stats['filtered_out'] += 1
        print(f"🔇 Discord filtrado: {reason}")
    
    async def _send_notification(self, message: str, priority: NotificationPriority) -> bool:
        """📤 Enviar notificación real a Discord"""
        
        if not self.webhook_url:
            print(f"📢 Discord (sin webhook): {message[:50]}...")
            return True
        
        try:
            # Color según prioridad
            colors = {
                NotificationPriority.LOW: 0x3498db,      # Azul
                NotificationPriority.MEDIUM: 0x2ecc71,   # Verde
                NotificationPriority.HIGH: 0xf39c12,     # Naranja
                NotificationPriority.CRITICAL: 0xe74c3c  # Rojo
            }
            
            payload = {
                "embeds": [{
                    "description": message,
                    "color": colors.get(priority, 0x2ecc71),
                    "timestamp": datetime.now().isoformat(),
                    "footer": {
                        "text": f"Bot Profesional | {priority.value}"
                    }
                }]
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.webhook_url, json=payload) as response:
                    if response.status == 200:
                        print(f"✅ Discord enviado: {message[:50]}...")
                        return True
                    else:
                        print(f"❌ Discord error: {response.status}")
                        self.stats['errors'] += 1
                        return False
                        
        except Exception as e:
            print(f"❌ Error Discord: {e}")
            self.stats['errors'] += 1
            return False
    
    def get_stats(self) -> Dict:
        """📊 Obtener estadísticas del notificador"""
        return {
            **self.stats,
            'notifications_today': len(self.notification_history),
            'last_notification': max(
                (n['timestamp'] for n in self.notification_history), 
                default=None
            )
        }
    
    def print_stats(self):
        """📊 Mostrar estadísticas"""
        stats = self.get_stats()
        print(f"\n📊 ESTADÍSTICAS DISCORD:")
        print(f"   ✅ Enviadas: {stats['total_sent']}")
        print(f"   🔇 Filtradas: {stats['filtered_out']}")
        print(f"   ⏱️ Rate limited: {stats['rate_limited']}")
        print(f"   ❌ Errores: {stats['errors']}")
        print(f"   📅 Hoy: {stats['notifications_today']}")

# Ejemplo de uso
async def test_smart_notifier():
    """🧪 Test del notificador inteligente"""
    
    notifier = SmartDiscordNotifier()
    
    # Configurar filtros conservadores
    notifier.configure_filters(
        min_trade_value_usd=12.0,
        min_pnl_percent_notify=2.0,
        max_notifications_per_hour=5,
        only_profitable_trades=False
    )
    
    # Test trade normal (debería pasar)
    await notifier.send_trade_notification({
        'symbol': 'BTCUSDT',
        'side': 'BUY',
        'value_usd': 15.30,
        'price': 109500,
        'confidence': 0.85,
        'pnl_percent': 0,
        'pnl_usd': 0
    })
    
    # Test trade pequeño (debería filtrarse)
    await notifier.send_trade_notification({
        'symbol': 'ETHUSDT',
        'side': 'BUY', 
        'value_usd': 5.0,
        'price': 2770,
        'confidence': 0.75,
        'pnl_percent': 0,
        'pnl_usd': 0
    })
    
    notifier.print_stats()

if __name__ == "__main__":
    asyncio.run(test_smart_notifier()) 