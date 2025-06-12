#!/usr/bin/env python3
"""
🚨 SISTEMA DE TRADING REAL ARREGLADO
Usando el predictor que ya funciona correctamente
"""

import os
import asyncio
import aiohttp
import time
from datetime import datetime
from typing import Dict
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

class FixedBinanceConfig:
    """🔧 Configuración simplificada de Binance"""
    
    def __init__(self):
        self.api_key = os.getenv('BINANCE_API_KEY')
        self.secret_key = os.getenv('BINANCE_SECRET_KEY')
        self.environment = os.getenv('ENVIRONMENT', 'testnet')
        
        # URLs según el entorno
        if self.environment == 'production':
            self.base_url = os.getenv('BINANCE_PRODUCTION_URL', 'https://api.binance.com')
        else:
            self.base_url = os.getenv('BINANCE_TESTNET_URL', 'https://testnet.binance.vision')
        
        self.discord_webhook = os.getenv('DISCORD_WEBHOOK_URL')
        self.trade_mode = os.getenv('TRADE_MODE', 'dry_run')
        
        self._validate_config()
    
    def _validate_config(self):
        """✅ Validar configuración crítica"""
        if not self.api_key or self.api_key == 'tu_api_key_de_binance_aqui':
            raise ValueError("❌ BINANCE_API_KEY no configurada. Configura tu .env")
        
        if not self.secret_key or self.secret_key == 'tu_secret_key_de_binance_aqui':
            raise ValueError("❌ BINANCE_SECRET_KEY no configurada. Configura tu .env")
        
        print(f"✅ Configuración validada:")
        print(f"   🌍 Entorno: {self.environment}")
        print(f"   📊 Modo trading: {self.trade_mode}")
        print(f"   🔑 API Key: {self.api_key[:8]}...")

class FixedRealTradingBot:
    """🤖 Bot de Trading Real usando predictor probado"""
    
    def __init__(self):
        self.config = FixedBinanceConfig()
        self.symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
        self.trading_stats = {
            'total_trades': 0,
            'total_predictions': 0,
            'start_time': datetime.now()
        }
        
        # Importar el predictor que ya funciona
        try:
            from final_real_binance_predictor import OptimizedTCNPredictor, OptimizedBinanceData
            self.predictor = OptimizedTCNPredictor()
            self.data_provider = OptimizedBinanceData()
            print("✅ Predictor TCN cargado exitosamente")
        except Exception as e:
            print(f"❌ Error cargando predictor: {e}")
            self.predictor = None
            self.data_provider = None
    
    async def send_discord_notification(self, message: str):
        """💬 Enviar notificación a Discord"""
        if not self.config.discord_webhook:
            print(f"📢 Discord (sin webhook): {message[:50]}...")
            return
        
        try:
            async with aiohttp.ClientSession() as session:
                payload = {"content": message}
                async with session.post(self.config.discord_webhook, json=payload) as response:
                    if response.status in [200, 204]:
                        print(f"✅ Discord: {message[:50]}...")
        except Exception as e:
            print(f"❌ Error Discord: {e}")
    
    async def get_account_balance(self) -> float:
        """💰 Obtener balance USDT de la cuenta"""
        try:
            import hmac
            import hashlib
            
            timestamp = int(time.time() * 1000)
            query_string = f"timestamp={timestamp}"
            signature = hmac.new(
                self.config.secret_key.encode('utf-8'),
                query_string.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            
            url = f"{self.config.base_url}/api/v3/account"
            headers = {'X-MBX-APIKEY': self.config.api_key}
            params = {'timestamp': timestamp, 'signature': signature}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        for balance in data['balances']:
                            if balance['asset'] == 'USDT':
                                return float(balance['free'])
                    else:
                        print(f"❌ Error obteniendo balance: {response.status}")
        except Exception as e:
            print(f"❌ Error balance: {e}")
        
        return 0.0
    
    async def run_trading_session(self, duration_minutes: int = 30):
        """🚀 Ejecutar sesión de trading"""
        
        print(f"\n🤖 INICIANDO TRADING REAL ARREGLADO")
        print(f"⏱️  Duración: {duration_minutes} minutos")
        print(f"🌍 Entorno: {self.config.environment}")
        print(f"📊 Modo: {self.config.trade_mode}")
        print("=" * 50)
        
        if not self.predictor:
            print("❌ Predictor no disponible")
            return
        
        # Verificar balance
        balance = await self.get_account_balance()
        print(f"💰 Balance USDT: ${balance:.2f}")
        
        await self.send_discord_notification(
            f"🚀 **BOT TRADING INICIADO**\n\n"
            f"🌍 **Entorno**: {self.config.environment}\n"
            f"📊 **Modo**: {self.config.trade_mode}\n"
            f"⏱️ **Duración**: {duration_minutes}min\n"
            f"💰 **Balance**: ${balance:.2f}"
        )
        
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        cycle = 0
        
        try:
            # Usar el data provider que ya funciona
            async with self.data_provider as provider:
                
                while time.time() < end_time:
                    cycle += 1
                    print(f"\n🔄 Ciclo {cycle} - {datetime.now().strftime('%H:%M:%S')}")
                    
                    for symbol in self.symbols:
                        try:
                            # Obtener datos del mercado
                            market_data = await provider.get_market_data(symbol)
                            
                            if market_data and market_data.get('klines'):
                                # Generar predicción
                                prediction = await self.predictor.predict_real_market(symbol, market_data)
                                
                                if prediction:
                                    signal = prediction['signal']
                                    confidence = prediction['confidence']
                                    
                                    self.trading_stats['total_predictions'] += 1
                                    print(f"📊 {symbol}: {signal} ({confidence:.1%})")
                            
                        except Exception as e:
                            print(f"❌ Error {symbol}: {e}")
                    
                    # Pausa entre ciclos (60 segundos)
                    if time.time() < end_time:
                        await asyncio.sleep(60)
                        
        except KeyboardInterrupt:
            print("\n⏹️  Sesión interrumpida por usuario")
        except Exception as e:
            print(f"\n❌ Error en sesión: {e}")
        
        # Resumen final
        duration = time.time() - start_time
        print(f"\n📊 RESUMEN FINAL:")
        print(f"   ⏱️  Duración: {duration/60:.1f} minutos")
        print(f"   📊 Predicciones: {self.trading_stats['total_predictions']}")

async def main():
    """🚀 Función principal"""
    print("🤖 SISTEMA DE TRADING REAL ARREGLADO")
    print("=" * 40)
    
    try:
        # Verificar archivo .env
        if not os.path.exists('.env'):
            print("❌ Archivo .env no encontrado")
            print("📝 Copia env_example a .env y configura tus API keys")
            return
        
        bot = FixedRealTradingBot()
        
        # Ejecutar sesión de 30 minutos
        await bot.run_trading_session(30)
            
    except ValueError as e:
        print(f"⚠️  Error de configuración: {e}")
        print("📝 Revisa tu archivo .env")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 