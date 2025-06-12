#!/usr/bin/env python3
"""
🚨 SISTEMA DE TRADING REAL CON APIs DE BINANCE
Sistema de configuración y trading real usando credenciales del archivo .env
"""

import os
import asyncio
import aiohttp
import time
from decimal import Decimal, ROUND_DOWN
import hmac
import hashlib
import json
from datetime import datetime
from typing import Dict, Optional, List, Tuple
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import tensorflow as tf

# Cargar variables de entorno
load_dotenv()

class BinanceConfig:
    """🔧 Configuración de Binance API"""
    
    def __init__(self):
        self.api_key = os.getenv('BINANCE_API_KEY')
        self.secret_key = os.getenv('BINANCE_SECRET_KEY')
        self.environment = os.getenv('ENVIRONMENT', 'testnet')
        
        # URLs según el entorno
        if self.environment == 'production':
            self.base_url = os.getenv('BINANCE_PRODUCTION_URL', 'https://api.binance.com')
        else:
            self.base_url = os.getenv('BINANCE_TESTNET_URL', 'https://testnet.binance.vision')
        
        # Discord
        self.discord_webhook = os.getenv('DISCORD_WEBHOOK_URL')
        
        # Configuración de trading
        self.max_position_percent = float(os.getenv('MAX_POSITION_SIZE_PERCENT', '5'))
        self.max_daily_loss_percent = float(os.getenv('MAX_DAILY_LOSS_PERCENT', '3'))
        self.min_confidence = float(os.getenv('MIN_CONFIDENCE_THRESHOLD', '0.70'))
        self.trade_mode = os.getenv('TRADE_MODE', 'dry_run')
        
        # Validar configuración
        self._validate_config()
    
    def _validate_config(self):
        """✅ Validar configuración crítica"""
        if not self.api_key or self.api_key == 'tu_api_key_de_binance_aqui':
            raise ValueError("❌ BINANCE_API_KEY no configurada. Configura tu .env")
        
        if not self.secret_key or self.secret_key == 'tu_secret_key_de_binance_aqui':
            raise ValueError("❌ BINANCE_SECRET_KEY no configurada. Configura tu .env")
        
        if self.environment not in ['testnet', 'production']:
            raise ValueError("❌ ENVIRONMENT debe ser 'testnet' o 'production'")
        
        if self.trade_mode not in ['dry_run', 'real']:
            raise ValueError("❌ TRADE_MODE debe ser 'dry_run' o 'real'")
        
        print(f"✅ Configuración validada:")
        print(f"   🌍 Entorno: {self.environment}")
        print(f"   📊 Modo trading: {self.trade_mode}")
        print(f"   🔑 API Key: {self.api_key[:8]}...")
        print(f"   🌐 Base URL: {self.base_url}")

class BinanceRealAPI:
    """🏦 API Real de Binance con autenticación"""
    
    def __init__(self, config: BinanceConfig):
        self.config = config
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def _generate_signature(self, query_string: str) -> str:
        """🔐 Generar firma HMAC para autenticación"""
        return hmac.new(
            self.config.secret_key.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    def _get_headers(self) -> Dict[str, str]:
        """📋 Headers para requests autenticados"""
        return {
            'X-MBX-APIKEY': self.config.api_key,
            'Content-Type': 'application/json'
        }
    
    async def get_account_info(self) -> Dict:
        """💰 Obtener información de la cuenta"""
        timestamp = int(time.time() * 1000)
        query_string = f"timestamp={timestamp}"
        signature = self._generate_signature(query_string)
        
        url = f"{self.config.base_url}/api/v3/account"
        params = {
            'timestamp': timestamp,
            'signature': signature
        }
        
        async with self.session.get(url, headers=self._get_headers(), params=params) as response:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                raise Exception(f"❌ Error obteniendo info cuenta: {response.status} - {error_text}")
    
    async def get_symbol_price(self, symbol: str) -> float:
        """💲 Precio actual de un símbolo"""
        url = f"{self.config.base_url}/api/v3/ticker/price"
        params = {'symbol': symbol}
        
        async with self.session.get(url, params=params) as response:
            if response.status == 200:
                data = await response.json()
                return float(data['price'])
            else:
                raise Exception(f"❌ Error obteniendo precio {symbol}: {response.status}")
    
    async def place_order(self, symbol: str, side: str, quantity: str, order_type: str = 'MARKET') -> Dict:
        """🛒 Colocar orden de trading"""
        if self.config.trade_mode == 'dry_run':
            # Simulación
            price = await self.get_symbol_price(symbol)
            return {
                'symbol': symbol,
                'orderId': int(time.time()),
                'side': side,
                'type': order_type,
                'origQty': quantity,
                'price': str(price),
                'status': 'FILLED',
                'timeInForce': 'GTC',
                'fills': [{'price': str(price), 'qty': quantity}],
                'dry_run': True
            }
        
        # Orden real
        timestamp = int(time.time() * 1000)
        params = {
            'symbol': symbol,
            'side': side,
            'type': order_type,
            'quantity': quantity,
            'timestamp': timestamp
        }
        
        query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
        signature = self._generate_signature(query_string)
        params['signature'] = signature
        
        url = f"{self.config.base_url}/api/v3/order"
        
        async with self.session.post(url, headers=self._get_headers(), data=params) as response:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                raise Exception(f"❌ Error colocando orden: {response.status} - {error_text}")

class RealTradingBot:
    """🤖 Bot de Trading Real con TCN y APIs de Binance"""
    
    def __init__(self):
        self.config = BinanceConfig()
        self.models = {}
        self.symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
        self.trading_stats = {
            'total_trades': 0,
            'profitable_trades': 0,
            'total_pnl': 0.0,
            'daily_trades': 0,
            'daily_pnl': 0.0,
            'start_time': datetime.now()
        }
        
    async def load_tcn_models(self):
        """🧠 Cargar modelos TCN entrenados compatibles"""
        print("🔄 Cargando modelos TCN compatibles...")
        
        # Intentar diferentes versiones de modelos
        model_candidates = [
            "ultra_model_{}.h5",      # Input: (50, 21) 
            "best_model_{}.h5",       # Alternativos
            "production_model_{}.h5"  # Última opción
        ]
        
        for symbol in self.symbols:
            model_loaded = False
            
            for model_template in model_candidates:
                try:
                    model_path = model_template.format(symbol)
                    if os.path.exists(model_path):
                        test_model = tf.keras.models.load_model(model_path)
                        input_shape = test_model.input_shape
                        
                        print(f"  🔍 {model_path}: {input_shape}")
                        
                        # Verificar compatibilidad con nuestro formato (None, 50, 21)
                        if len(input_shape) == 3 and input_shape[1] == 50 and input_shape[2] == 21:
                            self.models[symbol] = test_model
                            print(f"✅ {symbol}: Modelo compatible cargado ({model_path})")
                            model_loaded = True
                            break
                        else:
                            print(f"  ⚠️  Incompatible: esperamos (None, 50, 21)")
                            del test_model
                            
                except Exception as e:
                    print(f"  ❌ Error con {model_path}: {e}")
            
            if not model_loaded:
                print(f"⚠️  {symbol}: Ningún modelo compatible encontrado")
        
        print(f"📊 Modelos compatibles cargados: {len(self.models)}/{len(self.symbols)}")
        
        # Si no hay modelos, crear modelos temporales
        if len(self.models) == 0:
            print("🔧 Creando modelos temporales de respaldo...")
            await self._create_backup_models()
    
    async def _create_backup_models(self):
        """🔧 Crear modelos de respaldo si no hay compatibles"""
        try:
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Dense, Dropout, LSTM, Input
            
            for symbol in self.symbols:
                model = Sequential([
                    Input(shape=(50, 21)),
                    LSTM(64, return_sequences=True),
                    Dropout(0.2),
                    LSTM(32),
                    Dropout(0.2),
                    Dense(16, activation='relu'),
                    Dense(3, activation='softmax')  # 3 clases: SELL, HOLD, BUY
                ])
                
                model.compile(optimizer='adam', loss='categorical_crossentropy')
                self.models[symbol] = model
                print(f"  🔧 {symbol}: Modelo de respaldo creado")
                
        except Exception as e:
            print(f"❌ Error creando modelos de respaldo: {e}")
    
    async def get_market_data(self, symbol: str, interval: str = '1m', limit: int = 60) -> pd.DataFrame:
        """📈 Obtener datos del mercado"""
        async with BinanceRealAPI(self.config) as api:
            url = f"{self.config.base_url}/api/v3/klines"
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': limit
            }
            
            async with api.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    df = pd.DataFrame(data, columns=[
                        'timestamp', 'open', 'high', 'low', 'close', 'volume',
                        'close_time', 'quote_asset_volume', 'number_of_trades',
                        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                    ])
                    
                    # Convertir a tipos numéricos
                    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
                    for col in numeric_cols:
                        df[col] = pd.to_numeric(df[col])
                    
                    return df
                else:
                    raise Exception(f"❌ Error obteniendo datos {symbol}: {response.status}")
    
    def calculate_technical_features(self, df: pd.DataFrame) -> np.ndarray:
        """🔧 Calcular indicadores técnicos (21 features)"""
        data = df.copy()
        
        # Indicadores básicos
        data['sma_20'] = data['close'].rolling(20).mean()
        data['ema_12'] = data['close'].ewm(span=12).mean()
        data['ema_26'] = data['close'].ewm(span=26).mean()
        
        # RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        data['macd'] = data['ema_12'] - data['ema_26']
        data['macd_signal'] = data['macd'].ewm(span=9).mean()
        data['macd_histogram'] = data['macd'] - data['macd_signal']
        
        # Bollinger Bands
        bb_ma = data['close'].rolling(20).mean()
        bb_std = data['close'].rolling(20).std()
        data['bb_upper'] = bb_ma + (bb_std * 2)
        data['bb_lower'] = bb_ma - (bb_std * 2)
        data['bb_position'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
        
        # Volumen
        data['volume_sma'] = data['volume'].rolling(20).mean()
        data['volume_ratio'] = data['volume'] / data['volume_sma']
        
        # Price features
        data['price_change'] = data['close'].pct_change()
        data['volatility'] = data['price_change'].rolling(10).std()
        data['high_low_ratio'] = (data['high'] - data['low']) / data['close']
        
        # Momentum
        data['momentum_10'] = data['close'] / data['close'].shift(10)
        data['momentum_5'] = data['close'] / data['close'].shift(5)
        
        # Williams %R
        highest_high = data['high'].rolling(14).max()
        lowest_low = data['low'].rolling(14).min()
        data['williams_r'] = -100 * (highest_high - data['close']) / (highest_high - lowest_low)
        
        # Stochastic
        data['stoch_k'] = 100 * (data['close'] - lowest_low) / (highest_high - lowest_low)
        data['stoch_d'] = data['stoch_k'].rolling(3).mean()
        
        # ADX aproximado
        tr1 = data['high'] - data['low']
        tr2 = abs(data['high'] - data['close'].shift())
        tr3 = abs(data['low'] - data['close'].shift())
        data['atr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(14).mean()
        
        # Seleccionar features finales (21 características)
        feature_columns = [
            'open', 'high', 'low', 'close', 'volume',
            'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd',
            'macd_signal', 'macd_histogram', 'bb_upper', 'bb_lower', 'bb_position',
            'volume_ratio', 'price_change', 'volatility', 'high_low_ratio',
            'momentum_10', 'williams_r'
        ]
        
        # Rellenar NaN y normalizar
        for col in feature_columns:
            if col in data.columns:
                data[col] = data[col].fillna(data[col].mean())
            else:
                data[col] = 0
        
        # Normalización min-max simple
        feature_array = data[feature_columns].values
        for i in range(feature_array.shape[1]):
            col_data = feature_array[:, i]
            col_min, col_max = np.nanmin(col_data), np.nanmax(col_data)
            if col_max > col_min:
                feature_array[:, i] = (col_data - col_min) / (col_max - col_min)
        
        return feature_array
    
    async def predict_signal(self, symbol: str) -> Tuple[str, float]:
        """🎯 Predecir señal de trading"""
        if symbol not in self.models:
            return "HOLD", 0.0
        
        try:
            # Obtener datos
            df = await self.get_market_data(symbol, limit=60)
            features = self.calculate_technical_features(df)
            
            # Preparar input para el modelo (últimos 50 puntos)
            if len(features) >= 50:
                X = features[-50:].reshape(1, 50, 21)
                
                # Predicción
                prediction = self.models[symbol].predict(X, verbose=0)[0]
                
                # Interpretar predicción (asumiendo 3 clases: SELL, HOLD, BUY)
                class_idx = np.argmax(prediction)
                confidence = float(prediction[class_idx])
                
                signal_map = {0: "SELL", 1: "HOLD", 2: "BUY"}
                signal = signal_map.get(class_idx, "HOLD")
                
                return signal, confidence
            
        except Exception as e:
            print(f"❌ Error prediciendo {symbol}: {e}")
            
        return "HOLD", 0.0
    
    async def send_discord_notification(self, message: str):
        """💬 Enviar notificación a Discord"""
        if not self.config.discord_webhook:
            return
        
        try:
            async with aiohttp.ClientSession() as session:
                payload = {"content": message}
                async with session.post(self.config.discord_webhook, json=payload) as response:
                    if response.status == 200:
                        print(f"✅ Discord: {message[:50]}...")
        except Exception as e:
            print(f"❌ Error Discord: {e}")
    
    async def execute_trade(self, symbol: str, signal: str, confidence: float):
        """💼 Ejecutar trade según señal"""
        if signal == "HOLD" or confidence < self.config.min_confidence:
            return
        
        try:
            async with BinanceRealAPI(self.config) as api:
                # Obtener información de la cuenta
                account_info = await api.get_account_info()
                usdt_balance = 0
                
                for balance in account_info['balances']:
                    if balance['asset'] == 'USDT':
                        usdt_balance = float(balance['free'])
                        break
                
                if usdt_balance < 10:  # Mínimo $10
                    print(f"⚠️  Balance insuficiente: ${usdt_balance:.2f}")
                    return
                
                # Calcular cantidad
                price = await api.get_symbol_price(symbol)
                position_value = usdt_balance * (self.config.max_position_percent / 100)
                quantity = position_value / price
                
                # Redondear cantidad según el símbolo (simplificado)
                if 'BTC' in symbol:
                    quantity = round(quantity, 6)
                else:
                    quantity = round(quantity, 4)
                
                # Ejecutar orden
                side = 'BUY' if signal == 'BUY' else 'SELL'
                order = await api.place_order(symbol, side, str(quantity))
                
                # Estadísticas
                self.trading_stats['total_trades'] += 1
                self.trading_stats['daily_trades'] += 1
                
                # Notificación
                mode_emoji = "🔄" if self.config.trade_mode == 'dry_run' else "💰"
                message = (
                    f"{mode_emoji} **TRADE EJECUTADO**\n"
                    f"📊 **{symbol}**: {signal}\n"
                    f"🎯 **Confianza**: {confidence:.1%}\n"
                    f"💵 **Cantidad**: {quantity}\n"
                    f"💲 **Precio**: ${price:.4f}\n"
                    f"🔢 **Order ID**: {order.get('orderId', 'N/A')}\n"
                    f"⚡ **Modo**: {self.config.trade_mode.upper()}"
                )
                
                await self.send_discord_notification(message)
                print(f"✅ Trade ejecutado: {signal} {symbol} - Confianza: {confidence:.1%}")
                
        except Exception as e:
            print(f"❌ Error ejecutando trade {symbol}: {e}")
            await self.send_discord_notification(f"❌ Error en trade {symbol}: {str(e)[:100]}")
    
    async def run_trading_session(self, duration_minutes: int = 60):
        """🚀 Ejecutar sesión de trading"""
        print(f"\n🤖 INICIANDO SESIÓN DE TRADING REAL")
        print(f"⏱️  Duración: {duration_minutes} minutos")
        print(f"🌍 Entorno: {self.config.environment}")
        print(f"📊 Modo: {self.config.trade_mode}")
        print("=" * 50)
        
        await self.load_tcn_models()
        await self.send_discord_notification(
            f"🚀 **BOT INICIADO**\n"
            f"🌍 Entorno: {self.config.environment}\n"
            f"📊 Modo: {self.config.trade_mode}\n"
            f"⏱️ Duración: {duration_minutes}min\n"
            f"🧠 Modelos: {len(self.models)}"
        )
        
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        cycle = 0
        
        try:
            while time.time() < end_time:
                cycle += 1
                print(f"\n🔄 Ciclo {cycle} - {datetime.now().strftime('%H:%M:%S')}")
                
                for symbol in self.symbols:
                    if symbol in self.models:
                        signal, confidence = await self.predict_signal(symbol)
                        print(f"📊 {symbol}: {signal} ({confidence:.1%})")
                        
                        # Ejecutar trade si cumple criterios
                        if signal != "HOLD" and confidence >= self.config.min_confidence:
                            await self.execute_trade(symbol, signal, confidence)
                
                # Pausa entre ciclos
                await asyncio.sleep(60)  # 1 minuto
                
        except KeyboardInterrupt:
            print("\n⏹️  Sesión interrumpida por usuario")
        except Exception as e:
            print(f"\n❌ Error en sesión: {e}")
        
        # Resumen final
        duration = time.time() - start_time
        await self.send_discord_notification(
            f"🏁 **SESIÓN FINALIZADA**\n"
            f"⏱️ Duración: {duration/60:.1f}min\n"
            f"📈 Trades: {self.trading_stats['total_trades']}\n"
            f"💰 PnL: ${self.trading_stats['total_pnl']:.2f}"
        )
        
        print(f"\n📊 RESUMEN FINAL:")
        print(f"   ⏱️  Duración: {duration/60:.1f} minutos")
        print(f"   📈 Trades totales: {self.trading_stats['total_trades']}")
        print(f"   💰 PnL total: ${self.trading_stats['total_pnl']:.2f}")

async def main():
    """🚀 Función principal"""
    print("🤖 SISTEMA DE TRADING REAL CON TCN")
    print("=" * 40)
    
    try:
        # Verificar archivo .env
        if not os.path.exists('.env'):
            print("❌ Archivo .env no encontrado")
            print("📝 Copia env_example a .env y configura tus API keys")
            return
        
        bot = RealTradingBot()
        
        # Menú
        print("\n📋 OPCIONES DISPONIBLES:")
        print("1️⃣  Verificar configuración")
        print("2️⃣  Test de conexión Binance")
        print("3️⃣  Sesión de trading (30 min)")
        print("4️⃣  Sesión de trading (60 min)")
        print("5️⃣  Monitoreo continuo")
        
        choice = input("\n🔢 Selecciona una opción (1-5): ").strip()
        
        if choice == "1":
            print("✅ Configuración verificada exitosamente")
            
        elif choice == "2":
            print("\n🔄 Probando conexión a Binance...")
            async with BinanceRealAPI(bot.config) as api:
                account_info = await api.get_account_info()
                print(f"✅ Conexión exitosa")
                print(f"📊 Balances disponibles: {len(account_info['balances'])} assets")
                
                # Mostrar algunos balances
                for balance in account_info['balances'][:5]:
                    if float(balance['free']) > 0:
                        print(f"   💰 {balance['asset']}: {balance['free']}")
                
        elif choice == "3":
            await bot.run_trading_session(30)
            
        elif choice == "4":
            await bot.run_trading_session(60)
            
        elif choice == "5":
            print("🔄 Iniciando monitoreo continuo...")
            await bot.run_trading_session(24 * 60)  # 24 horas
            
        else:
            print("❌ Opción no válida")
            
    except ValueError as e:
        print(f"⚠️  Error de configuración: {e}")
        print("📝 Revisa tu archivo .env")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 