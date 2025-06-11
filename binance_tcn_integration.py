#!/usr/bin/env python3
"""
BINANCE TCN INTEGRATION - Sistema completo de trading automatizado
Integración final del TCN optimizado con Binance API para trading real
"""

import os
import time
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime, timedelta
import asyncio
import aiohttp
import hmac
import hashlib
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from sklearn.preprocessing import RobustScaler
import warnings
warnings.filterwarnings('ignore')

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('binance_tcn_trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class TradingConfig:
    """Configuración del sistema de trading"""
    # Pares de trading
    trading_pairs: List[str] = None
    
    # Configuración de modelo
    sequence_length: int = 50
    prediction_interval: int = 60  # segundos
    
    # Configuración de trading
    max_position_size: float = 0.1  # 10% del balance
    min_confidence: float = 0.75   # Confianza mínima para trade
    min_accuracy: float = 0.60     # Accuracy mínima requerida
    
    # Gestión de riesgo
    stop_loss_pct: float = 0.02    # 2% stop loss
    take_profit_pct: float = 0.04  # 4% take profit
    max_daily_trades: int = 10     # Máximo trades por día
    
    # Configuración API
    api_key: str = ""
    api_secret: str = ""
    testnet: bool = True  # Usar testnet inicialmente
    
    def __post_init__(self):
        if self.trading_pairs is None:
            self.trading_pairs = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]

class BinanceConnector:
    """Conector para Binance API"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.base_url = "https://testnet.binance.vision" if config.testnet else "https://api.binance.com"
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def _generate_signature(self, query_string: str) -> str:
        """Generar firma para requests autenticados"""
        return hmac.new(
            self.config.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    async def get_klines(self, symbol: str, interval: str = "1m", limit: int = 500) -> List[Dict]:
        """Obtener datos de velas"""
        url = f"{self.base_url}/api/v3/klines"
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        
        async with self.session.get(url, params=params) as response:
            if response.status == 200:
                data = await response.json()
                return [{
                    'timestamp': int(item[0]),
                    'open': float(item[1]),
                    'high': float(item[2]),
                    'low': float(item[3]),
                    'close': float(item[4]),
                    'volume': float(item[5])
                } for item in data]
            else:
                logger.error(f"Error obteniendo klines: {response.status}")
                return []
    
    async def get_account_info(self) -> Dict:
        """Obtener información de la cuenta"""
        timestamp = int(time.time() * 1000)
        query_string = f"timestamp={timestamp}"
        signature = self._generate_signature(query_string)
        
        url = f"{self.base_url}/api/v3/account"
        headers = {"X-MBX-APIKEY": self.config.api_key}
        params = {
            "timestamp": timestamp,
            "signature": signature
        }
        
        async with self.session.get(url, headers=headers, params=params) as response:
            if response.status == 200:
                return await response.json()
            else:
                logger.error(f"Error obteniendo info cuenta: {response.status}")
                return {}
    
    async def place_order(self, symbol: str, side: str, quantity: float, 
                         order_type: str = "MARKET") -> Dict:
        """Colocar orden"""
        timestamp = int(time.time() * 1000)
        
        params = {
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "quantity": quantity,
            "timestamp": timestamp
        }
        
        query_string = "&".join([f"{k}={v}" for k, v in params.items()])
        signature = self._generate_signature(query_string)
        params["signature"] = signature
        
        url = f"{self.base_url}/api/v3/order"
        headers = {"X-MBX-APIKEY": self.config.api_key}
        
        async with self.session.post(url, headers=headers, data=params) as response:
            result = await response.json()
            if response.status == 200:
                logger.info(f"Orden colocada: {symbol} {side} {quantity}")
                return result
            else:
                logger.error(f"Error colocando orden: {result}")
                return {}

class TCNPredictor:
    """Predictor TCN optimizado para trading real"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.models = {}
        self.scalers = {}
        self.features_cache = {}
        
    def load_model(self, pair: str) -> bool:
        """Cargar modelo entrenado para un par"""
        try:
            # Cargar modelo guardado (implementar después del entrenamiento)
            model_path = f"models/tcn_final_{pair.lower()}.h5"
            if os.path.exists(model_path):
                self.models[pair] = tf.keras.models.load_model(model_path)
                logger.info(f"Modelo cargado para {pair}")
                return True
            else:
                # Por ahora usar el modelo final del training anterior
                logger.warning(f"Modelo no encontrado para {pair}, usando configuración por defecto")
                return self._build_default_model(pair)
        except Exception as e:
            logger.error(f"Error cargando modelo {pair}: {e}")
            return False
    
    def _build_default_model(self, pair: str) -> bool:
        """Construir modelo por defecto basado en la arquitectura final"""
        try:
            from tcn_final_ready import FinalReadyTCN
            
            tcn = FinalReadyTCN(pair_name=pair)
            input_shape = (self.config.sequence_length, 21)  # 21 features
            model = tcn.build_confidence_model(input_shape)
            
            self.models[pair] = model
            logger.info(f"Modelo por defecto creado para {pair}")
            return True
        except Exception as e:
            logger.error(f"Error creando modelo por defecto {pair}: {e}")
            return False
    
    def create_features(self, klines_data: List[Dict]) -> pd.DataFrame:
        """Crear features para predicción"""
        if len(klines_data) < 100:
            return pd.DataFrame()
        
        df = pd.DataFrame(klines_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        features = pd.DataFrame(index=df.index)
        
        # Returns básicos
        for period in [1, 3, 5, 10, 20]:
            returns = df['close'].pct_change(period)
            features[f'returns_{period}'] = returns
            features[f'returns_{period}_ma'] = returns.rolling(5).mean()
        
        # Volatilidad
        for window in [10, 20, 50]:
            vol = df['close'].pct_change().rolling(window).std()
            features[f'vol_{window}'] = vol
        
        # Trend
        for short, long in [(10, 30), (20, 60)]:
            sma_s = df['close'].rolling(short).mean()
            sma_l = df['close'].rolling(long).mean()
            features[f'trend_{short}_{long}'] = (sma_s - sma_l) / df['close']
        
        # RSI
        features['rsi_14'] = self._calculate_rsi(df['close'], 14)
        features['rsi_neutral'] = abs(features['rsi_14'] - 50) / 50
        
        # MACD
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        features['macd'] = (ema12 - ema26) / df['close']
        features['macd_signal'] = features['macd'].ewm(span=9).mean()
        
        # Bollinger Bands
        bb_mid = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        features['bb_position'] = (df['close'] - bb_mid) / (2 * bb_std)
        
        # Volume
        features['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        
        return features.fillna(0)
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calcular RSI"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=window).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    async def predict(self, pair: str, features: pd.DataFrame) -> Optional[Dict]:
        """Realizar predicción"""
        if pair not in self.models or features.empty:
            return None
        
        try:
            # Preparar secuencia
            if len(features) < self.config.sequence_length:
                return None
            
            # Normalizar features
            if pair not in self.scalers:
                self.scalers[pair] = {}
                for col in features.columns:
                    scaler = RobustScaler()
                    self.scalers[pair][col] = scaler
                    features[col] = scaler.fit_transform(features[col].values.reshape(-1, 1)).flatten()
            else:
                for col in features.columns:
                    if col in self.scalers[pair]:
                        features[col] = self.scalers[pair][col].transform(features[col].values.reshape(-1, 1)).flatten()
            
            # Crear secuencia
            sequence = features.iloc[-self.config.sequence_length:].values
            sequence = np.expand_dims(sequence, axis=0)
            
            # Predicción
            prediction = self.models[pair].predict(sequence, verbose=0)
            probabilities = prediction[0]
            
            predicted_class = np.argmax(probabilities)
            confidence = float(np.max(probabilities))
            
            class_names = ['SELL', 'HOLD', 'BUY']
            signal = class_names[predicted_class]
            
            return {
                'signal': signal,
                'confidence': confidence,
                'probabilities': {
                    'SELL': float(probabilities[0]),
                    'HOLD': float(probabilities[1]),
                    'BUY': float(probabilities[2])
                },
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error en predicción {pair}: {e}")
            return None

class RiskManager:
    """Gestor de riesgo para trading"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.daily_trades = {}
        self.open_positions = {}
        self.daily_pnl = 0.0
        
    def can_trade(self, pair: str, signal: str, confidence: float) -> bool:
        """Verificar si se puede realizar el trade"""
        today = datetime.now().date()
        
        # Verificar límite diario de trades
        if today not in self.daily_trades:
            self.daily_trades[today] = 0
        
        if self.daily_trades[today] >= self.config.max_daily_trades:
            logger.info(f"Límite diario de trades alcanzado: {self.daily_trades[today]}")
            return False
        
        # Verificar confianza mínima
        if confidence < self.config.min_confidence:
            logger.info(f"Confianza insuficiente {pair}: {confidence:.3f} < {self.config.min_confidence}")
            return False
        
        # Verificar posición existente
        if pair in self.open_positions:
            logger.info(f"Posición ya abierta para {pair}")
            return False
        
        # Solo HOLD no genera trades
        if signal == 'HOLD':
            return False
            
        return True
    
    def calculate_position_size(self, pair: str, account_balance: float, current_price: float) -> float:
        """Calcular tamaño de posición"""
        max_position_value = account_balance * self.config.max_position_size
        position_size = max_position_value / current_price
        
        # Redondear a precisión de Binance (simplificado)
        return round(position_size, 6)
    
    def register_trade(self, pair: str, side: str, quantity: float, price: float):
        """Registrar trade realizado"""
        today = datetime.now().date()
        if today not in self.daily_trades:
            self.daily_trades[today] = 0
        self.daily_trades[today] += 1
        
        self.open_positions[pair] = {
            'side': side,
            'quantity': quantity,
            'entry_price': price,
            'timestamp': datetime.now()
        }
        
        logger.info(f"Trade registrado: {pair} {side} {quantity} @ {price}")

class BinanceTCNTrader:
    """Sistema principal de trading automatizado"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.connector = None
        self.predictor = TCNPredictor(config)
        self.risk_manager = RiskManager(config)
        self.running = False
        
    async def initialize(self):
        """Inicializar sistema"""
        logger.info("Inicializando Binance TCN Trader...")
        
        # Verificar configuración
        if not self.config.api_key or not self.config.api_secret:
            logger.error("API key y secret requeridos")
            return False
        
        # Cargar modelos
        for pair in self.config.trading_pairs:
            if not self.predictor.load_model(pair):
                logger.error(f"Error cargando modelo para {pair}")
                return False
        
        logger.info("Sistema inicializado correctamente")
        return True
    
    async def run_trading_loop(self):
        """Bucle principal de trading"""
        logger.info("Iniciando bucle de trading...")
        self.running = True
        
        async with BinanceConnector(self.config) as connector:
            self.connector = connector
            
            while self.running:
                try:
                    # Procesar cada par
                    for pair in self.config.trading_pairs:
                        await self.process_pair(pair)
                    
                    # Esperar antes del siguiente ciclo
                    await asyncio.sleep(self.config.prediction_interval)
                    
                except Exception as e:
                    logger.error(f"Error en bucle principal: {e}")
                    await asyncio.sleep(10)
    
    async def process_pair(self, pair: str):
        """Procesar un par específico"""
        try:
            # Obtener datos
            klines = await self.connector.get_klines(pair, "1m", 500)
            if not klines:
                return
            
            # Crear features
            features = self.predictor.create_features(klines)
            if features.empty:
                return
            
            # Realizar predicción
            prediction = await self.predictor.predict(pair, features)
            if not prediction:
                return
            
            current_price = klines[-1]['close']
            
            logger.info(f"{pair}: {prediction['signal']} (conf: {prediction['confidence']:.3f})")
            
            # Verificar si se puede tradear
            if not self.risk_manager.can_trade(pair, prediction['signal'], prediction['confidence']):
                return
            
            # Obtener información de cuenta
            account_info = await self.connector.get_account_info()
            if not account_info:
                return
            
            # Calcular balance USDT
            usdt_balance = 0.0
            for balance in account_info.get('balances', []):
                if balance['asset'] == 'USDT':
                    usdt_balance = float(balance['free'])
                    break
            
            if usdt_balance < 10:  # Mínimo $10
                logger.warning(f"Balance insuficiente: ${usdt_balance}")
                return
            
            # Calcular tamaño de posición
            position_size = self.risk_manager.calculate_position_size(pair, usdt_balance, current_price)
            if position_size * current_price < 10:  # Mínimo $10 por trade
                logger.warning(f"Posición muy pequeña para {pair}")
                return
            
            # Ejecutar trade
            side = "BUY" if prediction['signal'] == 'BUY' else "SELL"
            
            order_result = await self.connector.place_order(pair, side, position_size)
            if order_result:
                self.risk_manager.register_trade(pair, side, position_size, current_price)
                
                # Log detallado
                logger.info(f"""
                🎯 TRADE EJECUTADO:
                Par: {pair}
                Señal: {prediction['signal']}
                Confianza: {prediction['confidence']:.3f}
                Lado: {side}
                Cantidad: {position_size}
                Precio: ${current_price}
                Valor: ${position_size * current_price:.2f}
                """)
            
        except Exception as e:
            logger.error(f"Error procesando {pair}: {e}")
    
    async def stop(self):
        """Detener sistema"""
        logger.info("Deteniendo sistema de trading...")
        self.running = False

async def main():
    """Función principal"""
    # Configuración
    config = TradingConfig(
        trading_pairs=["BTCUSDT", "ETHUSDT", "BNBUSDT"],
        testnet=True,  # IMPORTANTE: Usar testnet inicialmente
        api_key=os.getenv("BINANCE_API_KEY", ""),
        api_secret=os.getenv("BINANCE_API_SECRET", ""),
        min_confidence=0.80,  # Alta confianza requerida
        max_position_size=0.05,  # 5% máximo por posición
    )
    
    # Crear trader
    trader = BinanceTCNTrader(config)
    
    # Inicializar
    if await trader.initialize():
        try:
            # Ejecutar
            await trader.run_trading_loop()
        except KeyboardInterrupt:
            logger.info("Interrupción manual detectada")
        finally:
            await trader.stop()
    else:
        logger.error("Error en inicialización")

if __name__ == "__main__":
    print("🚀 BINANCE TCN INTEGRATION SYSTEM")
    print("Sistema de trading automatizado con TCN optimizado")
    print("MODO: TESTNET (cambiar config.testnet=False para producción)")
    print("="*60)
    
    # Verificar variables de entorno
    if not os.getenv("BINANCE_API_KEY"):
        print("⚠️  CONFIGURAR: export BINANCE_API_KEY='tu_api_key'")
    if not os.getenv("BINANCE_API_SECRET"):
        print("⚠️  CONFIGURAR: export BINANCE_API_SECRET='tu_api_secret'")
    
    asyncio.run(main()) 