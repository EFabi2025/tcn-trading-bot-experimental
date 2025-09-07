#!/usr/bin/env python3
"""
🎯 SISTEMA DE BACKTEST AVANZADO PARA PREDICTORES TÉCNICOS
Sistema completo para evaluar predictores de 1m, 3m y 5m con configuraciones flexibles

CARACTERÍSTICAS:
- Soporte para múltiples timeframes (1m, 3m, 5m)
- Configuración flexible de estrategias
- Métricas avanzadas de evaluación
- Visualizaciones detalladas
- Comparación entre predictores
- Optimización de parámetros
"""

import asyncio
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# === IMPORTS DE PREDICTORES ===
try:
    from predictor1m_talib import get_ensemble_ready_prediction_talib
    from predictor3m_core_optimized import get_ensemble_ready_prediction_core_3m
    from predictor5m_talib import get_ensemble_ready_prediction_5m_talib
    PREDICTORS_AVAILABLE = True
    print("✅ Predictores técnicos cargados correctamente")
except ImportError as e:
    PREDICTORS_AVAILABLE = False
    print(f"⚠️ Error cargando predictores: {e}")

# === IMPORTS DE BINANCE ===
try:
    from binance.client import Client
    from binance.exceptions import BinanceAPIException
    BINANCE_AVAILABLE = True
except ImportError:
    BINANCE_AVAILABLE = False
    print("⚠️ Binance API no disponible")

# === CONFIGURACIÓN ===
SUPPORTED_TIMEFRAMES = ['1m', '3m', '5m']
SUPPORTED_SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT', 'BNBUSDT', 'XRPUSDT', 'SOLUSDT', 'POLUSDT']

class SignalType(Enum):
    """Tipos de señales de trading"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

class RiskLevel(Enum):
    """Niveles de riesgo"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"

@dataclass
class BacktestConfig:
    """Configuración del backtest"""
    # === CONFIGURACIÓN BÁSICA ===
    symbol: str = 'BTCUSDT'
    timeframe: str = '1m'
    start_date: str = '2024-01-01'
    end_date: str = '2024-12-31'
    initial_balance: float = 10000.0
    
    # === CONFIGURACIÓN DE PREDICTORES ===
    predictors: List[str] = field(default_factory=lambda: ['1m', '3m', '5m'])
    use_ensemble: bool = True
    ensemble_weights: Dict[str, float] = field(default_factory=lambda: {'1m': 0.4, '3m': 0.4, '5m': 0.2})
    
    # === CONFIGURACIÓN DE TRADING ===
    position_size_pct: float = 0.1  # 10% del balance por trade
    stop_loss_pct: float = 0.02     # 2% stop loss
    take_profit_pct: float = 0.04   # 4% take profit
    max_positions: int = 3          # Máximo 3 posiciones simultáneas
    
    # === CONFIGURACIÓN DE FILTROS ===
    min_confidence: float = 60.0    # Confianza mínima para trade
    min_signal_strength: float = 0.6 # Fuerza mínima de señal
    risk_level_filter: List[RiskLevel] = field(default_factory=lambda: [RiskLevel.LOW, RiskLevel.MEDIUM])
    
    # === CONFIGURACIÓN DE EVALUACIÓN ===
    commission_rate: float = 0.001  # 0.1% comisión
    slippage_rate: float = 0.0005   # 0.05% slippage
    benchmark_symbol: str = 'BTCUSDT'
    
    # === CONFIGURACIÓN DE OPTIMIZACIÓN ===
    optimize_params: bool = False
    param_ranges: Dict[str, List] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validar configuración después de inicialización"""
        if self.timeframe not in SUPPORTED_TIMEFRAMES:
            raise ValueError(f"Timeframe {self.timeframe} no soportado. Usar: {SUPPORTED_TIMEFRAMES}")
        
        if self.symbol not in SUPPORTED_SYMBOLS:
            raise ValueError(f"Símbolo {self.symbol} no soportado. Usar: {SUPPORTED_SYMBOLS}")
        
        # Normalizar pesos del ensemble
        total_weight = sum(self.ensemble_weights.values())
        if total_weight > 0:
            self.ensemble_weights = {k: v/total_weight for k, v in self.ensemble_weights.items()}

@dataclass
class Trade:
    """Representa una operación de trading"""
    timestamp: datetime
    symbol: str
    timeframe: str
    signal_type: SignalType
    entry_price: float
    quantity: float
    stop_loss: float
    take_profit: float
    confidence: float
    predictor: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Campos calculados
    exit_price: Optional[float] = None
    exit_timestamp: Optional[datetime] = None
    exit_reason: Optional[str] = None
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None

@dataclass
class BacktestResults:
    """Resultados del backtest"""
    # === MÉTRICAS BÁSICAS ===
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    
    # === MÉTRICAS DE RENDIMIENTO ===
    total_return: float = 0.0
    annualized_return: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    
    # === MÉTRICAS DE RIESGO ===
    volatility: float = 0.0
    var_95: float = 0.0  # Value at Risk 95%
    cvar_95: float = 0.0  # Conditional Value at Risk 95%
    
    # === MÉTRICAS AVANZADAS ===
    profit_factor: float = 0.0
    recovery_factor: float = 0.0
    expectancy: float = 0.0
    kelly_criterion: float = 0.0
    
    # === DATOS DETALLADOS ===
    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[Tuple[datetime, float]] = field(default_factory=list)
    drawdown_curve: List[Tuple[datetime, float]] = field(default_factory=list)
    
    # === CONFIGURACIÓN USADA ===
    config: BacktestConfig = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir resultados a diccionario para serialización"""
        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.win_rate,
            'total_return': self.total_return,
            'annualized_return': self.annualized_return,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'calmar_ratio': self.calmar_ratio,
            'volatility': self.volatility,
            'var_95': self.var_95,
            'cvar_95': self.cvar_95,
            'profit_factor': self.profit_factor,
            'recovery_factor': self.recovery_factor,
            'expectancy': self.expectancy,
            'kelly_criterion': self.kelly_criterion,
            'config': self.config.__dict__ if self.config else None
        }

class HistoricalDataProvider:
    """Proveedor de datos históricos para backtesting"""
    
    def __init__(self, api_key: str = None, api_secret: str = None):
        self.api_key = api_key or os.getenv('BINANCE_API_KEY')
        self.api_secret = api_secret or os.getenv('BINANCE_API_SECRET')
        self.client = None
        
        if BINANCE_AVAILABLE and self.api_key and self.api_secret:
            try:
                self.client = Client(self.api_key, self.api_secret)
                print("✅ Cliente Binance inicializado para datos históricos")
            except Exception as e:
                print(f"⚠️ Error inicializando cliente Binance: {e}")
    
    async def get_historical_data(
        self, 
        symbol: str, 
        timeframe: str, 
        start_date: str, 
        end_date: str,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Obtener datos históricos de Binance
        
        Args:
            symbol: Símbolo del par (ej: 'BTCUSDT')
            timeframe: Timeframe ('1m', '3m', '5m')
            start_date: Fecha de inicio (YYYY-MM-DD)
            end_date: Fecha de fin (YYYY-MM-DD)
            limit: Límite de velas por request
            
        Returns:
            DataFrame con datos OHLCV
        """
        if not self.client:
            # Generar datos sintéticos para testing
            return self._generate_synthetic_data(symbol, timeframe, start_date, end_date)
        
        try:
            # Convertir fechas a timestamps
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            
            # Obtener datos en chunks
            all_data = []
            current_start = start_dt
            
            while current_start < end_dt:
                try:
                    klines = self.client.get_historical_klines(
                        symbol=symbol,
                        interval=timeframe,
                        start_str=current_start.strftime('%d %b %Y %H:%M:%S'),
                        end_str=min(current_start + timedelta(days=10), end_dt).strftime('%d %b %Y %H:%M:%S'),
                        limit=limit 
                    )
                    
                    if klines:
                        all_data.extend(klines)
                        current_start += timedelta(days=10)
                    else:
                        break
                        
                except Exception as e:
                    print(f"⚠️ Error obteniendo datos para {symbol} {timeframe}: {e}")
                    break
            
            if not all_data:
                print(f"⚠️ No se obtuvieron datos para {symbol} {timeframe}, generando datos sintéticos")
                return self._generate_synthetic_data(symbol, timeframe, start_date, end_date)
            
            # Convertir a DataFrame
            df = pd.DataFrame(all_data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Limpiar y convertir tipos
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
            
            # Filtrar por fechas
            df = df[(df['timestamp'] >= start_dt) & (df['timestamp'] <= end_dt)]
            df = df.reset_index(drop=True)
            
            print(f"✅ Datos históricos obtenidos: {len(df)} velas para {symbol} {timeframe}")
            return df
            
        except Exception as e:
            print(f"⚠️ Error obteniendo datos históricos: {e}")
            return self._generate_synthetic_data(symbol, timeframe, start_date, end_date)
    
    def _generate_synthetic_data(
        self, 
        symbol: str, 
        timeframe: str, 
        start_date: str, 
        end_date: str
    ) -> pd.DataFrame:
        """Generar datos sintéticos para testing"""
        print(f"🔄 Generando datos sintéticos para {symbol} {timeframe}")
        
        # Calcular número de velas
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        if timeframe == '1m':
            freq = '1min'
        elif timeframe == '3m':
            freq = '3min'
        elif timeframe == '5m':
            freq = '5min'
        else:
            freq = '1min'
        
        # Generar timestamps
        timestamps = pd.date_range(start=start_dt, end=end_dt, freq=freq)
        
        # Generar precios sintéticos con random walk
        np.random.seed(42)  # Para reproducibilidad
        n = len(timestamps)
        
        # Precio inicial basado en el símbolo
        initial_prices = {
            'BTCUSDT': 45000,
            'ETHUSDT': 3000,
            'ADAUSDT': 0.5,
            'DOTUSDT': 20,
            'BNBUSDT': 300,
            'XRPUSDT': 0.6,
            'SOLUSDT': 100,
            'POLUSDT': 15
        }
        
        initial_price = initial_prices.get(symbol, 100)
        
        # Random walk con tendencia
        returns = np.random.normal(0.0001, 0.02, n)  # 0.01% retorno esperado, 2% volatilidad
        prices = [initial_price]
        
        for i in range(1, n):
            new_price = prices[-1] * (1 + returns[i])
            prices.append(max(new_price, 0.01))  # Precio mínimo
        
        # Generar OHLCV
        data = []
        for i, (ts, price) in enumerate(zip(timestamps, prices)):
            # Simular variación intradiaria
            volatility = 0.01
            high = price * (1 + abs(np.random.normal(0, volatility)))
            low = price * (1 - abs(np.random.normal(0, volatility)))
            open_price = prices[i-1] if i > 0 else price
            
            # Asegurar coherencia OHLC
            high = max(high, price, open_price)
            low = min(low, price, open_price)
            
            # Volumen sintético
            volume = np.random.lognormal(10, 1)
            
            data.append({
                'timestamp': ts,
                'open': round(open_price, 8),
                'high': round(high, 8),
                'low': round(low, 8),
                'close': round(price, 8),
                'volume': round(volume, 2)
            })
        
        df = pd.DataFrame(data)
        print(f"✅ Datos sintéticos generados: {len(df)} velas")
        return df

class TechnicalPredictorBacktest:
    """Motor principal de backtesting para predictores técnicos"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.data_provider = HistoricalDataProvider()
        self.results = BacktestResults(config=config)
        
        # Mapeo de predictores
        self.predictor_functions = {
            '1m': get_ensemble_ready_prediction_talib if PREDICTORS_AVAILABLE else None,
            '3m': get_ensemble_ready_prediction_core_3m if PREDICTORS_AVAILABLE else None,
            '5m': get_ensemble_ready_prediction_5m_talib if PREDICTORS_AVAILABLE else None
        }
    
    async def run_backtest(self) -> BacktestResults:
        """Ejecutar backtest completo"""
        print(f"🚀 Iniciando backtest para {self.config.symbol} {self.config.timeframe}")
        print(f"📅 Período: {self.config.start_date} - {self.config.end_date}")
        print(f"💰 Balance inicial: ${self.config.initial_balance:,.2f}")
        
        # Obtener datos históricos
        historical_data = await self.data_provider.get_historical_data(
            symbol=self.config.symbol,
            timeframe=self.config.timeframe,
            start_date=self.config.start_date,
            end_date=self.config.end_date
        )
        
        if historical_data.empty:
            print("❌ No se pudieron obtener datos históricos")
            return self.results
        
        # Simular trading
        await self._simulate_trading(historical_data)
        
        # Calcular métricas
        self._calculate_metrics()
        
        print(f"✅ Backtest completado: {self.results.total_trades} trades")
        print(f"📊 Win Rate: {self.results.win_rate:.2f}%")
        print(f"💰 Retorno Total: {self.results.total_return:.2f}%")
        print(f"📉 Max Drawdown: {self.results.max_drawdown:.2f}%")
        
        return self.results
    
    async def _simulate_trading(self, data: pd.DataFrame):
        """Simular estrategia de trading"""
        balance = self.config.initial_balance
        positions = []
        equity_curve = []
        
        print(f"🔄 Simulando {len(data)} velas...")
        
        for i, row in data.iterrows():
            current_time = row['timestamp']
            current_price = row['close']
            
            # Actualizar equity curve
            current_equity = balance + sum(pos['quantity'] * current_price for pos in positions)
            equity_curve.append((current_time, current_equity))
            
            # Verificar stops y targets de posiciones existentes
            positions = await self._check_exits(positions, current_price, current_time)
            
            # Generar señales si no hay muchas posiciones abiertas
            if len(positions) < self.config.max_positions:
                signal = await self._generate_signal(current_time, current_price)
                
                if signal and signal['signal_type'] != SignalType.HOLD:
                    # Calcular tamaño de posición
                    position_size = balance * self.config.position_size_pct
                    quantity = position_size / current_price
                    
                    # Crear nueva posición
                    trade = Trade(
                        timestamp=current_time,
                        symbol=self.config.symbol,
                        timeframe=self.config.timeframe,
                        signal_type=signal['signal_type'],
                        entry_price=current_price,
                        quantity=quantity,
                        stop_loss=current_price * (1 - self.config.stop_loss_pct) if signal['signal_type'] == SignalType.BUY else current_price * (1 + self.config.stop_loss_pct),
                        take_profit=current_price * (1 + self.config.take_profit_pct) if signal['signal_type'] == SignalType.BUY else current_price * (1 - self.config.take_profit_pct),
                        confidence=signal['confidence'],
                        predictor=signal['predictor'],
                        metadata=signal.get('metadata', {})
                    )
                    
                    positions.append(trade.__dict__)
                    self.results.trades.append(trade)
            
            # Actualizar balance (simplificado)
            if i % 1000 == 0:
                print(f"📈 Procesadas {i}/{len(data)} velas - Balance: ${balance:,.2f}")
        
        # Cerrar posiciones restantes
        for pos in positions:
            if isinstance(pos, dict):
                trade = Trade(**pos)
                trade.exit_price = data.iloc[-1]['close']
                trade.exit_timestamp = data.iloc[-1]['timestamp']
                trade.exit_reason = "End of backtest"
                trade.pnl = (trade.exit_price - trade.entry_price) * trade.quantity
                trade.pnl_pct = (trade.exit_price - trade.entry_price) / trade.entry_price * 100
                balance += trade.pnl
        
        self.results.equity_curve = equity_curve
    
    async def _generate_signal(self, timestamp: datetime, price: float) -> Optional[Dict]:
        """Generar señal de trading usando predictores"""
        if not PREDICTORS_AVAILABLE:
            # Generar señal aleatoria para testing
            import random
            signals = [SignalType.BUY, SignalType.SELL, SignalType.HOLD]
            weights = [0.3, 0.3, 0.4]  # 30% BUY, 30% SELL, 40% HOLD
            signal_type = random.choices(signals, weights=weights)[0]
            
            return {
                'signal_type': signal_type,
                'confidence': random.uniform(50, 90),
                'predictor': 'random',
                'metadata': {}
            }
        
        # Usar predictores reales
        predictions = {}
        
        for predictor_name in self.config.predictors:
            if predictor_name in self.predictor_functions and self.predictor_functions[predictor_name]:
                try:
                    pred = self.predictor_functions[predictor_name](self.config.symbol)
                    if pred:
                        predictions[predictor_name] = pred
                except Exception as e:
                    print(f"⚠️ Error en predictor {predictor_name}: {e}")
        
        if not predictions:
            return None
        
        # Combinar predicciones si hay múltiples predictores
        if len(predictions) > 1 and self.config.use_ensemble:
            return self._combine_predictions(predictions)
        else:
            # Usar predictor individual
            pred_name = list(predictions.keys())[0]
            pred = predictions[pred_name]
            return self._parse_prediction(pred, pred_name)
    
    def _combine_predictions(self, predictions: Dict[str, Dict]) -> Dict:
        """Combinar predicciones de múltiples predictores"""
        total_buy_prob = 0
        total_sell_prob = 0
        total_confidence = 0
        total_weight = 0
        
        for pred_name, pred in predictions.items():
            weight = self.config.ensemble_weights.get(pred_name, 1.0)
            
            buy_prob = pred.get('buy_probability', 0) / 100
            sell_prob = pred.get('sell_probability', 0) / 100
            confidence = pred.get('confidence', 0)
            
            total_buy_prob += buy_prob * weight
            total_sell_prob += sell_prob * weight
            total_confidence += confidence * weight
            total_weight += weight
        
        # Normalizar
        if total_weight > 0:
            total_buy_prob /= total_weight
            total_sell_prob /= total_weight
            total_confidence /= total_weight
        
        # Determinar señal
        if total_buy_prob > total_sell_prob and total_buy_prob > 0.6:
            signal_type = SignalType.BUY
        elif total_sell_prob > total_buy_prob and total_sell_prob > 0.6:
            signal_type = SignalType.SELL
        else:
            signal_type = SignalType.HOLD
        
        # Aplicar filtros
        if total_confidence < self.config.min_confidence:
            signal_type = SignalType.HOLD
        
        return {
            'signal_type': signal_type,
            'confidence': total_confidence,
            'predictor': 'ensemble',
            'metadata': {
                'individual_predictions': predictions,
                'buy_probability': total_buy_prob * 100,
                'sell_probability': total_sell_prob * 100
            }
        }
    
    def _parse_prediction(self, prediction: Dict, predictor_name: str) -> Dict:
        """Parsear predicción individual"""
        buy_prob = prediction.get('buy_probability', 0) / 100
        sell_prob = prediction.get('sell_probability', 0) / 100
        confidence = prediction.get('confidence', 0)
        
        # Determinar señal
        if buy_prob > sell_prob and buy_prob > 0.6:
            signal_type = SignalType.BUY
        elif sell_prob > buy_prob and sell_prob > 0.6:
            signal_type = SignalType.SELL
        else:
            signal_type = SignalType.HOLD
        
        # Aplicar filtros
        if confidence < self.config.min_confidence:
            signal_type = SignalType.HOLD
        
        return {
            'signal_type': signal_type,
            'confidence': confidence,
            'predictor': predictor_name,
            'metadata': prediction
        }
    
    async def _check_exits(self, positions: List[Dict], current_price: float, current_time: datetime) -> List[Dict]:
        """Verificar stops y targets de posiciones existentes"""
        remaining_positions = []
        
        for pos in positions:
            if isinstance(pos, dict):
                trade = Trade(**pos)
                
                # Verificar stop loss
                if trade.signal_type == SignalType.BUY and current_price <= trade.stop_loss:
                    trade.exit_price = trade.stop_loss
                    trade.exit_timestamp = current_time
                    trade.exit_reason = "Stop Loss"
                    trade.pnl = (trade.exit_price - trade.entry_price) * trade.quantity
                    trade.pnl_pct = (trade.exit_price - trade.entry_price) / trade.entry_price * 100
                    
                elif trade.signal_type == SignalType.SELL and current_price >= trade.stop_loss:
                    trade.exit_price = trade.stop_loss
                    trade.exit_timestamp = current_time
                    trade.exit_reason = "Stop Loss"
                    trade.pnl = (trade.entry_price - trade.exit_price) * trade.quantity
                    trade.pnl_pct = (trade.entry_price - trade.exit_price) / trade.entry_price * 100
                
                # Verificar take profit
                elif trade.signal_type == SignalType.BUY and current_price >= trade.take_profit:
                    trade.exit_price = trade.take_profit
                    trade.exit_timestamp = current_time
                    trade.exit_reason = "Take Profit"
                    trade.pnl = (trade.exit_price - trade.entry_price) * trade.quantity
                    trade.pnl_pct = (trade.exit_price - trade.entry_price) / trade.entry_price * 100
                    
                elif trade.signal_type == SignalType.SELL and current_price <= trade.take_profit:
                    trade.exit_price = trade.take_profit
                    trade.exit_timestamp = current_time
                    trade.exit_reason = "Take Profit"
                    trade.pnl = (trade.entry_price - trade.exit_price) * trade.quantity
                    trade.pnl_pct = (trade.entry_price - trade.exit_price) / trade.entry_price * 100
                
                else:
                    # Mantener posición abierta
                    remaining_positions.append(trade.__dict__)
                    continue
                
                # Actualizar trade en resultados
                for i, result_trade in enumerate(self.results.trades):
                    if (result_trade.timestamp == trade.timestamp and 
                        result_trade.entry_price == trade.entry_price):
                        self.results.trades[i] = trade
                        break
        
        return remaining_positions
    
    def _calculate_metrics(self):
        """Calcular métricas de rendimiento"""
        if not self.results.trades:
            return
        
        # Métricas básicas
        self.results.total_trades = len(self.results.trades)
        self.results.winning_trades = len([t for t in self.results.trades if t.pnl and t.pnl > 0])
        self.results.losing_trades = len([t for t in self.results.trades if t.pnl and t.pnl < 0])
        self.results.win_rate = (self.results.winning_trades / self.results.total_trades * 100) if self.results.total_trades > 0 else 0
        
        # Métricas de rendimiento
        total_pnl = sum(t.pnl for t in self.results.trades if t.pnl)
        self.results.total_return = (total_pnl / self.config.initial_balance) * 100
        
        # Calcular equity curve
        if self.results.equity_curve:
            equity_values = [eq[1] for eq in self.results.equity_curve]
            
            # Retorno anualizado
            days = (self.results.equity_curve[-1][0] - self.results.equity_curve[0][0]).days
            if days > 0:
                self.results.annualized_return = ((equity_values[-1] / equity_values[0]) ** (365 / days) - 1) * 100
            
            # Drawdown
            peak = equity_values[0]
            max_dd = 0
            drawdowns = []
            
            for eq in equity_values:
                if eq > peak:
                    peak = eq
                dd = (peak - eq) / peak * 100
                drawdowns.append(dd)
                max_dd = max(max_dd, dd)
            
            self.results.max_drawdown = max_dd
            self.results.drawdown_curve = [(self.results.equity_curve[i][0], dd) for i, dd in enumerate(drawdowns)]
            
            # Volatilidad
            if len(equity_values) > 1:
                returns = [(equity_values[i] - equity_values[i-1]) / equity_values[i-1] for i in range(1, len(equity_values))]
                self.results.volatility = np.std(returns) * np.sqrt(252) * 100  # Anualizada
            
            # Sharpe Ratio
            if self.results.volatility > 0:
                risk_free_rate = 0.02  # 2% tasa libre de riesgo
                excess_return = self.results.annualized_return - risk_free_rate
                self.results.sharpe_ratio = excess_return / (self.results.volatility / 100)
        
        # Métricas adicionales
        if self.results.total_trades > 0:
            pnls = [t.pnl for t in self.results.trades if t.pnl]
            if pnls:
                self.results.expectancy = np.mean(pnls)
                
                # Profit Factor
                gross_profit = sum(p for p in pnls if p > 0)
                gross_loss = abs(sum(p for p in pnls if p < 0))
                self.results.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
                
                # Kelly Criterion
                win_rate = self.results.win_rate / 100
                avg_win = np.mean([p for p in pnls if p > 0]) if any(p > 0 for p in pnls) else 0
                avg_loss = abs(np.mean([p for p in pnls if p < 0])) if any(p < 0 for p in pnls) else 1
                self.results.kelly_criterion = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win if avg_win > 0 else 0

class BacktestVisualizer:
    """Visualizador de resultados de backtest"""
    
    def __init__(self, results: BacktestResults):
        self.results = results
    
    def plot_equity_curve(self, save_path: str = None):
        """Plotear curva de equity"""
        if not self.results.equity_curve:
            print("❌ No hay datos de equity curve para plotear")
            return
        
        plt.figure(figsize=(12, 8))
        
        timestamps = [eq[0] for eq in self.results.equity_curve]
        equity_values = [eq[1] for eq in self.results.equity_curve]
        
        plt.subplot(2, 1, 1)
        plt.plot(timestamps, equity_values, linewidth=2, color='blue')
        plt.title(f'Curva de Equity - {self.results.config.symbol} {self.results.config.timeframe}')
        plt.ylabel('Equity ($)')
        plt.grid(True, alpha=0.3)
        
        # Drawdown
        plt.subplot(2, 1, 2)
        if self.results.drawdown_curve:
            dd_timestamps = [dd[0] for dd in self.results.drawdown_curve]
            dd_values = [dd[1] for dd in self.results.drawdown_curve]
            plt.fill_between(dd_timestamps, dd_values, 0, alpha=0.3, color='red')
            plt.plot(dd_timestamps, dd_values, color='red', linewidth=1)
        
        plt.title('Drawdown')
        plt.ylabel('Drawdown (%)')
        plt.xlabel('Fecha')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Gráfico guardado en: {save_path}")
        
        plt.show()
    
    def plot_trade_distribution(self, save_path: str = None):
        """Plotear distribución de trades"""
        if not self.results.trades:
            print("❌ No hay trades para plotear")
            return
        
        pnls = [t.pnl for t in self.results.trades if t.pnl]
        if not pnls:
            print("❌ No hay PnL calculado para plotear")
            return
        
        plt.figure(figsize=(15, 10))
        
        # Distribución de PnL
        plt.subplot(2, 2, 1)
        plt.hist(pnls, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('Distribución de PnL por Trade')
        plt.xlabel('PnL ($)')
        plt.ylabel('Frecuencia')
        plt.axvline(0, color='red', linestyle='--', alpha=0.7)
        plt.grid(True, alpha=0.3)
        
        # PnL acumulado
        plt.subplot(2, 2, 2)
        cumulative_pnl = np.cumsum(pnls)
        plt.plot(cumulative_pnl, linewidth=2, color='green')
        plt.title('PnL Acumulado')
        plt.xlabel('Número de Trade')
        plt.ylabel('PnL Acumulado ($)')
        plt.grid(True, alpha=0.3)
        
        # PnL por mes
        plt.subplot(2, 2, 3)
        monthly_pnl = {}
        for trade in self.results.trades:
            if trade.pnl and trade.exit_timestamp:
                month_key = trade.exit_timestamp.strftime('%Y-%m')
                if month_key not in monthly_pnl:
                    monthly_pnl[month_key] = 0
                monthly_pnl[month_key] += trade.pnl
        
        if monthly_pnl:
            months = list(monthly_pnl.keys())
            pnls_monthly = list(monthly_pnl.values())
            plt.bar(months, pnls_monthly, alpha=0.7, color='orange')
            plt.title('PnL por Mes')
            plt.xlabel('Mes')
            plt.ylabel('PnL ($)')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
        
        # Métricas de rendimiento
        plt.subplot(2, 2, 4)
        metrics = [
            f"Total Trades: {self.results.total_trades}",
            f"Win Rate: {self.results.win_rate:.2f}%",
            f"Total Return: {self.results.total_return:.2f}%",
            f"Max Drawdown: {self.results.max_drawdown:.2f}%",
            f"Sharpe Ratio: {self.results.sharpe_ratio:.2f}",
            f"Profit Factor: {self.results.profit_factor:.2f}"
        ]
        
        plt.text(0.1, 0.9, '\n'.join(metrics), transform=plt.gca().transAxes, 
                fontsize=12, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        plt.axis('off')
        plt.title('Métricas de Rendimiento')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Gráfico guardado en: {save_path}")
        
        plt.show()
    
    def generate_report(self, save_path: str = None) -> str:
        """Generar reporte detallado"""
        report = f"""
# 📊 REPORTE DE BACKTEST - {self.results.config.symbol} {self.results.config.timeframe}

## 📈 Resumen Ejecutivo
- **Período**: {self.results.config.start_date} - {self.results.config.end_date}
- **Balance Inicial**: ${self.results.config.initial_balance:,.2f}
- **Predictores**: {', '.join(self.results.config.predictors)}
- **Modo Ensemble**: {'Sí' if self.results.config.use_ensemble else 'No'}

## 🎯 Métricas de Trading
- **Total de Trades**: {self.results.total_trades}
- **Trades Ganadores**: {self.results.winning_trades}
- **Trades Perdedores**: {self.results.losing_trades}
- **Win Rate**: {self.results.win_rate:.2f}%

## 💰 Rendimiento
- **Retorno Total**: {self.results.total_return:.2f}%
- **Retorno Anualizado**: {self.results.annualized_return:.2f}%
- **Volatilidad**: {self.results.volatility:.2f}%

## ⚠️ Riesgo
- **Max Drawdown**: {self.results.max_drawdown:.2f}%
- **Sharpe Ratio**: {self.results.sharpe_ratio:.2f}
- **Sortino Ratio**: {self.results.sortino_ratio:.2f}
- **Calmar Ratio**: {self.results.calmar_ratio:.2f}

## 📊 Métricas Avanzadas
- **Profit Factor**: {self.results.profit_factor:.2f}
- **Expectancy**: ${self.results.expectancy:.2f}
- **Kelly Criterion**: {self.results.kelly_criterion:.4f}

## ⚙️ Configuración
- **Tamaño de Posición**: {self.results.config.position_size_pct*100:.1f}%
- **Stop Loss**: {self.results.config.stop_loss_pct*100:.1f}%
- **Take Profit**: {self.results.config.take_profit_pct*100:.1f}%
- **Confianza Mínima**: {self.results.config.min_confidence:.1f}%
- **Max Posiciones**: {self.results.config.max_positions}

## 📅 Análisis Temporal
"""
        
        if self.results.trades:
            # Análisis por mes
            monthly_stats = {}
            for trade in self.results.trades:
                if trade.exit_timestamp:
                    month = trade.exit_timestamp.strftime('%Y-%m')
                    if month not in monthly_stats:
                        monthly_stats[month] = {'trades': 0, 'pnl': 0, 'wins': 0}
                    monthly_stats[month]['trades'] += 1
                    if trade.pnl:
                        monthly_stats[month]['pnl'] += trade.pnl
                        if trade.pnl > 0:
                            monthly_stats[month]['wins'] += 1
            
            report += "\n### 📊 Rendimiento Mensual\n"
            report += "| Mes | Trades | PnL | Win Rate |\n"
            report += "|-----|--------|-----|----------|\n"
            
            for month in sorted(monthly_stats.keys()):
                stats = monthly_stats[month]
                win_rate = (stats['wins'] / stats['trades'] * 100) if stats['trades'] > 0 else 0
                report += f"| {month} | {stats['trades']} | ${stats['pnl']:.2f} | {win_rate:.1f}% |\n"
        
        report += f"\n---\n*Reporte generado el {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"📄 Reporte guardado en: {save_path}")
        
        return report

# === FUNCIONES DE UTILIDAD ===

async def run_single_backtest(config: BacktestConfig) -> BacktestResults:
    """Ejecutar un backtest individual"""
    backtest = TechnicalPredictorBacktest(config)
    return await backtest.run_backtest()

async def run_multiple_backtests(configs: List[BacktestConfig]) -> Dict[str, BacktestResults]:
    """Ejecutar múltiples backtests para comparación"""
    results = {}
    
    for i, config in enumerate(configs):
        print(f"\n🔄 Ejecutando backtest {i+1}/{len(configs)}: {config.symbol} {config.timeframe}")
        results[f"{config.symbol}_{config.timeframe}_{i}"] = await run_single_backtest(config)
    
    return results

def compare_backtests(results: Dict[str, BacktestResults]) -> pd.DataFrame:
    """Comparar resultados de múltiples backtests"""
    comparison_data = []
    
    for name, result in results.items():
        comparison_data.append({
            'Name': name,
            'Total Trades': result.total_trades,
            'Win Rate (%)': result.win_rate,
            'Total Return (%)': result.total_return,
            'Max Drawdown (%)': result.max_drawdown,
            'Sharpe Ratio': result.sharpe_ratio,
            'Profit Factor': result.profit_factor,
            'Volatility (%)': result.volatility
        })
    
    return pd.DataFrame(comparison_data)

# === EJEMPLO DE USO ===
if __name__ == "__main__":
    # Configuración de ejemplo
    config = BacktestConfig(
        symbol='BTCUSDT',
        timeframe='1m',
        start_date='2024-01-01',
        end_date='2024-01-31',
        initial_balance=10000.0,
        predictors=['1m', '3m', '5m'],
        use_ensemble=True,
        position_size_pct=0.1,
        stop_loss_pct=0.02,
        take_profit_pct=0.04,
        min_confidence=60.0
    )
    
    # Ejecutar backtest
    async def main():
        results = await run_single_backtest(config)
        
        # Visualizar resultados
        visualizer = BacktestVisualizer(results)
        visualizer.plot_equity_curve()
        visualizer.plot_trade_distribution()
        
        # Generar reporte
        report = visualizer.generate_report('backtest_report.md')
        print(report)
    
    # Ejecutar
    asyncio.run(main())
