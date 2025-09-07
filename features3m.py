#!/usr/bin/env python3
"""
ENHANCED FEATURES ENGINEERING FOR TCN 3M MODEL - INTEGRACIÓN COMPLETA
Conjunto completo de features optimizado para crypto trading en timeframe 3m
Integra indicadores técnicos core del predictor3m_core_optimized.py con features avanzadas

ARQUITECTURA:
1. Indicadores Técnicos Core (12): Extraídos del predictor core optimizado
2. Features Avanzadas (33): Microestructura, temporal, momentum, divergencias, etc.
3. Features de Integración (5): Combinaciones inteligentes de indicadores core
4. Total: ~50 features optimizadas para modelos TCN

COMPATIBILIDAD:
- Modelos TCN con secuencias temporales
- Ensemble híbrido con predictor 1m/3m/5m
- Sistema de gestión de riesgo avanzado
"""

import numpy as np
import pandas as pd
from datetime import datetime
import talib
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Importar el predictor core para obtener indicadores técnicos reales
try:
    from predictor3m_core_optimized import (
        CoreTechnicalAnalyzer3m, 
        CoreTechnicalIndicators3m,
        SUPPORTED_PAIRS,
        TIMEFRAME
    )
    CORE_PREDICTOR_AVAILABLE = True
    print("✅ Predictor 3M Core disponible para integración")
except ImportError as e:
    CORE_PREDICTOR_AVAILABLE = False
    print(f"⚠️ Predictor 3M Core no disponible: {e}")

# Carga de configuración desde .env
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Archivo .env cargado correctamente en features3m")
except ImportError:
    print("⚠️ python-dotenv no disponible, usando variables de entorno del sistema")


class TechnicalIndicatorsBridge3m:
    """
    Clase puente para extraer indicadores técnicos del predictor core
    y convertirlos en formato compatible con el motor de features avanzadas
    """
    
    @staticmethod
    def extract_core_indicators_to_dataframe(symbol: str, periods: int = 1000) -> Optional[pd.DataFrame]:
        """
        Extraer indicadores técnicos core y convertirlos a DataFrame
        
        Args:
            symbol: Símbolo del par crypto (ej: 'BTCUSDT')
            periods: Número de períodos a obtener
            
        Returns:
            DataFrame con OHLCV + 12 indicadores técnicos core
        """
        if not CORE_PREDICTOR_AVAILABLE:
            print("❌ Predictor core no disponible - usando datos simulados")
            return None
            
        if symbol not in SUPPORTED_PAIRS:
            print(f"❌ Símbolo {symbol} no soportado. Disponibles: {SUPPORTED_PAIRS}")
            return None
        
        try:
            # Obtener datos OHLCV del predictor core
            client = CoreTechnicalAnalyzer3m.get_binance_client()
            klines = client.get_klines(symbol=symbol, interval=TIMEFRAME, limit=periods)
            
            if len(klines) < 60:
                print(f"❌ Insuficientes datos para {symbol}")
                return None
            
            # Crear DataFrame base con OHLCV
            timestamps = pd.to_datetime([int(k[0]) for k in klines], unit='ms')
            df = pd.DataFrame({
                'open': [float(k[1]) for k in klines],
                'high': [float(k[2]) for k in klines],
                'low': [float(k[3]) for k in klines],
                'close': [float(k[4]) for k in klines],
                'volume': [float(k[5]) for k in klines]
            }, index=timestamps)
            
            # Limpiar datos
            df = df.sort_index()
            df = df[~df.index.duplicated(keep='last')]
            
            # Calcular indicadores técnicos para cada período
            print(f"🔄 Calculando indicadores técnicos para {len(df)} períodos...")
            
            # Inicializar columnas de indicadores
            indicator_columns = [
                'rsi_9', 'macd', 'macd_signal', 'macd_histogram',
                'ema_8', 'ema_17', 'bollinger_upper', 'bollinger_middle', 'bollinger_lower',
                'bollinger_position', 'vwap', 'vwap_distance', 'atr', 'atr_percent',
                'volume_delta', 'obv', 'obv_sma', 'volume_ratio',
                'price_momentum', 'stoch_k', 'stoch_d', 'heikin_ashi_signal'
            ]
            
            # Inicializar con NaN
            for col in indicator_columns:
                df[col] = np.nan
            
            # Calcular indicadores usando TA-Lib para eficiencia
            if hasattr(talib, 'RSI'):
                # RSI-9
                df['rsi_9'] = talib.RSI(df['close'].values, timeperiod=9)
                
                # MACD (8,17,6)
                macd, macd_signal, macd_hist = talib.MACD(
                    df['close'].values, fastperiod=8, slowperiod=17, signalperiod=6
                )
                df['macd'] = macd
                df['macd_signal'] = macd_signal
                df['macd_histogram'] = macd_hist
                
                # EMAs
                df['ema_8'] = talib.EMA(df['close'].values, timeperiod=8)
                df['ema_17'] = talib.EMA(df['close'].values, timeperiod=17)
                
                # Bollinger Bands
                bb_upper, bb_middle, bb_lower = talib.BBANDS(
                    df['close'].values, timeperiod=15, nbdevup=2, nbdevdn=2
                )
                df['bollinger_upper'] = bb_upper
                df['bollinger_middle'] = bb_middle
                df['bollinger_lower'] = bb_lower
                
                # Bollinger Position
                df['bollinger_position'] = np.where(
                    (df['bollinger_upper'] - df['bollinger_lower']) > 0,
                    (df['close'] - df['bollinger_lower']) / (df['bollinger_upper'] - df['bollinger_lower']),
                    0.5
                )
                
                # ATR
                df['atr'] = talib.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=10)
                df['atr_percent'] = (df['atr'] / df['close']) * 100
                
                # OBV
                df['obv'] = talib.OBV(df['close'].values, df['volume'].values)
                df['obv_sma'] = talib.SMA(df['obv'].values, timeperiod=6)
                
                # Stochastic
                stoch_k, stoch_d = talib.STOCH(
                    df['high'].values, df['low'].values, df['close'].values,
                    fastk_period=9, slowk_period=3, slowd_period=3
                )
                df['stoch_k'] = stoch_k
                df['stoch_d'] = stoch_d
                
                print("✅ Indicadores TA-Lib calculados")
            else:
                print("⚠️ TA-Lib no disponible - usando fallbacks básicos")
                # Fallbacks básicos
                df['rsi_9'] = 50.0
                df['macd'] = df['macd_signal'] = df['macd_histogram'] = 0.0
                df['ema_8'] = df['ema_17'] = df['close']
                df['bollinger_upper'] = df['close'] * 1.02
                df['bollinger_middle'] = df['close']
                df['bollinger_lower'] = df['close'] * 0.98
                df['bollinger_position'] = 0.5
                df['atr'] = df['atr_percent'] = 0.0
                df['obv'] = df['obv_sma'] = 0.0
                df['stoch_k'] = df['stoch_d'] = 50.0
            
            # Calcular indicadores personalizados
            TechnicalIndicatorsBridge3m._calculate_custom_indicators(df)
            
            # Limpiar datos faltantes
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            print(f"✅ DataFrame con indicadores creado: {df.shape}")
            return df
            
        except Exception as e:
            print(f"❌ Error extrayendo indicadores para {symbol}: {e}")
            return None
    
    @staticmethod
    def _calculate_custom_indicators(df: pd.DataFrame):
        """Calcular indicadores personalizados que requieren lógica especial"""
        
        # VWAP con reseteo diario
        try:
            # Usar el cálculo de VWAP del predictor core
            vwap_series = CoreTechnicalAnalyzer3m.calculate_session_vwap(df)
            if len(vwap_series) == len(df):
                df['vwap'] = vwap_series
            else:
                # Fallback: VWAP simple
                typical_price = (df['high'] + df['low'] + df['close']) / 3
                df['vwap'] = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
            
            # VWAP distance
            df['vwap_distance'] = ((df['close'] - df['vwap']) / df['vwap'] * 100).fillna(0)
            
        except Exception as e:
            print(f"⚠️ Error calculando VWAP: {e}")
            df['vwap'] = df['close']
            df['vwap_distance'] = 0.0
        
        # Volume Delta usando la lógica del predictor core
        try:
            # Aplicar la lógica avanzada de volume delta
            volume_deltas = []
            for i in range(len(df)):
                if i < 20:
                    volume_deltas.append(0.0)
                    continue
                
                # Obtener ventana de datos
                start_idx = max(0, i - 19)
                window_highs = df['high'].iloc[start_idx:i+1].values
                window_lows = df['low'].iloc[start_idx:i+1].values
                window_closes = df['close'].iloc[start_idx:i+1].values
                window_volumes = df['volume'].iloc[start_idx:i+1].values
                
                # Calcular volume delta para esta ventana
                vd = CoreTechnicalAnalyzer3m.calculate_volume_delta_core(
                    window_highs, window_lows, window_closes, window_volumes
                )
                volume_deltas.append(vd)
            
            df['volume_delta'] = volume_deltas
            
        except Exception as e:
            print(f"⚠️ Error calculando Volume Delta: {e}")
            # Fallback simple
            df['volume_delta'] = ((df['close'] - df['open']) / df['close']).fillna(0).clip(-1, 1)
        
        # Volume Ratio
        try:
            volume_sma = df['volume'].rolling(15).mean()
            df['volume_ratio'] = (df['volume'] / volume_sma).fillna(1.0)
        except:
            df['volume_ratio'] = 1.0
        
        # Price Momentum
        try:
            df['price_momentum'] = df['close'].pct_change(3) * 100
        except:
            df['price_momentum'] = 0.0
        
        # Heikin Ashi Signal (simplificado)
        try:
            # Simplificar para evitar dependencia completa del predictor core
            ha_close = (df['open'] + df['high'] + df['low'] + df['close']) / 4
            ha_open = (df['open'].shift(1) + df['close'].shift(1)) / 2
            ha_open.iloc[0] = df['open'].iloc[0]
            
            # Señal basada en últimas 2 velas
            bullish_condition = (ha_close > ha_open) & (ha_close.shift(1) > ha_open.shift(1))
            bearish_condition = (ha_close < ha_open) & (ha_close.shift(1) < ha_open.shift(1))
            
            df['heikin_ashi_signal'] = np.where(
                bullish_condition, 'BULLISH',
                np.where(bearish_condition, 'BEARISH', 'NEUTRAL')
            )
            
        except Exception as e:
            print(f"⚠️ Error calculando Heikin Ashi: {e}")
            df['heikin_ashi_signal'] = 'NEUTRAL'
    
    @staticmethod
    def validate_indicators_dataframe(df: pd.DataFrame) -> Dict[str, any]:
        """Validar calidad del DataFrame con indicadores"""
        
        required_columns = [
            'open', 'high', 'low', 'close', 'volume',  # OHLCV
            'rsi_9', 'macd', 'macd_signal', 'macd_histogram',  # Momentum
            'ema_8', 'ema_17', 'bollinger_position', 'vwap', 'vwap_distance',  # Trend
            'atr_percent', 'volume_delta', 'volume_ratio', 'price_momentum',  # Volatility/Volume
            'stoch_k', 'heikin_ashi_signal'  # Additional
        ]
        
        validation_result = {
            'total_columns': len(df.columns),
            'required_columns': len(required_columns),
            'missing_columns': [],
            'data_completeness': {},
            'quality_score': 0.0
        }
        
        # Verificar columnas faltantes
        for col in required_columns:
            if col not in df.columns:
                validation_result['missing_columns'].append(col)
        
        # Verificar completitud de datos
        for col in df.columns:
            non_null_pct = (df[col].count() / len(df)) * 100
            validation_result['data_completeness'][col] = round(non_null_pct, 2)
        
        # Calcular score de calidad
        missing_penalty = len(validation_result['missing_columns']) * 5
        avg_completeness = np.mean(list(validation_result['data_completeness'].values()))
        validation_result['quality_score'] = max(0, avg_completeness - missing_penalty)
        
        validation_result['is_valid'] = (
            len(validation_result['missing_columns']) == 0 and
            validation_result['quality_score'] > 80
        )
        
        return validation_result


class AdvancedFeaturesEngine3m:
    """
    Motor de features avanzadas para modelo TCN de 3 minutos
    Enfoque específico en características únicas del mercado crypto
    """
    
    @staticmethod
    def create_complete_feature_set(df: pd.DataFrame = None, symbol: str = None) -> pd.DataFrame:
        """
        Crear conjunto completo de features para TCN 3m
        
        Args:
            df: DataFrame con OHLCV + indicadores técnicos (opcional)
            symbol: Símbolo para extraer datos si df no se proporciona
            
        Returns:
            DataFrame con ~50 features optimizadas (OHLCV + 12 indicadores + 33 features avanzadas + 5 integración)
        """
        
        # Obtener datos con indicadores técnicos
        if df is None:
            if symbol is None:
                raise ValueError("Debe proporcionar df o symbol")
            
            print(f"🔄 Extrayendo indicadores técnicos para {symbol}...")
            df = TechnicalIndicatorsBridge3m.extract_core_indicators_to_dataframe(symbol)
            
            if df is None:
                raise ValueError(f"No se pudieron obtener datos para {symbol}")
        
        df = df.copy()
        print(f"📊 Iniciando creación de features con DataFrame de {df.shape}")
        
        # === GRUPO 1: MICROESTRUCTURA DE PRECIOS (6 features) ===
        df = AdvancedFeaturesEngine3m._add_microstructure_features(df)
        
        # === GRUPO 2: CONTEXTO TEMPORAL CRYPTO (4 features) ===
        df = AdvancedFeaturesEngine3m._add_temporal_context(df)
        
        # === GRUPO 3: MOMENTUM MULTI-TIMEFRAME (5 features) ===
        df = AdvancedFeaturesEngine3m._add_momentum_features(df)
        
        # === GRUPO 4: DIVERGENCIAS Y CORRELACIONES (4 features) ===
        df = AdvancedFeaturesEngine3m._add_divergence_features(df)
        
        # === GRUPO 5: RÉGIMEN DE MERCADO (3 features) ===
        df = AdvancedFeaturesEngine3m._add_market_regime_features(df)
        
        # === GRUPO 6: LIQUIDEZ Y FLUJO DE ÓRDENES (4 features) ===
        df = AdvancedFeaturesEngine3m._add_liquidity_features(df)
        
        # === GRUPO 7: FEATURES ESPECÍFICAS CRYPTO (4 features) ===
        df = AdvancedFeaturesEngine3m._add_crypto_specific_features(df)
        
        # === GRUPO 8: FEATURES DE CLUSTERING TEMPORAL (3 features) ===
        df = AdvancedFeaturesEngine3m._add_temporal_clustering_features(df)
        
        # === GRUPO 9: FEATURES DE INTEGRACIÓN (5 features) ===
        df = AdvancedFeaturesEngine3m._add_integration_features(df)
        
        print(f"✅ Features completas creadas: {df.shape}")
        return df

    @staticmethod
    def create_model_compatible_feature_set(df: pd.DataFrame = None, symbol: str = None, feature_columns: List[str] = None) -> pd.DataFrame:
        """
        🎯 NUEVA FUNCIÓN: Crear features compatibles con el modelo entrenado
        
        Args:
            df: DataFrame con OHLCV + indicadores técnicos (opcional)
            symbol: Símbolo para extraer datos si df no se proporciona
            feature_columns: Lista exacta de features que el modelo espera
            
        Returns:
            DataFrame con exactamente las features que el modelo necesita
        """
        
        # Obtener datos con indicadores técnicos
        if df is None:
            if symbol is None:
                raise ValueError("Debe proporcionar df o symbol")
            
            print(f"🔄 Extrayendo indicadores técnicos para {symbol}...")
            df = TechnicalIndicatorsBridge3m.extract_core_indicators_to_dataframe(symbol)
            
            if df is None:
                raise ValueError(f"No se pudieron obtener datos para {symbol}")
        
        df = df.copy()
        print(f"📊 Iniciando creación de features compatibles con DataFrame de {df.shape}")
        
        # === GRUPO 1: MICROESTRUCTURA DE PRECIOS (6 features) ===
        df = AdvancedFeaturesEngine3m._add_microstructure_features(df)
        
        # === GRUPO 2: CONTEXTO TEMPORAL CRYPTO (4 features) ===
        df = AdvancedFeaturesEngine3m._add_temporal_context(df)
        
        # === GRUPO 3: MOMENTUM MULTI-TIMEFRAME (5 features) ===
        df = AdvancedFeaturesEngine3m._add_momentum_features(df)
        
        # === GRUPO 4: DIVERGENCIAS Y CORRELACIONES (4 features) ===
        df = AdvancedFeaturesEngine3m._add_divergence_features(df)
        
        # === GRUPO 5: RÉGIMEN DE MERCADO (3 features) ===
        df = AdvancedFeaturesEngine3m._add_market_regime_features(df)
        
        # === GRUPO 6: LIQUIDEZ Y FLUJO DE ÓRDENES (4 features) ===
        df = AdvancedFeaturesEngine3m._add_liquidity_features(df)
        
        # === GRUPO 7: FEATURES ESPECÍFICAS CRYPTO (4 features) ===
        df = AdvancedFeaturesEngine3m._add_crypto_specific_features(df)
        
        # === GRUPO 8: FEATURES DE CLUSTERING TEMPORAL (3 features) ===
        df = AdvancedFeaturesEngine3m._add_temporal_clustering_features(df)
        
        # === GRUPO 9: FEATURES DE INTEGRACIÓN (5 features) ===
        df = AdvancedFeaturesEngine3m._add_integration_features(df)
        
        # 🎯 FILTRAR FEATURES PARA QUE COINCIDAN EXACTAMENTE CON EL MODELO
        if feature_columns is not None:
            print(f"🔧 Filtrando features para compatibilidad con modelo...")
            print(f"   📊 Features disponibles: {len(df.columns)}")
            print(f"   🎯 Features requeridas: {len(feature_columns)}")
            
            # Verificar que todas las features requeridas estén disponibles
            missing_features = [col for col in feature_columns if col not in df.columns]
            if missing_features:
                print(f"⚠️ Features faltantes: {missing_features}")
                # Crear features faltantes con valores por defecto
                for feature in missing_features:
                    if 'close_time' in feature:
                        df[feature] = pd.Timestamp.now().timestamp()
                    elif 'trades' in feature:
                        df[feature] = 1000  # Valor por defecto
                    else:
                        df[feature] = 0.0
                    print(f"   ➕ Feature faltante creada: {feature}")
            
            # Filtrar DataFrame para incluir solo las features requeridas
            df_filtered = df[feature_columns].copy()
            print(f"✅ Features filtradas para compatibilidad: {df_filtered.shape}")
            return df_filtered
        else:
            print(f"⚠️ No se especificaron feature_columns, retornando features completas")
            return df
    
    @staticmethod
    def _add_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        GRUPO 1: Microestructura de precios - características intrabar críticas
        Estas features capturan información que los indicadores tradicionales pierden
        """
        
        # 1. Price Velocity (velocidad del precio)
        df['price_velocity'] = df['close'].pct_change(1)
        
        # 2. Price Acceleration (aceleración del precio)
        df['price_acceleration'] = df['price_velocity'].diff(1)
        
        # 3. Upper Wick Ratio (ratio de mecha superior)
        # Mide presión vendedora - crítico para detectar rechazos
        df['wick_upper_ratio'] = np.where(
            (df['high'] - df['low']) > 0,
            (df['high'] - np.maximum(df['open'], df['close'])) / (df['high'] - df['low']),
            0.0
        )
        
        # 4. Lower Wick Ratio (ratio de mecha inferior) 
        # Mide presión compradora - crítico para detectar soporte
        df['wick_lower_ratio'] = np.where(
            (df['high'] - df['low']) > 0,
            (np.minimum(df['open'], df['close']) - df['low']) / (df['high'] - df['low']),
            0.0
        )
        
        # 5. Body Size Ratio (tamaño del cuerpo)
        # Mide convicción del movimiento
        df['body_ratio'] = np.where(
            (df['high'] - df['low']) > 0,
            np.abs(df['close'] - df['open']) / (df['high'] - df['low']),
            0.0
        )
        
        # 6. Price Position in Range (posición del cierre en el rango)
        # 0 = close at low, 1 = close at high
        df['close_position'] = np.where(
            (df['high'] - df['low']) > 0,
            (df['close'] - df['low']) / (df['high'] - df['low']),
            0.5
        )
        
        return df
    
    @staticmethod
    def _add_temporal_context(df: pd.DataFrame) -> pd.DataFrame:
        """
        GRUPO 2: Contexto temporal específico para crypto
        Crypto nunca duerme, pero tiene patrones horarios claros
        """
        
        # Extraer componentes temporales
        df['hour'] = df.index.hour
        df['minute'] = df.index.minute
        df['day_of_week'] = df.index.dayofweek
        
        # 1. Hour of Day (codificación cíclica)
        # Captura patrones de volatilidad por hora
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # 2. Market Session Intensity
        # Combina sesiones de mercados tradicionales
        # 0-6 UTC: Asia baja, 6-14 UTC: Europa alta, 14-22 UTC: USA alta
        def get_session_intensity(hour):
            if 6 <= hour < 14:  # European session
                return 0.8
            elif 14 <= hour < 22:  # US session
                return 1.0
            elif 22 <= hour or hour < 6:  # Asian session
                return 0.6
            else:
                return 0.4
        
        df['session_intensity'] = df['hour'].apply(get_session_intensity)
        
        # 3. Weekend Effect (efecto fin de semana)
        # Crypto sigue operando pero con menor volumen institucional
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(float)
        
        # Limpiar columnas temporarias
        df = df.drop(['hour', 'minute', 'day_of_week'], axis=1)
        
        return df
    
    @staticmethod
    def _add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        GRUPO 3: Momentum multi-timeframe optimizado para 3m
        Detecta cambios de momentum en diferentes horizontes temporales
        """
        
        # 1. Short-term Momentum (9 minutos)
        df['momentum_3'] = df['close'].pct_change(3)
        
        # 2. Medium-term Momentum (15 minutos)
        df['momentum_5'] = df['close'].pct_change(5)
        
        # 3. Long-term Momentum (30 minutos)
        df['momentum_10'] = df['close'].pct_change(10)
        
        # 4. Momentum Convergence/Divergence
        # Detecta si los diferentes momentums están alineados
        df['momentum_alignment'] = np.sign(df['momentum_3']) + np.sign(df['momentum_5']) + np.sign(df['momentum_10'])
        
        # 5. Momentum Acceleration
        # Segunda derivada del precio - detecta cambios de tendencia
        df['momentum_acceleration'] = df['momentum_3'].diff(1)
        
        return df
    
    @staticmethod
    def _add_divergence_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        GRUPO 4: Divergencias y correlaciones - detección de señales avanzadas
        Estas features detectan cuando precio e indicadores se desalinean
        """
        
        # Requiere que los indicadores base estén presentes
        required_indicators = ['rsi_9', 'volume_delta', 'macd_histogram']
        
        # 1. RSI-Price Divergence (rolling correlation 10 períodos)
        if 'rsi_9' in df.columns:
            try:
                price_changes = df['close'].pct_change(1)
                rsi_changes = df['rsi_9'].diff(1)
                # Calcular correlación rolling manualmente para evitar errores
                rsi_price_corr = []
                for i in range(len(df)):
                    if i < 10:
                        rsi_price_corr.append(0.0)
                    else:
                        pc_window = price_changes.iloc[i-9:i+1]
                        rc_window = rsi_changes.iloc[i-9:i+1]
                        
                        # Verificar que tenemos datos válidos
                        if len(pc_window.dropna()) > 2 and len(rc_window.dropna()) > 2:
                            corr = pc_window.corr(rc_window)
                            rsi_price_corr.append(corr if not np.isnan(corr) else 0.0)
                        else:
                            rsi_price_corr.append(0.0)
                
                df['rsi_price_divergence'] = rsi_price_corr
            except Exception:
                df['rsi_price_divergence'] = 0.0
        else:
            df['rsi_price_divergence'] = 0.0
        
        # 2. Volume-Price Divergence 
        try:
            volume_changes = df['volume'].pct_change(1)
            price_changes = df['close'].pct_change(1)
            
            volume_price_corr = []
            for i in range(len(df)):
                if i < 10:
                    volume_price_corr.append(0.0)
                else:
                    vc_window = volume_changes.iloc[i-9:i+1]
                    pc_window = price_changes.iloc[i-9:i+1]
                    
                    if len(vc_window.dropna()) > 2 and len(pc_window.dropna()) > 2:
                        corr = vc_window.corr(pc_window)
                        volume_price_corr.append(corr if not np.isnan(corr) else 0.0)
                    else:
                        volume_price_corr.append(0.0)
            
            df['volume_price_divergence'] = volume_price_corr
        except Exception:
            df['volume_price_divergence'] = 0.0
        
        # 3. MACD-Price Divergence
        if 'macd_histogram' in df.columns:
            try:
                macd_changes = df['macd_histogram'].diff(1)
                price_changes = df['close'].pct_change(1)
                
                macd_price_corr = []
                for i in range(len(df)):
                    if i < 10:
                        macd_price_corr.append(0.0)
                    else:
                        mc_window = macd_changes.iloc[i-9:i+1]
                        pc_window = price_changes.iloc[i-9:i+1]
                        
                        if len(mc_window.dropna()) > 2 and len(pc_window.dropna()) > 2:
                            corr = mc_window.corr(pc_window)
                            macd_price_corr.append(corr if not np.isnan(corr) else 0.0)
                        else:
                            macd_price_corr.append(0.0)
                
                df['macd_price_divergence'] = macd_price_corr
            except Exception:
                df['macd_price_divergence'] = 0.0
        else:
            df['macd_price_divergence'] = 0.0
        
        # 4. Composite Divergence Score
        # Combina todas las divergencias en un score único
        df['divergence_score'] = (
            df['rsi_price_divergence'].fillna(0) * 0.4 +
            df['volume_price_divergence'].fillna(0) * 0.4 +
            df['macd_price_divergence'].fillna(0) * 0.2
        )
        
        return df
    
    @staticmethod
    def _add_market_regime_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        GRUPO 5: Identificación de régimen de mercado
        Clasifica el estado actual del mercado para contextualizar señales
        """
        
        # 1. Volatility Regime (régimen de volatilidad)
        # Usa percentil rolling de ATR para clasificar
        if 'atr_percent' in df.columns:
            atr_percentile = df['atr_percent'].rolling(60).rank(pct=True)
            df['vol_regime'] = np.where(
                atr_percentile > 0.8, 1.0,  # High vol
                np.where(atr_percentile < 0.2, -1.0, 0.0)  # Low vol, Medium vol
            )
        else:
            df['vol_regime'] = 0.0
        
        # 2. Trend Strength (fuerza de tendencia)
        # Basado en separación de EMAs
        if 'ema_8' in df.columns and 'ema_17' in df.columns:
            df['trend_strength'] = np.abs(df['ema_8'] - df['ema_17']) / df['ema_17']
        else:
            df['trend_strength'] = 0.0
        
        # 3. Market State (estado del mercado)
        # Combina volatilidad y tendencia para clasificar estado
        # 0 = Ranging, 1 = Trending, -1 = Choppy
        df['market_state'] = np.where(
            (df['vol_regime'] == 1.0) & (df['trend_strength'] > 0.02), 1.0,  # Trending
            np.where(df['vol_regime'] == 1.0, -1.0, 0.0)  # Choppy, Ranging
        )
        
        return df
    
    @staticmethod
    def _add_liquidity_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        GRUPO 6: Características de liquidez y flujo de órdenes
        Críticas para entender la microestructura del mercado crypto
        """
        
        # 1. Volume Intensity (intensidad de volumen normalizada)
        volume_ma = df['volume'].rolling(20).mean()
        df['volume_intensity'] = df['volume'] / volume_ma
        
        # 2. Price Impact (impacto en el precio por unidad de volumen)
        # Mide eficiencia del mercado
        price_change = np.abs(df['close'].pct_change(1))
        volume_norm = df['volume'] / df['volume'].rolling(20).mean()
        df['price_impact'] = np.where(
            volume_norm > 0,
            price_change / volume_norm,
            0.0
        )
        
        # 3. Bid-Ask Spread Proxy (aproximación del spread)
        # Usa el rango high-low como proxy del spread real
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        df['spread_proxy'] = (df['high'] - df['low']) / typical_price
        
        # 4. Liquidity Score (score de liquidez combinado)
        # Alto volumen + bajo impacto = alta liquidez
        df['liquidity_score'] = np.where(
            df['price_impact'] > 0,
            df['volume_intensity'] / (1 + df['price_impact']),
            df['volume_intensity']
        )
        
        return df
    
    @staticmethod
    def _add_crypto_specific_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        GRUPO 7: Features específicas del mercado crypto
        Características únicas que no se encuentran en mercados tradicionales
        """
        
        # 1. Pump Detection (detección de pumps)
        # Combina volumen extremo + movimiento de precio rápido
        volume_spike = df['volume'] > df['volume'].rolling(20).mean() * 3
        price_spike = np.abs(df['close'].pct_change(1)) > 0.03  # 3% en 3 minutos
        df['pump_signal'] = (volume_spike & price_spike).astype(float)
        
        # 2. Dump Detection (detección de dumps)
        price_dump = df['close'].pct_change(1) < -0.03  # -3% en 3 minutos
        df['dump_signal'] = (volume_spike & price_dump).astype(float)
        
        # 3. FOMO Indicator (indicador de FOMO)
        # Secuencia de velas verdes consecutivas con volumen creciente
        green_candle = (df['close'] > df['open']).astype(int)
        volume_increasing = (df['volume'] > df['volume'].shift(1)).astype(int)
        fomo_condition = green_candle & volume_increasing
        df['fomo_intensity'] = fomo_condition.rolling(5).sum() / 5
        
        # 4. Whale Activity Proxy (proxy de actividad de ballenas)
        # Detecta transacciones inusualmente grandes
        volume_z_score = (df['volume'] - df['volume'].rolling(60).mean()) / df['volume'].rolling(60).std()
        df['whale_activity'] = np.clip(volume_z_score, -3, 3) / 3  # Normalizado [-1, 1]
        
        return df
    
    @staticmethod
    def _add_temporal_clustering_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        GRUPO 8: Features de clustering temporal
        Detectan patrones de agrupamiento en tiempo
        """
        
        # 1. Volatility Clustering (clustering de volatilidad)
        # GARCH-like effect: alta volatilidad tiende a seguir a alta volatilidad
        returns = df['close'].pct_change(1)
        vol_proxy = returns.abs().rolling(10).mean()
        df['vol_clustering'] = vol_proxy / vol_proxy.rolling(30).mean()
        
        # 2. Volume Clustering (clustering de volumen)
        # Períodos de alto volumen tienden a continuar
        vol_ma_short = df['volume'].rolling(5).mean()
        vol_ma_long = df['volume'].rolling(20).mean()
        df['volume_clustering'] = vol_ma_short / vol_ma_long
        
        # 3. Pattern Persistence (persistencia de patrones)
        # Mide la tendencia del precio a continuar en la misma dirección
        direction = np.sign(df['close'].pct_change(1))
        df['pattern_persistence'] = direction.rolling(10).mean()
        
        return df
    
    @staticmethod
    def _add_integration_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        GRUPO 9: Features de integración - combinaciones inteligentes de indicadores core
        Estas features combinan múltiples indicadores para detectar patrones complejos
        """
        
        # 1. Technical Confluence Score
        # Combina señales de RSI, MACD y Stochastic para momentum
        try:
            rsi_signal = np.where(df['rsi_9'] < 30, 1, np.where(df['rsi_9'] > 70, -1, 0))
            macd_signal = np.where(df['macd_histogram'] > 0, 1, -1)
            stoch_signal = np.where(df['stoch_k'] < 20, 1, np.where(df['stoch_k'] > 80, -1, 0))
            
            df['technical_confluence'] = (rsi_signal + macd_signal + stoch_signal) / 3
        except Exception:
            df['technical_confluence'] = 0.0
        
        # 2. Trend-Volume Alignment
        # Detecta alineación entre tendencia (EMA) y volumen
        try:
            price_above_ema = (df['close'] > df['ema_17']).astype(int)
            volume_strength = np.where(df['volume_ratio'] > 1.2, 1, 0)
            volume_delta_direction = np.where(df['volume_delta'] > 0.1, 1, 
                                           np.where(df['volume_delta'] < -0.1, -1, 0))
            
            # Alineación: precio alcista + volumen fuerte + delta positivo = 1
            # Desalineación: señales contradictorias = -1
            df['trend_volume_alignment'] = np.where(
                (price_above_ema == 1) & (volume_strength == 1) & (volume_delta_direction == 1), 1,
                np.where(
                    (price_above_ema == 0) & (volume_strength == 1) & (volume_delta_direction == -1), -1,
                    0
                )
            )
        except Exception:
            df['trend_volume_alignment'] = 0.0
        
        # 3. Volatility-Adjusted Momentum
        # Ajusta el price momentum por la volatilidad actual
        try:
            # Normalizar momentum por ATR para ajustar por volatilidad
            atr_norm = df['atr_percent'].rolling(20).mean()
            momentum_adj = np.where(
                atr_norm > 0,
                df['price_momentum'] / (atr_norm + 0.01),  # +0.01 para evitar división por 0
                df['price_momentum']
            )
            df['volatility_adjusted_momentum'] = momentum_adj
        except Exception:
            df['volatility_adjusted_momentum'] = df.get('price_momentum', 0.0)
        
        # 4. Support/Resistance Proximity
        # Detecta proximidad a niveles de soporte/resistencia usando Bollinger y VWAP
        try:
            # Distancia a bandas de Bollinger normalizada
            bb_distance = np.minimum(
                np.abs(df['bollinger_position'] - 0),  # Distancia a banda inferior
                np.abs(df['bollinger_position'] - 1)   # Distancia a banda superior
            )
            
            # Distancia a VWAP normalizada (convertir a escala 0-1)
            vwap_dist_norm = np.abs(df['vwap_distance']) / 100
            vwap_dist_norm = np.clip(vwap_dist_norm, 0, 1)
            
            # Proximidad combinada (0 = lejos de niveles, 1 = cerca de niveles)
            df['support_resistance_proximity'] = 1 - ((bb_distance + vwap_dist_norm) / 2)
        except Exception:
            df['support_resistance_proximity'] = 0.5
        
        # 5. Market Regime Consistency
        # Verifica consistencia entre diferentes indicadores de régimen
        try:
            # Señales de tendencia
            ema_trend = np.where(df['ema_8'] > df['ema_17'], 1, -1)
            macd_trend = np.where(df['macd'] > df['macd_signal'], 1, -1)
            
            # Señales de momentum
            rsi_momentum = np.where(df['rsi_9'] > 50, 1, -1)
            price_mom_sign = np.sign(df['price_momentum'])
            
            # Consistencia: todas las señales apuntan en la misma dirección
            all_signals = np.array([ema_trend, macd_trend, rsi_momentum, price_mom_sign])
            signal_consistency = np.abs(np.mean(all_signals, axis=0))
            
            df['market_regime_consistency'] = signal_consistency
        except Exception:
            df['market_regime_consistency'] = 0.5
        
        return df
    
    @staticmethod
    def get_feature_importance_guide() -> Dict[str, Dict[str, str]]:
        """
        Guía de importancia y uso de features para el modelo TCN
        """
        return {
            "ULTRA_ALTA_PRIORIDAD": {
                "indicadores_core": "rsi_9, macd_histogram, ema_8, ema_17, bollinger_position, vwap_distance, volume_delta, volume_ratio, price_momentum, stoch_k",
                "integracion": "technical_confluence, trend_volume_alignment, market_regime_consistency"
            },
            "ALTA_PRIORIDAD": {
                "microestructura": "price_velocity, wick_upper_ratio, wick_lower_ratio, body_ratio, close_position",
                "momentum_avanzado": "momentum_3, momentum_5, momentum_alignment, volatility_adjusted_momentum",
                "flujo_ordenes": "volume_intensity, price_impact, liquidity_score",
                "crypto_specific": "pump_signal, dump_signal, fomo_intensity",
                "niveles_clave": "support_resistance_proximity"
            },
            "MEDIA_PRIORIDAD": {
                "temporal": "hour_sin, hour_cos, session_intensity",
                "divergencias": "rsi_price_divergence, volume_price_divergence, macd_price_divergence, divergence_score",
                "regimen": "vol_regime, trend_strength, market_state"
            },
            "BAJA_PRIORIDAD": {
                "clustering": "vol_clustering, volume_clustering, pattern_persistence",
                "auxiliares": "is_weekend, whale_activity, spread_proxy, price_acceleration"
            },
            "FEATURES_TECNICAS_BASE": {
                "momentum_base": "rsi_9, macd, macd_signal, stoch_d",
                "tendencia_base": "bollinger_upper, bollinger_middle, bollinger_lower, vwap",
                "volatilidad_base": "atr, atr_percent",
                "volumen_base": "obv, obv_sma",
                "estructura_base": "heikin_ashi_signal"
            }
        }
    
    @staticmethod
    def validate_feature_set(df: pd.DataFrame) -> Dict[str, any]:
        """
        Validar el conjunto de features creado con arquitectura integrada
        """
        # Categorías de features
        base_features = ['open', 'high', 'low', 'close', 'volume']  # 5 features
        technical_indicators = 22  # 22 indicadores técnicos del predictor core
        advanced_features = 33     # Features avanzadas originales
        integration_features = 5   # Nuevas features de integración
        
        total_expected = len(base_features) + technical_indicators + advanced_features + integration_features
        
        # Validación detallada por categorías
        validation_details = {}
        
        # Verificar features base OHLCV
        base_present = [col for col in base_features if col in df.columns]
        validation_details['base_features'] = {
            'expected': len(base_features),
            'present': len(base_present),
            'missing': [col for col in base_features if col not in df.columns],
            'complete': len(base_present) == len(base_features)
        }
        
        # Verificar indicadores técnicos core
        core_indicators = [
            'rsi_9', 'macd', 'macd_signal', 'macd_histogram',
            'ema_8', 'ema_17', 'bollinger_upper', 'bollinger_middle', 'bollinger_lower',
            'bollinger_position', 'vwap', 'vwap_distance', 'atr', 'atr_percent',
            'volume_delta', 'obv', 'obv_sma', 'volume_ratio',
            'price_momentum', 'stoch_k', 'stoch_d', 'heikin_ashi_signal'
        ]
        core_present = [col for col in core_indicators if col in df.columns]
        validation_details['core_indicators'] = {
            'expected': len(core_indicators),
            'present': len(core_present),
            'missing': [col for col in core_indicators if col not in df.columns],
            'complete': len(core_present) == len(core_indicators)
        }
        
        # Verificar features de integración
        integration_indicators = [
            'technical_confluence', 'trend_volume_alignment', 'volatility_adjusted_momentum',
            'support_resistance_proximity', 'market_regime_consistency'
        ]
        integration_present = [col for col in integration_indicators if col in df.columns]
        validation_details['integration_features'] = {
            'expected': len(integration_indicators),
            'present': len(integration_present),
            'missing': [col for col in integration_indicators if col not in df.columns],
            'complete': len(integration_present) == len(integration_indicators)
        }
        
        # Calcular métricas generales
        null_percentage = (df.isnull().sum() / len(df)).mean()
        total_critical_features = len(base_features) + len(core_indicators) + len(integration_indicators)
        critical_present = len(base_present) + len(core_present) + len(integration_present)
        
        # Score de calidad basado en completitud crítica
        completeness_score = (critical_present / total_critical_features) * 100
        data_quality_score = max(0, completeness_score - (null_percentage * 100))
        
        return {
            "total_features": len(df.columns),
            "expected_features": total_expected,
            "critical_features_present": critical_present,
            "critical_features_expected": total_critical_features,
            "features_complete": critical_present >= total_critical_features,
            "null_percentage": round(null_percentage * 100, 2),
            "completeness_score": round(completeness_score, 2),
            "data_quality_score": round(data_quality_score, 2),
            "data_quality": "EXCELLENT" if data_quality_score > 95 else "GOOD" if data_quality_score > 80 else "POOR",
            "recommendation": AdvancedFeaturesEngine3m._get_quality_recommendation(data_quality_score, validation_details),
            "details": validation_details,
            "integration_ready": (
                validation_details['base_features']['complete'] and
                validation_details['core_indicators']['complete'] and
                validation_details['integration_features']['complete'] and
                null_percentage < 0.05
            )
        }
    
    @staticmethod
    def _get_quality_recommendation(quality_score: float, details: Dict) -> str:
        """Generar recomendación basada en el score de calidad"""
        if quality_score > 95:
            return "✅ EXCELENTE: Listo para entrenamiento TCN avanzado"
        elif quality_score > 80:
            return "✅ BUENO: Listo para entrenamiento TCN básico"
        elif quality_score > 60:
            missing_critical = []
            for category, info in details.items():
                if not info['complete'] and info['missing']:
                    missing_critical.extend(info['missing'])
            return f"⚠️ MEJORABLE: Faltan features críticas: {missing_critical[:3]}..."
        else:
            return "❌ POBRE: Requiere corrección de datos antes del entrenamiento"


# === FUNCIONES DE INTEGRACIÓN PRINCIPAL ===

def create_tcn_dataset_3m(symbol: str = None, df: pd.DataFrame = None, sequence_length: int = 60, 
                         target_lookahead: int = 3, buy_threshold: float = 0.015, 
                         sell_threshold: float = -0.01) -> Tuple[np.ndarray, np.ndarray, Dict[str, any]]:
    """
    Crear dataset completo para entrenamiento del modelo TCN 3m con integración completa
    
    Args:
        symbol: Símbolo del par crypto (requerido si df es None)
        df: DataFrame con datos (opcional, se obtendrá de symbol si no se proporciona)
        sequence_length: Longitud de secuencia (60 = 3 horas de datos en 3m)
        target_lookahead: Períodos hacia adelante para predicción (3 = 9 minutos)
        buy_threshold: Umbral para clasificar como BUY (1.5% por defecto)
        sell_threshold: Umbral para clasificar como SELL (-1% por defecto)
    
    Returns:
        X: Features array (samples, timesteps, features)
        y: Labels array (samples, classes) [0=SELL, 1=HOLD, 2=BUY]
        metadata: Diccionario con información del dataset
    """
    
    print(f"🚀 CREANDO DATASET TCN 3M INTEGRADO")
    print("=" * 50)
    
    # Crear features completas con integración
    if df is None:
        if symbol is None:
            raise ValueError("Debe proporcionar 'symbol' o 'df'")
        
        print(f"📊 Extrayendo datos completos para {symbol}...")
        df_features = AdvancedFeaturesEngine3m.create_complete_feature_set(symbol=symbol)
    else:
        print(f"📊 Procesando DataFrame proporcionado...")
        df_features = AdvancedFeaturesEngine3m.create_complete_feature_set(df=df)
    
    if df_features is None or len(df_features) < sequence_length + target_lookahead + 10:
        raise ValueError(f"Datos insuficientes. Necesarios: {sequence_length + target_lookahead + 10}, disponibles: {len(df_features) if df_features is not None else 0}")
    
    # Validar calidad de features
    validation_result = AdvancedFeaturesEngine3m.validate_feature_set(df_features)
    print(f"🔍 Validación de features: {validation_result['data_quality']} ({validation_result['data_quality_score']:.1f}%)")
    
    if not validation_result['integration_ready']:
        print(f"⚠️ ADVERTENCIA: {validation_result['recommendation']}")
        if validation_result['data_quality_score'] < 60:
            raise ValueError("Calidad de datos insuficiente para entrenamiento TCN")
    
    # Crear labels con lógica avanzada
    print(f"🎯 Generando labels con lookahead={target_lookahead}, buy_threshold={buy_threshold:.1%}, sell_threshold={sell_threshold:.1%}")
    
    # Calcular retornos futuros
    future_return = df_features['close'].shift(-target_lookahead) / df_features['close'] - 1
    
    # Clasificación multi-clase con lógica mejorada
    labels = np.where(
        future_return > buy_threshold, 2,      # BUY si retorno > buy_threshold
        np.where(future_return < sell_threshold, 0, 1)  # SELL si retorno < sell_threshold, else HOLD
    )
    
    # Estadísticas de labels
    unique_labels, label_counts = np.unique(labels[~np.isnan(labels)], return_counts=True)
    label_distribution = dict(zip(unique_labels, label_counts))
    print(f"📊 Distribución de labels: {label_distribution}")
    
    # Preparar features para secuencias
    print(f"🔧 Preparando secuencias de longitud {sequence_length}...")
    
    # Seleccionar features excluyendo target y columnas no numéricas
    exclude_columns = ['close']  # Excluir precio de cierre como feature
    numeric_columns = df_features.select_dtypes(include=[np.number]).columns.tolist()
    feature_columns = [col for col in numeric_columns if col not in exclude_columns]
    
    print(f"📈 Features seleccionadas: {len(feature_columns)}")
    
    # Limpiar datos y preparar array
    features_df = df_features[feature_columns].copy()
    
    # Manejo robusto de valores faltantes
    # 1. Forward fill para continuidad temporal
    features_df = features_df.fillna(method='ffill')
    # 2. Backward fill para valores iniciales
    features_df = features_df.fillna(method='bfill')
    # 3. Rellenar con mediana como último recurso
    features_df = features_df.fillna(features_df.median())
    
    features_array = features_df.values
    
    # Validar que no hay NaN en features
    if np.isnan(features_array).any():
        nan_cols = features_df.columns[features_df.isna().any()].tolist()
        print(f"⚠️ ADVERTENCIA: NaN encontrados en columnas: {nan_cols}")
        # Rellenar NaN restantes con 0
        features_array = np.nan_to_num(features_array, nan=0.0)
    
    # Crear secuencias para TCN
    print(f"🔄 Generando secuencias TCN...")
    X, y = [], []
    valid_indices = []
    
    for i in range(sequence_length, len(features_array) - target_lookahead):
        # Verificar que el label es válido
        if not np.isnan(labels[i]):
            X.append(features_array[i-sequence_length:i])
            y.append(int(labels[i]))
            valid_indices.append(i)
    
    X = np.array(X)
    y = np.array(y)
    
    # Verificar shapes
    if len(X) == 0:
        raise ValueError("No se pudieron generar secuencias válidas")
    
    print(f"✅ Dataset TCN creado exitosamente:")
    print(f"   📊 Shape X: {X.shape} (samples, timesteps, features)")
    print(f"   🎯 Shape y: {y.shape} (samples,)")
    print(f"   📈 Features por timestep: {X.shape[2]}")
    print(f"   ⏰ Secuencia temporal: {X.shape[1]} períodos ({X.shape[1] * 3} minutos)")
    
    # Estadísticas finales
    final_label_distribution = dict(zip(*np.unique(y, return_counts=True)))
    print(f"   📊 Distribución final: SELL={final_label_distribution.get(0, 0)}, HOLD={final_label_distribution.get(1, 0)}, BUY={final_label_distribution.get(2, 0)}")
    
    # Crear metadata completa
    metadata = {
        'symbol': symbol,
        'total_samples': len(X),
        'sequence_length': sequence_length,
        'features_count': X.shape[2],
        'target_lookahead': target_lookahead,
        'buy_threshold': buy_threshold,
        'sell_threshold': sell_threshold,
        'label_distribution': final_label_distribution,
        'feature_columns': feature_columns,
        'data_quality': validation_result['data_quality'],
        'data_quality_score': validation_result['data_quality_score'],
        'validation_details': validation_result,
        'dataset_shape': {'X': X.shape, 'y': y.shape},
        'temporal_coverage_minutes': sequence_length * 3,
        'prediction_horizon_minutes': target_lookahead * 3,
        'creation_timestamp': datetime.now().isoformat(),
        'tcn_ready': True
    }
    
    return X, y, metadata


def create_tcn_dataset_multiple_symbols(symbols: List[str], sequence_length: int = 60, 
                                      combine_datasets: bool = True) -> Dict[str, Tuple[np.ndarray, np.ndarray, Dict]]:
    """
    Crear datasets TCN para múltiples símbolos
    
    Args:
        symbols: Lista de símbolos a procesar
        sequence_length: Longitud de secuencia
        combine_datasets: Si True, combina todos los datasets en uno solo
        
    Returns:
        Diccionario con datasets por símbolo o dataset combinado
    """
    print(f"🚀 CREANDO DATASETS TCN PARA {len(symbols)} SÍMBOLOS")
    print("=" * 60)
    
    datasets = {}
    
    for symbol in symbols:
        try:
            print(f"\n📊 Procesando {symbol}...")
            X, y, metadata = create_tcn_dataset_3m(symbol=symbol, sequence_length=sequence_length)
            datasets[symbol] = (X, y, metadata)
            print(f"✅ {symbol} completado: {X.shape[0]} muestras")
        except Exception as e:
            print(f"❌ Error procesando {symbol}: {e}")
            continue
    
    if combine_datasets and len(datasets) > 1:
        print(f"\n🔗 Combinando datasets de {len(datasets)} símbolos...")
        
        # Combinar todos los X e y
        combined_X = np.concatenate([data[0] for data in datasets.values()], axis=0)
        combined_y = np.concatenate([data[1] for data in datasets.values()], axis=0)
        
        # Crear metadata combinada
        combined_metadata = {
            'symbols': list(datasets.keys()),
            'total_samples': len(combined_X),
            'individual_counts': {symbol: data[0].shape[0] for symbol, data in datasets.items()},
            'sequence_length': sequence_length,
            'features_count': combined_X.shape[2],
            'combined_distribution': dict(zip(*np.unique(combined_y, return_counts=True))),
            'dataset_shape': {'X': combined_X.shape, 'y': combined_y.shape},
            'creation_timestamp': datetime.now().isoformat(),
            'tcn_ready': True
        }
        
        datasets['COMBINED'] = (combined_X, combined_y, combined_metadata)
        print(f"✅ Dataset combinado creado: {combined_X.shape[0]} muestras totales")
    
    return datasets


# === EJEMPLO DE USO INTEGRADO ===
if __name__ == "__main__":
    print("🚀 ENHANCED FEATURES ENGINEERING FOR TCN 3M MODEL - INTEGRACIÓN COMPLETA")
    print("=" * 80)
    
    try:
        # Test 1: Crear features con datos reales de Binance
        print("\n📊 TEST 1: INTEGRACIÓN CON DATOS REALES")
        print("-" * 50)
        
        test_symbol = "BTCUSDT"
        print(f"🔄 Testear integración completa con {test_symbol}...")
        
        # Crear features completas usando la nueva arquitectura
        enhanced_df = AdvancedFeaturesEngine3m.create_complete_feature_set(symbol=test_symbol)
        
        if enhanced_df is not None:
            print(f"✅ Features creadas exitosamente: {enhanced_df.shape}")
            
            # Validar conjunto de features
            validation = AdvancedFeaturesEngine3m.validate_feature_set(enhanced_df)
            
            print(f"\n🔍 VALIDACIÓN DE FEATURES:")
            print(f"   Total features: {validation['total_features']}")
            print(f"   Features críticas: {validation['critical_features_present']}/{validation['critical_features_expected']}")
            print(f"   Score de calidad: {validation['data_quality_score']:.1f}%")
            print(f"   Estado: {validation['data_quality']}")
            print(f"   Recomendación: {validation['recommendation']}")
            print(f"   Listo para TCN: {'✅' if validation['integration_ready'] else '❌'}")
            
            # Mostrar detalles por categoría
            for category, details in validation['details'].items():
                status = "✅" if details['complete'] else "❌"
                print(f"   {category}: {details['present']}/{details['expected']} {status}")
        
        # Test 2: Crear dataset TCN completo
        print(f"\n🧠 TEST 2: CREACIÓN DE DATASET TCN")
        print("-" * 50)
        
        X, y, metadata = create_tcn_dataset_3m(
            symbol=test_symbol,
            sequence_length=60,
            target_lookahead=3,
            buy_threshold=0.015,
            sell_threshold=-0.01
        )
        
        print(f"\n✅ DATASET TCN CREADO EXITOSAMENTE:")
        print(f"   📊 Forma X: {X.shape}")
        print(f"   🎯 Forma y: {y.shape}")
        print(f"   📈 Features: {metadata['features_count']}")
        print(f"   ⏰ Secuencia: {metadata['sequence_length']} períodos ({metadata['temporal_coverage_minutes']} min)")
        print(f"   🔮 Predicción: {metadata['target_lookahead']} períodos ({metadata['prediction_horizon_minutes']} min)")
        print(f"   📊 Distribución: {metadata['label_distribution']}")
        print(f"   🎯 Calidad: {metadata['data_quality']} ({metadata['data_quality_score']:.1f}%)")
        
        # Test 3: Mostrar guía de features
        print(f"\n📋 TEST 3: GUÍA DE IMPORTANCIA DE FEATURES")
        print("-" * 50)
        
        feature_guide = AdvancedFeaturesEngine3m.get_feature_importance_guide()
        for priority, categories in feature_guide.items():
            print(f"\n🎯 {priority}:")
            for category, features in categories.items():
                feature_count = len(features.split(', '))
                print(f"   📈 {category} ({feature_count} features): {features[:80]}...")
        
        # Test 4: Features de integración específicas
        print(f"\n🔗 TEST 4: FEATURES DE INTEGRACIÓN ESPECÍFICAS")
        print("-" * 50)
        
        integration_features = [
            'technical_confluence', 'trend_volume_alignment', 'volatility_adjusted_momentum',
            'support_resistance_proximity', 'market_regime_consistency'
        ]
        
        for feature in integration_features:
            if feature in enhanced_df.columns:
                values = enhanced_df[feature].dropna()
                if len(values) > 0:
                    print(f"   ✅ {feature}: min={values.min():.3f}, max={values.max():.3f}, mean={values.mean():.3f}")
                else:
                    print(f"   ⚠️ {feature}: Sin datos válidos")
            else:
                print(f"   ❌ {feature}: No encontrada")
        
        # Test 5: Verificar compatibilidad TCN
        print(f"\n🧠 TEST 5: VERIFICACIÓN COMPATIBILIDAD TCN")
        print("-" * 50)
        
        # Verificar dimensiones
        assert len(X.shape) == 3, f"X debe ser 3D, got shape {X.shape}"
        assert len(y.shape) == 1, f"y debe ser 1D, got shape {y.shape}"
        assert X.shape[0] == y.shape[0], f"Mismatch en samples: X={X.shape[0]}, y={y.shape[0]}"
        
        # Verificar valores
        assert not np.isnan(X).any(), "X contiene NaN"
        assert not np.isnan(y).any(), "y contiene NaN"
        assert set(y).issubset({0, 1, 2}), f"y debe contener solo 0,1,2, got {set(y)}"
        
        print(f"   ✅ Dimensiones correctas: X{X.shape}, y{y.shape}")
        print(f"   ✅ Sin valores NaN")
        print(f"   ✅ Labels válidas: {set(y)}")
        print(f"   ✅ Listo para entrenamiento TCN")
        
        print(f"\n🎉 INTEGRACIÓN COMPLETA EXITOSA")
        print("=" * 80)
        print("🏆 RESUMEN DE CAPACIDADES IMPLEMENTADAS:")
        print("   ✅ Extracción automática de indicadores técnicos del predictor core")
        print("   ✅ Generación de 38 features avanzadas adicionales")
        print("   ✅ 5 features de integración que combinan indicadores core")
        print("   ✅ Sistema de validación de calidad de datos")
        print("   ✅ Creación automática de datasets TCN listos para entrenamiento")
        print("   ✅ Soporte para múltiples símbolos y combinación de datasets")
        print("   ✅ Metadata completa para reproducibilidad")
        print("   ✅ Manejo robusto de valores faltantes")
        print("   ✅ Verificación de compatibilidad TCN")
        print("")
        print("📈 TOTAL FEATURES DISPONIBLES: ~65 (OHLCV + 22 técnicas + 33 avanzadas + 5 integración)")
        print("🎯 OPTIMIZADO PARA: Modelos TCN con secuencias temporales en timeframe 3m")
        print("🔗 COMPATIBLE CON: Ensemble híbrido, sistema de gestión de riesgo")
        
    except Exception as e:
        print(f"\n❌ ERROR EN INTEGRACIÓN: {e}")
        print("⚠️ Verifique configuración de API keys y conexión a Binance")
        
        # Test con datos simulados como fallback
        print(f"\n🔄 FALLBACK: Probando con datos simulados...")
        
        # Crear datos simulados básicos
        dates = pd.date_range('2024-01-01', periods=200, freq='3min')
        sample_df = pd.DataFrame({
            'open': np.random.randn(200).cumsum() + 50000,
            'high': np.random.randn(200).cumsum() + 50200,
            'low': np.random.randn(200).cumsum() + 49800,
            'close': np.random.randn(200).cumsum() + 50000,
            'volume': np.random.lognormal(10, 1, 200),
        }, index=dates)
        
        # Intentar crear features básicas
        try:
            enhanced_df_sim = AdvancedFeaturesEngine3m.create_complete_feature_set(df=sample_df)
            if enhanced_df_sim is not None:
                print(f"✅ Features simuladas creadas: {enhanced_df_sim.shape}")
            else:
                print("❌ Error creando features simuladas")
        except Exception as sim_error:
            print(f"❌ Error con datos simulados: {sim_error}")
    
    print(f"\n🔚 Motor de features 3M integrado completado")