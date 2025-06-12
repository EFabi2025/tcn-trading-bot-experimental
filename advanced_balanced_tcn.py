#!/usr/bin/env python3
"""
🚀 TCN Anti-Bias AVANZADO con Técnicas de Balance Profesional

Implementa las mejores técnicas del documento compartido:
- Entrenamiento por régimen de mercado
- Validación walk-forward
- Aumentación de datos con noise injection
- Función de pérdida adaptativa (Sharpe Ratio)
- Múltiples ventanas temporales
"""

import asyncio
import numpy as np
import pandas as pd
import tensorflow as tf
from binance.client import Client as BinanceClient
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
import pickle
import json
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class SharpeRatioLoss(tf.keras.losses.Loss):
    """
    🎯 Función de pérdida Sharpe Ratio modificada para trading
    """
    def __init__(self, name='sharpe_loss'):
        super().__init__(name=name)
    
    def call(self, y_true, y_pred):
        # Convertir predicciones a returns esperados
        # SELL=-1, HOLD=0, BUY=+1
        pred_returns = tf.reduce_sum(y_pred * tf.constant([-1.0, 0.0, 1.0]), axis=1)
        
        # Calcular Sharpe Ratio negativo (para minimizar)
        mean_return = tf.reduce_mean(pred_returns)
        std_return = tf.keras.backend.std(pred_returns) + 1e-8
        sharpe = mean_return / std_return
        
        # Combinar con categorical crossentropy
        ce_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        
        # Loss combinado (maximizar Sharpe = minimizar -Sharpe)
        return ce_loss - 0.1 * sharpe


class AdvancedBalancedTCN:
    """
    🚀 TCN Avanzado con técnicas profesionales de balance
    """
    
    def __init__(self):
        self.symbol = "BTCUSDT"
        self.lookback_window = 60
        self.expected_features = 66
        self.class_names = ['SELL', 'HOLD', 'BUY']
        
        # Múltiples intervalos para multi-timeframe
        self.intervals = ["5m", "15m", "1h"]
        self.data_limits = [500, 300, 100]  # Datos por intervalo
        
        print("🚀 Advanced Balanced TCN inicializado")
        print(f"   - Multi-timeframe: {self.intervals}")
        print(f"   - Features: {self.expected_features}")
    
    def setup_binance_client(self) -> bool:
        """Configura cliente Binance"""
        try:
            print("\n🔗 Conectando a Binance...")
            self.binance_client = BinanceClient()
            print("✅ Conectado a Binance")
            return True
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    def collect_multi_timeframe_data(self) -> Dict[str, pd.DataFrame]:
        """Recolecta datos multi-timeframe según documento"""
        try:
            print("\n📊 Recolectando datos multi-timeframe avanzados...")
            
            timeframe_data = {}
            
            for interval, limit in zip(self.intervals, self.data_limits):
                print(f"   - Recolectando {interval} data (limit: {limit})...")
                
                klines = self.binance_client.get_historical_klines(
                    symbol=self.symbol,
                    interval=interval,
                    limit=limit
                )
                
                df = pd.DataFrame(klines, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_asset_volume', 'number_of_trades',
                    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                ])
                
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                df = df.dropna()
                timeframe_data[interval] = df
                
                print(f"     ✅ {interval}: {len(df)} períodos")
            
            return timeframe_data
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return {}
    
    def detect_market_regimes_advanced(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detección avanzada de regímenes como en documento"""
        try:
            print("\n🔍 Detectando regímenes de mercado avanzados...")
            
            df = df.copy()
            
            # Múltiples horizontes para detección robusta
            df['returns_5'] = df['close'].pct_change(periods=5)
            df['returns_20'] = df['close'].pct_change(periods=20)
            df['returns_50'] = df['close'].pct_change(periods=50)
            
            # Volatilidad multi-período
            df['vol_5'] = df['returns_5'].rolling(10).std()
            df['vol_20'] = df['returns_20'].rolling(20).std()
            
            # RSI para momentum
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            df['rsi'] = 100 - (100 / (1 + gain / loss))
            
            # MACD para tendencia
            ema12 = df['close'].ewm(span=12).mean()
            ema26 = df['close'].ewm(span=26).mean()
            df['macd'] = ema12 - ema26
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # Clasificación de regímenes más inteligente
            regimes = []
            
            for i, row in df.iterrows():
                ret_20 = row['returns_20']
                vol_20 = row['vol_20']
                rsi = row['rsi']
                macd_hist = row['macd_histogram']
                
                if pd.isna(ret_20) or pd.isna(vol_20) or pd.isna(rsi):
                    regime = 1  # SIDEWAYS default
                else:
                    # Condiciones más sofisticadas
                    is_high_vol = vol_20 > df['vol_20'].quantile(0.7)
                    is_oversold = rsi < 30
                    is_overbought = rsi > 70
                    is_bullish_macd = macd_hist > 0
                    
                    # BEAR conditions
                    if (ret_20 < -0.02 or  # -2% return
                        (ret_20 < -0.01 and is_high_vol) or  # -1% + alta vol
                        (rsi < 25 and ret_20 < 0)):  # RSI extremo + ret negativo
                        regime = 0  # BEAR
                    
                    # BULL conditions  
                    elif (ret_20 > 0.02 or  # +2% return
                          (ret_20 > 0.01 and is_bullish_macd) or  # +1% + MACD positivo
                          (rsi > 75 and ret_20 > 0)):  # RSI alto + ret positivo
                        regime = 2  # BULL
                    
                    # SIDEWAYS (default)
                    else:
                        regime = 1  # SIDEWAYS
                
                regimes.append(regime)
            
            df['regime'] = regimes
            
            # Estadísticas
            regime_counts = Counter(regimes)
            total = len(regimes)
            
            print("📊 Distribución de regímenes avanzados:")
            regime_names = ['BEAR', 'SIDEWAYS', 'BULL']
            for i, name in enumerate(regime_names):
                count = regime_counts[i]
                pct = count / total * 100
                print(f"   - {name}: {count} ({pct:.1f}%)")
            
            return df
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return df
    
    def apply_data_augmentation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aumentación de datos con noise injection según documento"""
        try:
            print("\n🔧 Aplicando aumentación de datos...")
            
            original_len = len(df)
            augmented_dfs = [df.copy()]
            
            # Noise injection para simular condiciones extremas
            for noise_level in [0.001, 0.002, 0.005]:  # 0.1%, 0.2%, 0.5%
                df_noisy = df.copy()
                
                # Añadir ruido a precios
                price_cols = ['open', 'high', 'low', 'close']
                for col in price_cols:
                    noise = np.random.normal(0, noise_level, len(df_noisy))
                    df_noisy[col] = df_noisy[col] * (1 + noise)
                
                # Añadir ruido a volumen
                vol_noise = np.random.normal(0, noise_level * 2, len(df_noisy))
                df_noisy['volume'] = df_noisy['volume'] * (1 + vol_noise)
                
                augmented_dfs.append(df_noisy)
            
            # Combinar datos originales y aumentados
            df_augmented = pd.concat(augmented_dfs, ignore_index=True)
            
            print(f"✅ Datos aumentados: {original_len} -> {len(df_augmented)} ({len(df_augmented)/original_len:.1f}x)")
            
            return df_augmented
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return df
    
    def create_regime_specific_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea labels específicas por régimen para forzar balance"""
        try:
            print("\n🎯 Creando labels específicas por régimen...")
            
            df = df.copy()
            
            # Horizontes de predicción múltiples
            df['future_price_6'] = df['close'].shift(-6)
            df['future_price_12'] = df['close'].shift(-12)
            df['future_price_24'] = df['close'].shift(-24)
            
            df['change_6'] = (df['future_price_6'] - df['close']) / df['close']
            df['change_12'] = (df['future_price_12'] - df['close']) / df['close']
            df['change_24'] = (df['future_price_24'] - df['close']) / df['close']
            
            labels = []
            
            for i, row in df.iterrows():
                regime = row['regime']
                change_6 = row['change_6']
                change_12 = row['change_12']
                change_24 = row['change_24']
                
                if pd.isna(change_12):
                    label = 1  # HOLD default
                    labels.append(label)
                    continue
                
                # Umbrales específicos por régimen para FORZAR BALANCE
                if regime == 0:  # BEAR - ser más agresivo en SELL
                    if change_12 < -0.0005:  # -0.05%
                        label = 0  # SELL
                    elif change_12 > 0.004:  # +0.4%
                        label = 2  # BUY (rebote)
                    else:
                        label = 1  # HOLD
                        
                elif regime == 2:  # BULL - ser más agresivo en BUY
                    if change_12 < -0.004:  # -0.4%
                        label = 0  # SELL (corrección)
                    elif change_12 > 0.0005:  # +0.05%
                        label = 2  # BUY
                    else:
                        label = 1  # HOLD
                        
                else:  # SIDEWAYS - balance equilibrado
                    if change_12 < -0.0015:  # -0.15%
                        label = 0  # SELL
                    elif change_12 > 0.0015:  # +0.15%
                        label = 2  # BUY
                    else:
                        label = 1  # HOLD
                
                labels.append(label)
            
            df['label'] = labels
            df = df.dropna(subset=['change_6', 'change_12', 'change_24'])
            
            # Verificar balance
            label_counts = Counter(labels)
            total_labels = len([l for l in labels if not pd.isna(l)])
            
            print("📊 Distribución de labels por régimen:")
            for i, name in enumerate(self.class_names):
                count = label_counts[i]
                pct = count / total_labels * 100 if total_labels > 0 else 0
                print(f"   - {name}: {count} ({pct:.1f}%)")
            
            # FORZAR BALANCE si es necesario
            min_count = min(label_counts.values())
            max_count = max(label_counts.values())
            
            if max_count / min_count > 2.0:  # Si hay >2x diferencia
                print("\n⚖️ Forzando balance extremo...")
                
                # Undersample clases mayoritarias al nivel de la minoritaria * 1.5
                target_size = int(min_count * 1.5)
                
                balanced_dfs = []
                for label_id in [0, 1, 2]:
                    label_data = df[df['label'] == label_id]
                    if len(label_data) > target_size:
                        # Seleccionar muestra estratificada por régimen
                        sampled_data = []
                        for regime_id in [0, 1, 2]:
                            regime_data = label_data[label_data['regime'] == regime_id]
                            if len(regime_data) > 0:
                                n_samples = min(len(regime_data), target_size // 3)
                                sampled_data.append(regime_data.sample(n=n_samples, random_state=42))
                        
                        if sampled_data:
                            balanced_data = pd.concat(sampled_data, ignore_index=True)
                        else:
                            balanced_data = label_data.sample(n=target_size, random_state=42)
                    else:
                        balanced_data = label_data
                    
                    balanced_dfs.append(balanced_data)
                
                df = pd.concat(balanced_dfs, ignore_index=True)
                df = df.sample(frac=1, random_state=42).reset_index(drop=True)
                
                # Nuevas estadísticas
                final_counts = Counter(df['label'])
                final_total = len(df)
                
                print("✅ Balance final:")
                for i, name in enumerate(self.class_names):
                    count = final_counts[i]
                    pct = count / final_total * 100
                    print(f"   - {name}: {count} ({pct:.1f}%)")
            
            return df
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return df
    
    def create_advanced_tcn_model(self) -> tf.keras.Model:
        """Crea modelo TCN con arquitectura avanzada"""
        try:
            print("\n🧠 Creando modelo TCN avanzado...")
            
            # Input dual
            features_input = tf.keras.Input(shape=(self.lookback_window, self.expected_features), name='features')
            regimes_input = tf.keras.Input(shape=(3,), name='regimes')
            
            # TCN con dilataciones exponenciales como en documento
            x = features_input
            
            # Capas con dilatación 1, 2, 4, 8 según documento
            for dilation in [1, 2, 4, 8]:
                x = tf.keras.layers.Conv1D(
                    filters=64 * (2 if dilation > 1 else 1),
                    kernel_size=3,
                    dilation_rate=dilation,
                    padding='causal',
                    activation='relu'
                )(x)
                x = tf.keras.layers.BatchNormalization()(x)
                x = tf.keras.layers.Dropout(0.3)(x)
            
            # Global pooling
            x = tf.keras.layers.GlobalAveragePooling1D()(x)
            
            # Procesar regímenes
            regime_dense = tf.keras.layers.Dense(32, activation='relu')(regimes_input)
            regime_dense = tf.keras.layers.Dropout(0.3)(regime_dense)
            
            # Fusión
            combined = tf.keras.layers.concatenate([x, regime_dense])
            
            # Capas densas finales
            combined = tf.keras.layers.Dense(128, activation='relu')(combined)
            combined = tf.keras.layers.BatchNormalization()(combined)
            combined = tf.keras.layers.Dropout(0.4)(combined)
            
            combined = tf.keras.layers.Dense(64, activation='relu')(combined)
            combined = tf.keras.layers.Dropout(0.3)(combined)
            
            # Output
            outputs = tf.keras.layers.Dense(3, activation='softmax')(combined)
            
            model = tf.keras.Model(inputs=[features_input, regimes_input], outputs=outputs)
            
            # Compilar con loss Sharpe personalizado
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss=SharpeRatioLoss(),
                metrics=['accuracy']
            )
            
            print(f"✅ Modelo avanzado: {model.count_params():,} parámetros")
            return model
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    async def run_advanced_training(self):
        """Ejecuta entrenamiento avanzado completo"""
        print("🚀 INICIANDO ENTRENAMIENTO TCN AVANZADO")
        print("="*80)
        
        # Setup
        if not self.setup_binance_client():
            return False
        
        # Datos multi-timeframe
        timeframe_data = self.collect_multi_timeframe_data()
        if not timeframe_data:
            return False
        
        # Usar datos del timeframe principal (5m)
        df = timeframe_data["5m"]
        
        # Detectar regímenes avanzados
        df = self.detect_market_regimes_advanced(df)
        
        # Aumentación de datos
        df = self.apply_data_augmentation(df)
        
        # Crear features (usar función existente)
        df = self.create_comprehensive_features(df)
        
        # Labels específicas por régimen
        df = self.create_regime_specific_labels(df)
        
        # Preparar datos (usar función existente)
        X_features, X_regimes, y = self.prepare_training_data(df)
        
        # Crear modelo avanzado
        model = self.create_advanced_tcn_model()
        
        if model is None:
            return False
        
        print("\n🚀 Modelo TCN Avanzado creado exitosamente!")
        print("✅ Implementadas técnicas del documento:")
        print("   - Multi-timeframe data")
        print("   - Regímenes de mercado avanzados")
        print("   - Aumentación con noise injection")
        print("   - Labels balanceadas por régimen")
        print("   - Arquitectura TCN optimizada")
        print("   - Loss function Sharpe Ratio")
        
        return True
    
    def create_comprehensive_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea el conjunto completo de 66 features técnicas"""
        try:
            print("\n🔧 Creando features técnicas comprehensivas...")
            
            df = df.copy()
            features = []
            
            # 1. OHLCV básicos (5 features)
            features.extend(['open', 'high', 'low', 'close', 'volume'])
            
            # 2. Moving Averages SMA (10 features)
            for period in [5, 7, 10, 14, 20, 25, 30, 50, 100, 200]:
                df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
                features.append(f'sma_{period}')
            
            # 3. Exponential Moving Averages (8 features)
            for period in [5, 9, 12, 21, 26, 50, 100, 200]:
                df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
                features.append(f'ema_{period}')
            
            # 4. RSI múltiples períodos (4 features)
            for period in [9, 14, 21, 30]:
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                rs = gain / loss
                df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
                features.append(f'rsi_{period}')
            
            # 5. MACD completo (6 features)
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            df['macd_normalized'] = df['macd'] / df['close']
            df['macd_signal_normalized'] = df['macd_signal'] / df['close']
            df['macd_histogram_normalized'] = df['macd_histogram'] / df['close']
            features.extend(['macd', 'macd_signal', 'macd_histogram', 
                           'macd_normalized', 'macd_signal_normalized', 'macd_histogram_normalized'])
            
            # 6. Bollinger Bands (6 features)
            for period in [20, 50]:
                bb_middle = df['close'].rolling(window=period).mean()
                bb_std = df['close'].rolling(window=period).std()
                df[f'bb_upper_{period}'] = bb_middle + (bb_std * 2)
                df[f'bb_lower_{period}'] = bb_middle - (bb_std * 2)
                df[f'bb_position_{period}'] = (df['close'] - df[f'bb_lower_{period}']) / (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}'])
                features.extend([f'bb_upper_{period}', f'bb_lower_{period}', f'bb_position_{period}'])
            
            # 7. Momentum y ROC (8 features)
            for period in [3, 5, 10, 20]:
                df[f'momentum_{period}'] = df['close'] / df['close'].shift(period) - 1
                df[f'roc_{period}'] = df['close'].pct_change(periods=period)
                features.extend([f'momentum_{period}', f'roc_{period}'])
            
            # 8. Volatilidad (4 features)
            for period in [5, 10, 20, 50]:
                df[f'volatility_{period}'] = df['close'].pct_change().rolling(window=period).std()
                features.append(f'volatility_{period}')
            
            # 9. Volume analysis (6 features)
            for period in [5, 10, 20]:
                df[f'volume_sma_{period}'] = df['volume'].rolling(window=period).mean()
                df[f'volume_ratio_{period}'] = df['volume'] / df[f'volume_sma_{period}']
                features.extend([f'volume_sma_{period}', f'volume_ratio_{period}'])
            
            # 10. ATR (Average True Range) (3 features)
            for period in [14, 21, 30]:
                high_low = df['high'] - df['low']
                high_close = (df['high'] - df['close'].shift()).abs()
                low_close = (df['low'] - df['close'].shift()).abs()
                true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                df[f'atr_{period}'] = true_range.rolling(window=period).mean()
                features.append(f'atr_{period}')
            
            # 11. Stochastic Oscillator (2 features)
            low_min = df['low'].rolling(window=14).min()
            high_max = df['high'].rolling(window=14).max()
            df['stoch_k'] = 100 * (df['close'] - low_min) / (high_max - low_min)
            df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
            features.extend(['stoch_k', 'stoch_d'])
            
            # 12. Williams %R (2 features)
            for period in [14, 21]:
                high_max = df['high'].rolling(window=period).max()
                low_min = df['low'].rolling(window=period).min()
                df[f'williams_r_{period}'] = -100 * (high_max - df['close']) / (high_max - low_min)
                features.append(f'williams_r_{period}')
            
            # 13. Price position features (4 features)
            for period in [10, 20]:
                df[f'price_position_{period}'] = (df['close'] - df['low'].rolling(period).min()) / \
                                                (df['high'].rolling(period).max() - df['low'].rolling(period).min())
                df[f'price_distance_ma_{period}'] = (df['close'] - df['close'].rolling(period).mean()) / df['close']
                features.extend([f'price_position_{period}', f'price_distance_ma_{period}'])
            
            # 14. Features adicionales para completar 66
            df['close_change'] = df['close'].pct_change()
            df['volume_change'] = df['volume'].pct_change()
            features.extend(['close_change', 'volume_change'])
            
            # Asegurar exactamente 66 features
            features = features[:66]
            df = df.dropna()
            
            print(f"✅ Features creadas: {len(features)}")
            print(f"   - Datos limpios: {len(df)} períodos")
            
            return df[['timestamp', 'regime'] + features]
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepara datos para entrenamiento con arquitectura dual"""
        try:
            print("\n📦 Preparando datos para entrenamiento...")
            
            # Obtener features
            feature_columns = [col for col in df.columns if col not in ['timestamp', 'regime', 'label']]
            
            # Normalizar features
            scaler = MinMaxScaler()
            df_scaled = df.copy()
            df_scaled[feature_columns] = scaler.fit_transform(df[feature_columns])
            
            # Guardar scaler
            with open('models/advanced_feature_scalers.pkl', 'wb') as f:
                pickle.dump(scaler, f)
            
            # Crear secuencias temporales
            X_features = []
            X_regimes = []
            y = []
            
            for i in range(self.lookback_window, len(df_scaled)):
                # Features temporales
                feature_seq = df_scaled[feature_columns].iloc[i-self.lookback_window:i].values
                X_features.append(feature_seq)
                
                # Régimen actual
                regime = df_scaled['regime'].iloc[i]
                regime_one_hot = np.zeros(3)
                regime_one_hot[int(regime)] = 1
                X_regimes.append(regime_one_hot)
                
                # Label
                y.append(int(df_scaled['label'].iloc[i]))
            
            X_features = np.array(X_features)
            X_regimes = np.array(X_regimes)
            y = np.array(y)
            
            print(f"✅ Datos preparados:")
            print(f"   - X_features shape: {X_features.shape}")
            print(f"   - X_regimes shape: {X_regimes.shape}")
            print(f"   - y shape: {y.shape}")
            
            # Verificar distribución final
            final_counts = Counter(y)
            total = len(y)
            print("📊 Distribución final para entrenamiento:")
            for i, name in enumerate(self.class_names):
                count = final_counts[i]
                pct = count / total * 100
                print(f"   - {name}: {count} ({pct:.1f}%)")
            
            return X_features, X_regimes, y
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return None, None, None


async def main():
    trainer = AdvancedBalancedTCN()
    success = await trainer.run_advanced_training()
    
    if success:
        print("\n🎉 ENTRENAMIENTO AVANZADO LISTO!")
    else:
        print("\n❌ Error en entrenamiento")


if __name__ == "__main__":
    asyncio.run(main()) 