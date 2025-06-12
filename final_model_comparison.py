#!/usr/bin/env python3
"""
🔍 Comparación Final de Modelos TCN Anti-Bias

Script para comparar el modelo re-entrenado final con el original
y evaluar las mejoras obtenidas en la reducción de sesgo.
"""

import asyncio
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import tensorflow as tf
from binance.client import Client as BinanceClient
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import pickle
import json


class ModelComparator:
    """
    🔍 Comparador de modelos TCN Anti-Bias
    """
    
    def __init__(self):
        self.symbol = "BTCUSDT"
        self.interval = "5m"
        self.lookback_window = 60
        self.expected_features = 66
        self.class_names = ['SELL', 'HOLD', 'BUY']
        
        # Rutas de modelos
        self.original_model_path = "models/tcn_model_original.h5"
        self.final_model_path = "models/tcn_anti_bias_final.h5"
        self.final_scaler_path = "models/feature_scalers_final.pkl"
        
        print("🔍 Model Comparator inicializado")
    
    def setup_binance_client(self) -> bool:
        """Configura cliente de Binance"""
        try:
            print("\n🔗 Conectando a Binance...")
            self.binance_client = BinanceClient()
            print(f"✅ Conectado a Binance")
            return True
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    def collect_test_data(self) -> Optional[pd.DataFrame]:
        """Recolecta datos frescos para testing"""
        try:
            print(f"\n📊 Recolectando datos frescos para testing...")
            
            klines = self.binance_client.get_historical_klines(
                symbol=self.symbol,
                interval=self.interval,
                limit=800
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
            
            print(f"✅ Datos de test recolectados: {len(df)} períodos")
            return df
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    def create_test_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea features para testing"""
        try:
            print("\n🔧 Creando features para testing...")
            
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
            
            # 14. Features adicionales
            df['close_change'] = df['close'].pct_change()
            df['volume_change'] = df['volume'].pct_change()
            features.extend(['close_change', 'volume_change'])
            
            # Market regime
            df['returns_medium'] = df['close'].pct_change(periods=20)
            returns_medium = df['returns_medium'].dropna()
            p25 = returns_medium.quantile(0.25)
            p75 = returns_medium.quantile(0.75)
            
            regimes = []
            for i, row in df.iterrows():
                ret_medium = row['returns_medium']
                if pd.isna(ret_medium):
                    regime = 1  # SIDEWAYS
                else:
                    if ret_medium >= p75:
                        regime = 2  # BULL
                    elif ret_medium <= p25:
                        regime = 0  # BEAR
                    else:
                        regime = 1  # SIDEWAYS
                regimes.append(regime)
            
            df['regime'] = regimes
            
            # Asegurar exactamente 66 features
            features = features[:66]
            df = df.dropna()
            
            print(f"✅ Features de test creadas: {len(features)}")
            return df[['timestamp', 'regime'] + features]
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    def prepare_test_data(self, df: pd.DataFrame, scaler: MinMaxScaler) -> Tuple[np.ndarray, np.ndarray]:
        """Prepara datos de test"""
        try:
            print("\n🔧 Preparando datos de test...")
            
            feature_columns = [col for col in df.columns if col not in ['timestamp', 'regime']]
            features_data = df[feature_columns].values
            regimes_data = df['regime'].values
            
            # Normalizar features con scaler ya entrenado
            features_scaled = scaler.transform(features_data)
            
            # Crear secuencias
            X_features, X_regimes = [], []
            
            for i in range(self.lookback_window, len(features_scaled)):
                # Secuencia de features
                X_features.append(features_scaled[i-self.lookback_window:i])
                
                # Régimen actual como one-hot
                regime = regimes_data[i]
                regime_onehot = [0, 0, 0]
                regime_onehot[int(regime)] = 1
                X_regimes.append(regime_onehot)
            
            X_features = np.array(X_features)
            X_regimes = np.array(X_regimes)
            
            print(f"✅ Datos de test preparados:")
            print(f"   - X_features: {X_features.shape}")
            print(f"   - X_regimes: {X_regimes.shape}")
            
            return X_features, X_regimes
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return None, None
    
    def analyze_model_bias(self, model_name: str, predictions: np.ndarray) -> Dict:
        """Analiza el sesgo de un modelo"""
        pred_classes = np.argmax(predictions, axis=1)
        pred_confidences = np.max(predictions, axis=1)
        
        # Distribución de clases
        pred_counts = Counter(pred_classes)
        total_preds = len(pred_classes)
        
        class_percentages = []
        for i in range(3):
            count = pred_counts[i]
            pct = count / total_preds * 100
            class_percentages.append(pct)
        
        # Métricas de sesgo
        max_class_pct = max(class_percentages)
        min_class_pct = min(class_percentages)
        gini_coefficient = sum(abs(p - 100/3) for p in class_percentages) / (2 * 100/3)
        
        # Diversidad temporal
        window_size = 50
        temporal_bias_windows = 0
        total_windows = 0
        
        for i in range(0, len(pred_classes) - window_size, 25):
            window_preds = pred_classes[i:i + window_size]
            window_counts = Counter(window_preds)
            max_window_pct = max(window_counts.values()) / len(window_preds) * 100
            
            if max_window_pct > 75:
                temporal_bias_windows += 1
            total_windows += 1
        
        temporal_bias_pct = temporal_bias_windows / total_windows * 100 if total_windows > 0 else 0
        
        return {
            'model_name': model_name,
            'distribution': class_percentages,
            'max_class_pct': max_class_pct,
            'min_class_pct': min_class_pct,
            'gini_coefficient': gini_coefficient,
            'temporal_bias_pct': temporal_bias_pct,
            'avg_confidence': pred_confidences.mean(),
            'std_confidence': pred_confidences.std(),
            'total_predictions': total_preds
        }
    
    def generate_comparison_report(self, original_results: Dict, final_results: Dict):
        """Genera reporte de comparación"""
        print("\n" + "="*80)
        print("📊 REPORTE DE COMPARACIÓN DE MODELOS")
        print("="*80)
        
        print(f"\n📈 MODELO ORIGINAL:")
        print(f"   - Total predicciones: {original_results['total_predictions']}")
        print(f"   - Distribución:")
        for i, (name, pct) in enumerate(zip(self.class_names, original_results['distribution'])):
            print(f"     * {name}: {pct:.1f}%")
        print(f"   - Clase dominante: {original_results['max_class_pct']:.1f}%")
        print(f"   - Clase minoritaria: {original_results['min_class_pct']:.1f}%")
        print(f"   - Gini coefficient: {original_results['gini_coefficient']:.3f}")
        print(f"   - Sesgo temporal: {original_results['temporal_bias_pct']:.1f}%")
        print(f"   - Confianza promedio: {original_results['avg_confidence']:.3f}")
        
        print(f"\n🚀 MODELO FINAL RE-ENTRENADO:")
        print(f"   - Total predicciones: {final_results['total_predictions']}")
        print(f"   - Distribución:")
        for i, (name, pct) in enumerate(zip(self.class_names, final_results['distribution'])):
            print(f"     * {name}: {pct:.1f}%")
        print(f"   - Clase dominante: {final_results['max_class_pct']:.1f}%")
        print(f"   - Clase minoritaria: {final_results['min_class_pct']:.1f}%")
        print(f"   - Gini coefficient: {final_results['gini_coefficient']:.3f}")
        print(f"   - Sesgo temporal: {final_results['temporal_bias_pct']:.1f}%")
        print(f"   - Confianza promedio: {final_results['avg_confidence']:.3f}")
        
        print(f"\n📊 MEJORAS OBTENIDAS:")
        
        # Mejora en diversidad de clases
        diversity_improvement = original_results['gini_coefficient'] - final_results['gini_coefficient']
        print(f"   - Diversidad (Gini): {diversity_improvement:+.3f} {'✅' if diversity_improvement > 0 else '❌'}")
        
        # Mejora en clase dominante
        dominance_improvement = original_results['max_class_pct'] - final_results['max_class_pct']
        print(f"   - Reducción dominancia: {dominance_improvement:+.1f}pp {'✅' if dominance_improvement > 0 else '❌'}")
        
        # Mejora en clase minoritaria
        minority_improvement = final_results['min_class_pct'] - original_results['min_class_pct']
        print(f"   - Mejora minoritaria: {minority_improvement:+.1f}pp {'✅' if minority_improvement > 0 else '❌'}")
        
        # Mejora en sesgo temporal
        temporal_improvement = original_results['temporal_bias_pct'] - final_results['temporal_bias_pct']
        print(f"   - Reducción sesgo temporal: {temporal_improvement:+.1f}pp {'✅' if temporal_improvement > 0 else '❌'}")
        
        # Evaluación general
        improvements = sum([
            diversity_improvement > 0,
            dominance_improvement > 0,
            minority_improvement > 0,
            temporal_improvement > 0
        ])
        
        print(f"\n🎯 EVALUACIÓN GENERAL:")
        print(f"   - Mejoras obtenidas: {improvements}/4")
        
        if improvements >= 3:
            print(f"   ✅ RE-ENTRENAMIENTO EXITOSO - Mejoras significativas")
            return True
        elif improvements >= 2:
            print(f"   ⚠️ RE-ENTRENAMIENTO PARCIAL - Algunas mejoras")
            return True
        else:
            print(f"   ❌ RE-ENTRENAMIENTO INSUFICIENTE - Pocas mejoras")
            return False
    
    async def run_comparison(self):
        """Ejecuta la comparación completa"""
        print("🔍 Iniciando comparación de modelos TCN Anti-Bias")
        print("="*80)
        
        # 1. Configurar Binance
        if not self.setup_binance_client():
            return False
        
        # 2. Recolectar datos de test
        df = self.collect_test_data()
        if df is None:
            return False
        
        # 3. Crear features de test
        df_features = self.create_test_features(df)
        if df_features is None:
            return False
        
        # 4. Cargar scaler final
        try:
            print(f"\n📥 Cargando scaler final...")
            with open(self.final_scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            print(f"✅ Scaler cargado")
        except Exception as e:
            print(f"❌ Error cargando scaler: {e}")
            return False
        
        # 5. Preparar datos de test
        X_features, X_regimes = self.prepare_test_data(df_features, scaler)
        if X_features is None:
            return False
        
        # 6. Cargar y evaluar modelo original
        try:
            print(f"\n📥 Cargando modelo original...")
            original_model = tf.keras.models.load_model(self.original_model_path, compile=False)
            print(f"✅ Modelo original cargado")
            
            print(f"🔍 Evaluando modelo original...")
            original_predictions = original_model.predict([X_features, X_regimes], verbose=0)
            original_results = self.analyze_model_bias("Original", original_predictions)
            
        except Exception as e:
            print(f"❌ Error con modelo original: {e}")
            return False
        
        # 7. Cargar y evaluar modelo final
        try:
            print(f"\n📥 Cargando modelo final...")
            final_model = tf.keras.models.load_model(self.final_model_path, compile=False)
            print(f"✅ Modelo final cargado")
            
            print(f"🔍 Evaluando modelo final...")
            final_predictions = final_model.predict([X_features, X_regimes], verbose=0)
            final_results = self.analyze_model_bias("Final", final_predictions)
            
        except Exception as e:
            print(f"❌ Error con modelo final: {e}")
            return False
        
        # 8. Generar reporte de comparación
        success = self.generate_comparison_report(original_results, final_results)
        
        print("\n" + "="*80)
        if success:
            print("🎉 ¡COMPARACIÓN COMPLETADA EXITOSAMENTE!")
            print("✅ El re-entrenamiento ha mejorado el modelo")
        else:
            print("⚠️ COMPARACIÓN COMPLETADA CON LIMITACIONES")
            print("🔧 Se requieren mejoras adicionales")
        
        return success


async def main():
    print("🔍 Model Comparison Tool")
    print("="*80)
    
    comparator = ModelComparator()
    
    try:
        success = await comparator.run_comparison()
        
        if success:
            print("\n✅ ¡Comparación exitosa!")
            print("📈 Mejoras detectadas en el modelo re-entrenado")
        else:
            print("\n⚠️ Comparación completada con limitaciones.")
            print("🔧 Considerar mejoras adicionales")
    
    except Exception as e:
        print(f"\n💥 Error: {e}")


if __name__ == "__main__":
    asyncio.run(main()) 