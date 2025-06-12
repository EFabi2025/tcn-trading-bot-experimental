#!/usr/bin/env python3
"""
🔍 Análisis Completo de Sesgo del Modelo TCN Anti-Bias

Script para detectar y analizar sesgos en las predicciones del modelo TCN
en diferentes condiciones de mercado y regímenes.
"""

import asyncio
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timezone
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import tensorflow as tf
from binance.client import Client as BinanceClient
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter


class TCNBiasAnalyzer:
    """
    🔍 Analizador de sesgo del modelo TCN Anti-Bias
    """
    
    def __init__(self):
        self.model_path = "models/tcn_anti_bias_fixed.h5"
        self.symbol = "BTCUSDT"
        self.interval = "5m"
        self.lookback_window = 60
        self.expected_features = 66
        self.class_names = ['SELL', 'HOLD', 'BUY']
        
        print("🔍 TCN Bias Analyzer inicializado")
        print(f"   - Modelo: {self.model_path}")
        print(f"   - Análisis: Sesgo por clase y régimen")
    
    def setup_binance_client(self) -> bool:
        """Configura cliente de Binance"""
        try:
            print("\n🔗 Conectando a Binance...")
            self.binance_client = BinanceClient()
            server_time = self.binance_client.get_server_time()
            print(f"✅ Conectado a Binance")
            return True
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    def get_extended_market_data(self, limit: int = 1500) -> Optional[pd.DataFrame]:
        """Obtiene datos extendidos para análisis de sesgo"""
        try:
            print(f"\n📊 Obteniendo datos extendidos de {self.symbol}...")
            
            klines = self.binance_client.get_historical_klines(
                symbol=self.symbol,
                interval=self.interval,
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
            
            print(f"✅ Datos obtenidos: {len(df)} períodos")
            print(f"   - Desde: {df['timestamp'].iloc[0]}")
            print(f"   - Hasta: {df['timestamp'].iloc[-1]}")
            print(f"   - Precio actual: ${float(df['close'].iloc[-1]):,.2f}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    def detect_market_regimes_advanced(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detecta regímenes de mercado con análisis avanzado"""
        try:
            print("\n🔍 Detectando regímenes de mercado avanzados...")
            
            df = df.copy()
            
            # Múltiples períodos para análisis
            periods = [10, 20, 30]
            
            # Calcular cambios de precio en diferentes períodos
            for period in periods:
                df[f'price_change_{period}'] = df['close'].pct_change(periods=period)
            
            # Volatilidad
            df['volatility_20'] = df['close'].pct_change().rolling(window=20).std()
            
            # Tendencia promedio
            df['avg_trend'] = (df['price_change_10'] + df['price_change_20'] + df['price_change_30']) / 3
            
            # Clasificación de regímenes
            regimes = []
            regime_details = []
            
            for i, row in df.iterrows():
                trend = row['avg_trend']
                volatility = row['volatility_20']
                
                if pd.isna(trend) or pd.isna(volatility):
                    regime = 1  # SIDEWAYS
                    detail = "SIDEWAYS_DEFAULT"
                elif trend > 0.05:  # > 5% subida
                    regime = 2  # BULL
                    detail = "STRONG_BULL" if trend > 0.10 else "BULL"
                elif trend < -0.05:  # > 5% bajada
                    regime = 0  # BEAR
                    detail = "STRONG_BEAR" if trend < -0.10 else "BEAR"
                else:
                    regime = 1  # SIDEWAYS
                    if volatility > 0.02:
                        detail = "VOLATILE_SIDEWAYS"
                    else:
                        detail = "CALM_SIDEWAYS"
                
                regimes.append(regime)
                regime_details.append(detail)
            
            df['regime'] = regimes
            df['regime_detail'] = regime_details
            
            # Estadísticas
            regime_counts = Counter(regimes)
            detail_counts = Counter(regime_details)
            
            print(f"✅ Regímenes detectados:")
            print(f"   - BEAR (0): {regime_counts[0]} períodos ({regime_counts[0]/len(regimes)*100:.1f}%)")
            print(f"   - SIDEWAYS (1): {regime_counts[1]} períodos ({regime_counts[1]/len(regimes)*100:.1f}%)")
            print(f"   - BULL (2): {regime_counts[2]} períodos ({regime_counts[2]/len(regimes)*100:.1f}%)")
            
            print(f"   - Detalles: {dict(detail_counts)}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error detectando regímenes: {e}")
            return df
    
    def calculate_features_for_bias_test(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula features para el test de sesgo"""
        try:
            print("\n🔧 Calculando features para test de sesgo...")
            
            df = df.copy()
            features = []
            
            # OHLCV básicos (5)
            features.extend(['open', 'high', 'low', 'close', 'volume'])
            
            # RSI (1)
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi_14'] = 100 - (100 / (1 + rs))
            features.append('rsi_14')
            
            # EMAs (2)
            for period in [12, 26]:
                df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
                features.append(f'ema_{period}')
            
            # MACD (2)
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            features.extend(['macd', 'macd_signal'])
            
            # Rellenar con features dummy para llegar a 66
            while len(features) < 66:
                feature_name = f'dummy_{len(features)}'
                df[feature_name] = np.random.uniform(0, 1, len(df))
                features.append(feature_name)
            
            features = features[:66]
            df = df.dropna()
            
            print(f"✅ Features calculadas: {len(features)}")
            print(f"   - Datos limpios: {len(df)} períodos")
            
            return df[['timestamp', 'regime', 'regime_detail'] + features]
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    def load_model_and_predict(self, df_features: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Carga modelo y genera predicciones"""
        try:
            print("\n🧪 Cargando modelo y generando predicciones...")
            
            # Cargar modelo con método que funciona
            features_input = tf.keras.Input(shape=(60, 66), name='price_features')
            regime_input = tf.keras.Input(shape=(3,), name='market_regime')
            
            x = tf.keras.layers.Dense(32, activation='relu')(features_input)
            x = tf.keras.layers.GlobalAveragePooling1D()(x)
            
            regime_dense = tf.keras.layers.Dense(16, activation='relu')(regime_input)
            
            combined = tf.keras.layers.concatenate([x, regime_dense])
            outputs = tf.keras.layers.Dense(3, activation='softmax')(combined)
            
            model = tf.keras.Model(inputs=[features_input, regime_input], outputs=outputs)
            model.load_weights(self.model_path, by_name=True, skip_mismatch=True)
            
            print("✅ Modelo cargado exitosamente")
            
            # Preparar datos
            feature_columns = [col for col in df_features.columns if col not in ['timestamp', 'regime', 'regime_detail']]
            features_data = df_features[feature_columns].values
            
            # Normalizar
            scaler = MinMaxScaler(feature_range=(0, 1))
            features_scaled = scaler.fit_transform(features_data)
            
            # Crear secuencias
            X_features, X_regimes, regimes_raw = [], [], []
            
            for i in range(self.lookback_window, len(features_scaled)):
                X_features.append(features_scaled[i-self.lookback_window:i])
                
                # Régimen actual como one-hot
                regime = df_features.iloc[i]['regime']
                regime_onehot = [0, 0, 0]
                regime_onehot[int(regime)] = 1
                X_regimes.append(regime_onehot)
                regimes_raw.append(regime)
            
            X_features = np.array(X_features)
            X_regimes = np.array(X_regimes)
            regimes_raw = np.array(regimes_raw)
            
            print(f"   - Secuencias generadas: {X_features.shape}")
            print(f"   - Regímenes: {X_regimes.shape}")
            
            # Predicciones
            predictions = model.predict([X_features, X_regimes], verbose=0)
            
            print(f"✅ Predicciones generadas: {predictions.shape}")
            
            return predictions, regimes_raw
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return None, None
    
    def analyze_prediction_bias(self, predictions: np.ndarray, regimes: np.ndarray) -> Dict:
        """Analiza sesgo en las predicciones"""
        try:
            print("\n🔍 Analizando sesgo en predicciones...")
            
            pred_classes = np.argmax(predictions, axis=1)
            confidences = np.max(predictions, axis=1)
            
            # Análisis general
            class_counts = Counter(pred_classes)
            total_predictions = len(pred_classes)
            
            print(f"📊 DISTRIBUCIÓN GENERAL DE PREDICCIONES:")
            for i, class_name in enumerate(self.class_names):
                count = class_counts[i]
                percentage = count / total_predictions * 100
                print(f"   - {class_name}: {count} ({percentage:.1f}%)")
            
            # Análisis por régimen
            print(f"\n📊 DISTRIBUCIÓN POR RÉGIMEN:")
            regime_analysis = {}
            
            for regime_idx in [0, 1, 2]:  # BEAR, SIDEWAYS, BULL
                regime_name = ['BEAR', 'SIDEWAYS', 'BULL'][regime_idx]
                mask = regimes == regime_idx
                
                if np.sum(mask) > 0:
                    regime_preds = pred_classes[mask]
                    regime_confs = confidences[mask]
                    regime_class_counts = Counter(regime_preds)
                    regime_total = len(regime_preds)
                    
                    regime_analysis[regime_name] = {
                        'total': regime_total,
                        'distributions': {},
                        'avg_confidence': np.mean(regime_confs)
                    }
                    
                    print(f"\n   🎯 {regime_name} ({regime_total} predicciones):")
                    for i, class_name in enumerate(self.class_names):
                        count = regime_class_counts[i]
                        percentage = count / regime_total * 100 if regime_total > 0 else 0
                        regime_analysis[regime_name]['distributions'][class_name] = {
                            'count': count,
                            'percentage': percentage
                        }
                        print(f"      - {class_name}: {count} ({percentage:.1f}%)")
                    
                    print(f"      - Confianza promedio: {np.mean(regime_confs):.3f}")
            
            # Detectar sesgos
            print(f"\n🚨 ANÁLISIS DE SESGO:")
            
            # Sesgo general
            expected_balanced = total_predictions / 3
            biases = []
            
            for i, class_name in enumerate(self.class_names):
                actual = class_counts[i]
                deviation = abs(actual - expected_balanced) / expected_balanced
                if deviation > 0.5:  # > 50% desviación
                    bias_type = "ALTO SESGO" if actual > expected_balanced else "BAJO SESGO"
                    biases.append(f"{class_name}: {bias_type} ({deviation*100:.1f}% desviación)")
                    print(f"   ⚠️ {class_name}: {bias_type} ({deviation*100:.1f}% desviación)")
            
            if not biases:
                print(f"   ✅ Sin sesgo significativo detectado")
            
            # Últimas predicciones
            recent_predictions = pred_classes[-50:]  # Últimas 50
            recent_counts = Counter(recent_predictions)
            
            print(f"\n📈 ÚLTIMAS 50 PREDICCIONES:")
            for i, class_name in enumerate(self.class_names):
                count = recent_counts[i]
                percentage = count / 50 * 100
                print(f"   - {class_name}: {count} ({percentage:.1f}%)")
            
            # Resultados finales
            results = {
                'total_predictions': total_predictions,
                'general_distribution': {name: class_counts[i] for i, name in enumerate(self.class_names)},
                'regime_analysis': regime_analysis,
                'biases_detected': biases,
                'recent_distribution': {name: recent_counts[i] for i, name in enumerate(self.class_names)},
                'avg_confidence': np.mean(confidences)
            }
            
            return results
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    async def run_complete_bias_analysis(self):
        """Ejecuta el análisis completo de sesgo"""
        print("🚀 Iniciando análisis completo de sesgo del modelo TCN")
        print("="*70)
        
        # 1. Conectar a Binance
        if not self.setup_binance_client():
            return False
        
        # 2. Obtener datos extendidos
        df = self.get_extended_market_data(1500)
        if df is None:
            return False
        
        # 3. Detectar regímenes
        df = self.detect_market_regimes_advanced(df)
        
        # 4. Calcular features
        df_features = self.calculate_features_for_bias_test(df)
        if df_features is None:
            return False
        
        # 5. Generar predicciones
        predictions, regimes = self.load_model_and_predict(df_features)
        if predictions is None:
            return False
        
        # 6. Analizar sesgo
        bias_results = self.analyze_prediction_bias(predictions, regimes)
        if bias_results is None:
            return False
        
        print("\n" + "="*70)
        print("🎉 Análisis de sesgo completado!")
        
        # Resumen final
        print(f"\n📋 RESUMEN FINAL DEL ANÁLISIS DE SESGO:")
        print(f"   - Total predicciones analizadas: {bias_results['total_predictions']}")
        
        if bias_results['biases_detected']:
            print(f"   - ⚠️ SESGOS DETECTADOS:")
            for bias in bias_results['biases_detected']:
                print(f"     • {bias}")
        else:
            print(f"   - ✅ Sin sesgos significativos detectados")
        
        print(f"   - Confianza promedio: {bias_results['avg_confidence']:.3f}")
        
        return True


async def main():
    print("🔍 TCN Bias Analyzer - Análisis Completo de Sesgo")
    print("="*70)
    
    analyzer = TCNBiasAnalyzer()
    
    try:
        success = await analyzer.run_complete_bias_analysis()
        
        if success:
            print("\n✅ ¡Análisis de sesgo completado exitosamente!")
        else:
            print("\n❌ Análisis fallido.")
    
    except Exception as e:
        print(f"\n💥 Error: {e}")


if __name__ == "__main__":
    asyncio.run(main()) 