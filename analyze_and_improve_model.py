#!/usr/bin/env python3
"""
🔍 Análisis y Mejora del Modelo TCN Re-entrenado

Script para analizar el modelo re-entrenado y corregir problemas:
- Análisis profundo de sesgo
- Evaluación de calidad de datos
- Recomendaciones de mejora
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from binance.client import Client as BinanceClient
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import pickle
import warnings
warnings.filterwarnings('ignore')


class ModelAnalyzer:
    """
    🔍 Analizador del modelo TCN re-entrenado
    """
    
    def __init__(self):
        self.model_path = "models/tcn_anti_bias_retrained.h5"
        self.scaler_path = "models/feature_scalers_retrained.pkl"
        self.class_names = ['SELL', 'HOLD', 'BUY']
        self.symbol = "BTCUSDT"
        self.interval = "5m"
        self.lookback_window = 60
        
        print("🔍 Analizador del Modelo TCN inicializado")
    
    def load_model_and_scaler(self) -> bool:
        """Carga el modelo y scaler re-entrenados"""
        try:
            print("\n📥 Cargando modelo y artefactos...")
            
            # Cargar modelo
            self.model = tf.keras.models.load_model(self.model_path)
            print(f"✅ Modelo cargado: {self.model_path}")
            
            # Cargar scaler
            with open(self.scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            print(f"✅ Scaler cargado: {self.scaler_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error cargando: {e}")
            return False
    
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
    
    def collect_fresh_test_data(self, limit: int = 500) -> pd.DataFrame:
        """Recolecta datos frescos para testing"""
        try:
            print(f"\n📊 Recolectando datos frescos para testing...")
            
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
            
            print(f"✅ Datos de test: {len(df)} períodos")
            return df
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    def create_features_for_testing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea features para testing (mismo proceso que entrenamiento)"""
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
            
            # 14. Features adicionales para completar 66
            df['close_change'] = df['close'].pct_change()
            df['volume_change'] = df['volume'].pct_change()
            features.extend(['close_change', 'volume_change'])
            
            # Detectar regímenes
            df['trend_medium'] = df['close'].pct_change(periods=20)
            df['momentum'] = df['close'] / df['close'].shift(15) - 1
            
            regimes = []
            for i, row in df.iterrows():
                trend_medium = row['trend_medium'] 
                momentum = row['momentum']
                
                if pd.isna(trend_medium):
                    regime = 1  # SIDEWAYS
                else:
                    if trend_medium > 0.02 or momentum > 0.03:
                        regime = 2  # BULL
                    elif trend_medium < -0.02 or momentum < -0.03:
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
    
    def prepare_test_data(self, df: pd.DataFrame) -> tuple:
        """Prepara datos de test para predicción"""
        try:
            print("\n🔧 Preparando datos de test...")
            
            feature_columns = [col for col in df.columns if col not in ['timestamp', 'regime']]
            features_data = df[feature_columns].values
            regimes_data = df['regime'].values
            
            # Normalizar features con el scaler entrenado
            features_scaled = self.scaler.transform(features_data)
            
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
            
            return X_features, X_regimes, df.iloc[self.lookback_window:].reset_index(drop=True)
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return None, None, None
    
    def comprehensive_bias_analysis(self, X_features: np.ndarray, X_regimes: np.ndarray, df_context: pd.DataFrame):
        """Análisis comprehensivo de sesgo del modelo"""
        try:
            print("\n🧪 ANÁLISIS COMPREHENSIVO DE SESGO")
            print("="*60)
            
            # Hacer predicciones
            predictions = self.model.predict([X_features, X_regimes], verbose=0)
            pred_classes = np.argmax(predictions, axis=1)
            pred_confidences = np.max(predictions, axis=1)
            
            # 1. Análisis de distribución general
            print(f"\n📊 1. DISTRIBUCIÓN GENERAL DE PREDICCIONES:")
            pred_counts = Counter(pred_classes)
            total_preds = len(pred_classes)
            
            for i, name in enumerate(self.class_names):
                count = pred_counts[i]
                pct = count / total_preds * 100
                print(f"   - {name}: {count} ({pct:.1f}%)")
            
            # 2. Análisis por régimen de mercado
            print(f"\n📊 2. ANÁLISIS POR RÉGIMEN DE MERCADO:")
            regimes_decoded = np.argmax(X_regimes, axis=1)
            regime_names = ['BEAR', 'SIDEWAYS', 'BULL']
            
            for regime_idx in range(3):
                regime_mask = regimes_decoded == regime_idx
                regime_count = np.sum(regime_mask)
                
                if regime_count > 0:
                    regime_preds = pred_classes[regime_mask]
                    regime_pred_counts = Counter(regime_preds)
                    
                    print(f"\n   {regime_names[regime_idx]} Market ({regime_count} muestras):")
                    for i, name in enumerate(self.class_names):
                        count = regime_pred_counts[i]
                        pct = count / regime_count * 100 if regime_count > 0 else 0
                        print(f"     - {name}: {count} ({pct:.1f}%)")
            
            # 3. Análisis de confianza
            print(f"\n📊 3. ANÁLISIS DE CONFIANZA:")
            print(f"   - Confianza promedio: {pred_confidences.mean():.3f}")
            print(f"   - Confianza mínima: {pred_confidences.min():.3f}")
            print(f"   - Confianza máxima: {pred_confidences.max():.3f}")
            print(f"   - Desviación estándar: {pred_confidences.std():.3f}")
            
            # 4. Análisis temporal (últimas predicciones)
            print(f"\n📊 4. ANÁLISIS TEMPORAL (Últimas 20 predicciones):")
            recent_preds = pred_classes[-20:]
            recent_counts = Counter(recent_preds)
            
            for i, name in enumerate(self.class_names):
                count = recent_counts[i]
                pct = count / len(recent_preds) * 100
                print(f"   - {name}: {count} ({pct:.1f}%)")
            
            # 5. Detección de SESGO CRÍTICO
            print(f"\n🚨 5. DETECCIÓN DE SESGO CRÍTICO:")
            bias_issues = []
            
            # Sesgo de clase dominante
            max_class_pct = max([count / total_preds * 100 for count in pred_counts.values()])
            if max_class_pct > 80:
                bias_issues.append(f"SESGO EXTREMO: Una clase domina {max_class_pct:.1f}% de predicciones")
            
            # Sesgo de clase ausente
            min_class_pct = min([count / total_preds * 100 for count in pred_counts.values()])
            if min_class_pct < 5:
                bias_issues.append(f"CLASE AUSENTE: Una clase representa solo {min_class_pct:.1f}% de predicciones")
            
            # Sesgo temporal
            recent_max_pct = max([count / len(recent_preds) * 100 for count in recent_counts.values()])
            if recent_max_pct > 90:
                bias_issues.append(f"SESGO TEMPORAL: {recent_max_pct:.1f}% de predicciones recientes son iguales")
            
            # Resultado del análisis de sesgo
            if bias_issues:
                print(f"❌ SESGO DETECTADO:")
                for issue in bias_issues:
                    print(f"   - {issue}")
                
                print(f"\n🚨 RECOMENDACIONES:")
                print(f"   - Re-entrenar con datos más balanceados")
                print(f"   - Ajustar umbrales de clasificación") 
                print(f"   - Implementar técnicas de augmentación de datos")
                print(f"   - Revisar arquitectura del modelo")
                
                return False
            else:
                print(f"✅ NO SE DETECTÓ SESGO CRÍTICO")
                print(f"   - Distribución de clases aceptable")
                print(f"   - Comportamiento temporal estable") 
                return True
            
        except Exception as e:
            print(f"❌ Error en análisis: {e}")
            return False
    
    def market_condition_analysis(self, df_context: pd.DataFrame):
        """Analiza las condiciones actuales del mercado"""
        try:
            print(f"\n📈 ANÁLISIS DE CONDICIONES DE MERCADO:")
            
            # Análisis de precio
            current_price = df_context['close'].iloc[-1]
            price_24h_ago = df_context['close'].iloc[-288] if len(df_context) >= 288 else df_context['close'].iloc[0]  # 24h = 288 períodos de 5min
            price_change_24h = (current_price - price_24h_ago) / price_24h_ago * 100
            
            print(f"   - Precio actual: ${current_price:,.2f}")
            print(f"   - Cambio 24h: {price_change_24h:+.2f}%")
            
            # Análisis de volatilidad
            recent_returns = df_context['close'].pct_change().tail(100)
            volatility = recent_returns.std() * np.sqrt(288) * 100  # Anualizada
            
            print(f"   - Volatilidad (100 períodos): {volatility:.2f}%")
            
            # Análisis de régimen
            recent_regimes = df_context['regime'].tail(50)
            regime_counts = Counter(recent_regimes)
            regime_names = ['BEAR', 'SIDEWAYS', 'BULL']
            
            print(f"   - Régimen dominante (50 períodos):")
            for i, name in enumerate(regime_names):
                count = regime_counts[i]
                pct = count / len(recent_regimes) * 100
                print(f"     - {name}: {count} ({pct:.1f}%)")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def generate_improvement_recommendations(self):
        """Genera recomendaciones de mejora"""
        print(f"\n💡 RECOMENDACIONES DE MEJORA:")
        print(f"="*60)
        
        print(f"1. 📊 DATOS DE ENTRENAMIENTO:")
        print(f"   - Recolectar más datos históricos (>2000 períodos)")
        print(f"   - Incluir diferentes condiciones de mercado")
        print(f"   - Balancear regímenes artificialmente si es necesario")
        
        print(f"\n2. 🏗️ ARQUITECTURA DEL MODELO:")
        print(f"   - Experimentar con diferentes kernel sizes en TCN")
        print(f"   - Ajustar dilation rates para capturar patrones temporales")
        print(f"   - Considerar arquitecturas híbridas (TCN + LSTM)")
        
        print(f"\n3. ⚖️ TÉCNICAS ANTI-SESGO:")
        print(f"   - Implementar focal loss para clases desbalanceadas")
        print(f"   - Usar técnicas de oversampling/undersampling")
        print(f"   - Validación cruzada estratificada por régimen")
        
        print(f"\n4. 🎯 OPTIMIZACIÓN:")
        print(f"   - Hyperparameter tuning sistemático")
        print(f"   - Ensembles de modelos")
        print(f"   - Regularización adicional")
    
    def run_complete_analysis(self):
        """Ejecuta análisis completo del modelo"""
        print("🔍 ANÁLISIS COMPLETO DEL MODELO TCN RE-ENTRENADO")
        print("="*80)
        
        # 1. Cargar modelo y scaler
        if not self.load_model_and_scaler():
            return False
        
        # 2. Conectar a Binance
        if not self.setup_binance_client():
            return False
        
        # 3. Recolectar datos frescos
        df = self.collect_fresh_test_data(500)
        if df is None:
            return False
        
        # 4. Crear features
        df_features = self.create_features_for_testing(df)
        if df_features is None:
            return False
        
        # 5. Preparar datos de test
        X_features, X_regimes, df_context = self.prepare_test_data(df_features)
        if X_features is None:
            return False
        
        # 6. Análisis comprehensivo de sesgo
        is_bias_free = self.comprehensive_bias_analysis(X_features, X_regimes, df_context)
        
        # 7. Análisis de condiciones de mercado
        self.market_condition_analysis(df_context)
        
        # 8. Recomendaciones
        self.generate_improvement_recommendations()
        
        print(f"\n" + "="*80)
        if is_bias_free:
            print("✅ MODELO APROBADO - Sin sesgo crítico detectado")
        else:
            print("❌ MODELO REQUIERE MEJORAS - Sesgo crítico detectado")
        
        return is_bias_free


def main():
    print("🔍 Análisis del Modelo TCN Re-entrenado")
    print("="*80)
    
    analyzer = ModelAnalyzer()
    
    try:
        analyzer.run_complete_analysis()
        
    except Exception as e:
        print(f"\n💥 Error: {e}")


if __name__ == "__main__":
    main() 