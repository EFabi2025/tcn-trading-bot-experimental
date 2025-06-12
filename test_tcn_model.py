#!/usr/bin/env python3
"""
🧪 Test del Modelo TCN con Datos Reales de Binance

Script para validar que el modelo TCN está prediciendo correctamente
usando datos reales de mercado de Binance.

Pruebas incluidas:
1. Carga del modelo TCN
2. Obtención de datos reales de Binance
3. Feature engineering de 14 indicadores técnicos  
4. Predicciones del modelo
5. Evaluación de calidad de predicciones
6. Visualización de resultados
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
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# Configurar el path para imports
sys.path.append(str(Path(__file__).parent))

# Configuración básica
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class TCNModelTester:
    """
    🧪 Tester experimental del modelo TCN
    
    Valida el rendimiento del modelo con datos reales de Binance
    """
    
    def __init__(self):
        self.model: Optional[tf.keras.Model] = None
        self.binance_client: Optional[BinanceClient] = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
        # Configuración del modelo
        self.model_path = "models/tcn_anti_bias_fixed.h5"
        self.symbol = "BTCUSDT"
        self.interval = "5m"
        self.lookback_window = 60  # Ventana de observación
        self.prediction_steps = 12  # Predicción a 12 períodos (1 hora)
        
        print("🧪 TCN Model Tester inicializado")
    
    def load_model(self) -> bool:
        """Carga el modelo TCN entrenado"""
        try:
            if not Path(self.model_path).exists():
                print(f"❌ Modelo no encontrado en: {self.model_path}")
                return False
            
            print(f"📥 Cargando modelo TCN desde: {self.model_path}")
            self.model = tf.keras.models.load_model(self.model_path)
            
            print(f"✅ Modelo cargado exitosamente")
            print(f"   - Parámetros: {self.model.count_params():,}")
            print(f"   - Input shape: {self.model.input_shape}")
            print(f"   - Output shape: {self.model.output_shape}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
            return False
    
    def setup_binance_client(self) -> bool:
        """Configura cliente de Binance (sin API keys para datos públicos)"""
        try:
            print("🔗 Configurando cliente Binance para datos públicos...")
            # Usar cliente sin API keys para datos públicos
            self.binance_client = BinanceClient()
            
            # Test de conectividad
            server_time = self.binance_client.get_server_time()
            print(f"✅ Conectado a Binance")
            print(f"   - Servidor: {datetime.fromtimestamp(server_time['serverTime']/1000)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error conectando a Binance: {e}")
            return False
    
    def get_market_data(self, limit: int = 1000) -> Optional[pd.DataFrame]:
        """Obtiene datos reales de mercado de Binance"""
        try:
            print(f"📊 Obteniendo datos de {self.symbol} ({self.interval})...")
            
            # Obtener klines históricas
            klines = self.binance_client.get_historical_klines(
                symbol=self.symbol,
                interval=self.interval,
                limit=limit
            )
            
            if not klines:
                print("❌ No se obtuvieron datos de Binance")
                return None
            
            # Convertir a DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Limpiar y convertir tipos
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df = df.dropna()
            
            print(f"✅ Datos obtenidos: {len(df)} períodos")
            print(f"   - Desde: {df['timestamp'].min()}")
            print(f"   - Hasta: {df['timestamp'].max()}")
            print(f"   - Precio actual: ${float(df['close'].iloc[-1]):,.2f}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error obteniendo datos de mercado: {e}")
            return None
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula los 14 indicadores técnicos para el modelo"""
        try:
            print("🔧 Calculando indicadores técnicos...")
            
            # Asegurar que tenemos las columnas necesarias
            df = df.copy()
            
            # 1. Moving Averages
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['ema_12'] = df['close'].ewm(span=12).mean()
            df['ema_26'] = df['close'].ewm(span=26).mean()
            
            # 2. RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # 3. MACD
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # 4. Bollinger Bands
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            bb_std = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            
            # 5. Volume indicators
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # 6. Price momentum
            df['momentum'] = df['close'] / df['close'].shift(10) - 1
            df['roc'] = df['close'].pct_change(periods=10)
            
            # 7. Volatility
            df['volatility'] = df['close'].pct_change().rolling(window=20).std()
            
            # Seleccionar las 14 features principales
            feature_columns = [
                'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal',
                'macd_histogram', 'bb_upper', 'bb_lower', 'bb_width',
                'volume_ratio', 'momentum', 'roc', 'volatility'
            ]
            
            # Limpiar NaN
            df = df.dropna()
            
            print(f"✅ Indicadores calculados: {len(feature_columns)} features")
            print(f"   - Datos limpios: {len(df)} períodos")
            
            return df[['timestamp', 'close'] + feature_columns]
            
        except Exception as e:
            print(f"❌ Error calculando indicadores: {e}")
            return None
    
    def prepare_data_for_model(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepara datos para el modelo TCN"""
        try:
            print("🔄 Preparando datos para el modelo...")
            
            # Extraer features y target
            feature_columns = [col for col in df.columns if col not in ['timestamp', 'close']]
            features = df[feature_columns].values
            prices = df['close'].values
            
            # Normalizar features
            features_scaled = self.scaler.fit_transform(features)
            
            # Crear secuencias para el modelo
            X, y = [], []
            
            for i in range(self.lookback_window, len(features_scaled) - self.prediction_steps):
                # Ventana de features
                X.append(features_scaled[i-self.lookback_window:i])
                # Precio futuro a predecir
                y.append(prices[i + self.prediction_steps])
            
            X = np.array(X)
            y = np.array(y)
            
            print(f"✅ Datos preparados:")
            print(f"   - X shape: {X.shape}")
            print(f"   - y shape: {y.shape}")
            print(f"   - Features: {len(feature_columns)}")
            print(f"   - Lookback: {self.lookback_window} períodos")
            print(f"   - Prediction horizon: {self.prediction_steps} períodos")
            
            return X, y, prices
            
        except Exception as e:
            print(f"❌ Error preparando datos: {e}")
            return None, None, None
    
    def make_predictions(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Hace predicciones con el modelo TCN"""
        try:
            print("🔮 Haciendo predicciones con modelo TCN...")
            
            # Dividir en train/test (últimos 20% para test)
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            print(f"   - Train samples: {len(X_train)}")
            print(f"   - Test samples: {len(X_test)}")
            
            # Predicciones
            y_pred_train = self.model.predict(X_train, verbose=0)
            y_pred_test = self.model.predict(X_test, verbose=0)
            
            # Flatten predictions si es necesario
            if y_pred_train.ndim > 1:
                y_pred_train = y_pred_train.flatten()
            if y_pred_test.ndim > 1:
                y_pred_test = y_pred_test.flatten()
            
            results = {
                'y_train': y_train,
                'y_test': y_test,
                'y_pred_train': y_pred_train,
                'y_pred_test': y_pred_test,
                'split_idx': split_idx
            }
            
            print("✅ Predicciones completadas")
            
            return results
            
        except Exception as e:
            print(f"❌ Error haciendo predicciones: {e}")
            return None
    
    def evaluate_predictions(self, results: Dict) -> Dict:
        """Evalúa la calidad de las predicciones"""
        try:
            print("📈 Evaluando calidad de predicciones...")
            
            # Métricas para conjunto de entrenamiento
            train_mae = mean_absolute_error(results['y_train'], results['y_pred_train'])
            train_rmse = np.sqrt(mean_squared_error(results['y_train'], results['y_pred_train']))
            train_mape = np.mean(np.abs((results['y_train'] - results['y_pred_train']) / results['y_train'])) * 100
            
            # Métricas para conjunto de prueba
            test_mae = mean_absolute_error(results['y_test'], results['y_pred_test'])
            test_rmse = np.sqrt(mean_squared_error(results['y_test'], results['y_pred_test']))
            test_mape = np.mean(np.abs((results['y_test'] - results['y_pred_test']) / results['y_test'])) * 100
            
            # Correlación
            train_corr = np.corrcoef(results['y_train'], results['y_pred_train'])[0,1]
            test_corr = np.corrcoef(results['y_test'], results['y_pred_test'])[0,1]
            
            # Precisión direccional (predicción de subida/bajada)
            train_direction_acc = np.mean(
                np.sign(np.diff(results['y_train'][-len(results['y_pred_train'])+1:])) == 
                np.sign(np.diff(results['y_pred_train']))
            ) * 100
            
            test_direction_acc = np.mean(
                np.sign(np.diff(results['y_test'][-len(results['y_pred_test'])+1:])) == 
                np.sign(np.diff(results['y_pred_test']))
            ) * 100
            
            evaluation = {
                'train_mae': train_mae,
                'train_rmse': train_rmse,
                'train_mape': train_mape,
                'train_corr': train_corr,
                'train_direction_acc': train_direction_acc,
                'test_mae': test_mae,
                'test_rmse': test_rmse,
                'test_mape': test_mape,
                'test_corr': test_corr,
                'test_direction_acc': test_direction_acc
            }
            
            print("✅ Evaluación completada:")
            print(f"\n📊 MÉTRICAS DE ENTRENAMIENTO:")
            print(f"   - MAE: ${train_mae:.2f}")
            print(f"   - RMSE: ${train_rmse:.2f}")
            print(f"   - MAPE: {train_mape:.2f}%")
            print(f"   - Correlación: {train_corr:.4f}")
            print(f"   - Precisión direccional: {train_direction_acc:.2f}%")
            
            print(f"\n🎯 MÉTRICAS DE PRUEBA:")
            print(f"   - MAE: ${test_mae:.2f}")
            print(f"   - RMSE: ${test_rmse:.2f}")
            print(f"   - MAPE: {test_mape:.2f}%")
            print(f"   - Correlación: {test_corr:.4f}")
            print(f"   - Precisión direccional: {test_direction_acc:.2f}%")
            
            # Interpretación de resultados
            print(f"\n🔍 INTERPRETACIÓN:")
            if test_corr > 0.7:
                print("   ✅ Correlación EXCELENTE (>0.7)")
            elif test_corr > 0.5:
                print("   ✅ Correlación BUENA (>0.5)")
            elif test_corr > 0.3:
                print("   ⚠️ Correlación MODERADA (>0.3)")
            else:
                print("   ❌ Correlación BAJA (<0.3)")
            
            if test_direction_acc > 60:
                print("   ✅ Precisión direccional BUENA (>60%)")
            elif test_direction_acc > 55:
                print("   ⚠️ Precisión direccional MODERADA (>55%)")
            else:
                print("   ❌ Precisión direccional BAJA (<55%)")
            
            if test_mape < 2:
                print("   ✅ Error porcentual EXCELENTE (<2%)")
            elif test_mape < 5:
                print("   ✅ Error porcentual BUENO (<5%)")
            elif test_mape < 10:
                print("   ⚠️ Error porcentual MODERADO (<10%)")
            else:
                print("   ❌ Error porcentual ALTO (>10%)")
            
            return evaluation
            
        except Exception as e:
            print(f"❌ Error evaluando predicciones: {e}")
            return None
    
    def plot_results(self, df: pd.DataFrame, results: Dict, evaluation: Dict):
        """Visualiza los resultados"""
        try:
            print("📊 Generando visualizaciones...")
            
            # Crear figura con subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'🧪 Evaluación Modelo TCN - {self.symbol}', fontsize=16, fontweight='bold')
            
            # 1. Predicciones vs Realidad (Test set)
            ax1 = axes[0, 0]
            ax1.scatter(results['y_test'], results['y_pred_test'], alpha=0.6, color='blue')
            ax1.plot([results['y_test'].min(), results['y_test'].max()], 
                    [results['y_test'].min(), results['y_test'].max()], 'r--', lw=2)
            ax1.set_xlabel('Precio Real')
            ax1.set_ylabel('Precio Predicho')
            ax1.set_title(f'Predicciones vs Realidad\nCorr: {evaluation["test_corr"]:.3f}')
            ax1.grid(True, alpha=0.3)
            
            # 2. Serie temporal de predicciones
            ax2 = axes[0, 1]
            test_length = len(results['y_test'])
            x_test = range(test_length)
            
            ax2.plot(x_test, results['y_test'], label='Real', color='blue', linewidth=2)
            ax2.plot(x_test, results['y_pred_test'], label='Predicho', color='red', linewidth=2, alpha=0.8)
            ax2.set_xlabel('Tiempo')
            ax2.set_ylabel('Precio')
            ax2.set_title(f'Serie Temporal - Test Set\nMAE: ${evaluation["test_mae"]:.2f}')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # 3. Residuos
            ax3 = axes[1, 0]
            residuals = results['y_test'] - results['y_pred_test']
            ax3.scatter(results['y_pred_test'], residuals, alpha=0.6, color='green')
            ax3.axhline(y=0, color='red', linestyle='--')
            ax3.set_xlabel('Predicciones')
            ax3.set_ylabel('Residuos')
            ax3.set_title(f'Análisis de Residuos\nRMSE: ${evaluation["test_rmse"]:.2f}')
            ax3.grid(True, alpha=0.3)
            
            # 4. Histograma de errores
            ax4 = axes[1, 1]
            ax4.hist(residuals, bins=30, alpha=0.7, color='purple', edgecolor='black')
            ax4.axvline(x=0, color='red', linestyle='--', linewidth=2)
            ax4.set_xlabel('Error de Predicción')
            ax4.set_ylabel('Frecuencia')
            ax4.set_title(f'Distribución de Errores\nMAPE: {evaluation["test_mape"]:.2f}%')
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Guardar gráfico
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"tcn_evaluation_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"✅ Gráfico guardado: {filename}")
            
            plt.show()
            
        except Exception as e:
            print(f"❌ Error generando visualizaciones: {e}")
    
    async def run_complete_test(self):
        """Ejecuta la prueba completa del modelo TCN"""
        print("🚀 Iniciando prueba completa del modelo TCN")
        print("="*60)
        
        # 1. Cargar modelo
        if not self.load_model():
            return False
        
        # 2. Configurar Binance
        if not self.setup_binance_client():
            return False
        
        # 3. Obtener datos de mercado
        df = self.get_market_data(limit=1000)
        if df is None:
            return False
        
        # 4. Calcular indicadores técnicos
        df_features = self.calculate_technical_indicators(df)
        if df_features is None:
            return False
        
        # 5. Preparar datos para modelo
        X, y, prices = self.prepare_data_for_model(df_features)
        if X is None:
            return False
        
        # 6. Hacer predicciones
        results = self.make_predictions(X, y)
        if results is None:
            return False
        
        # 7. Evaluar predicciones
        evaluation = self.evaluate_predictions(results)
        if evaluation is None:
            return False
        
        # 8. Visualizar resultados
        self.plot_results(df_features, results, evaluation)
        
        print("\n" + "="*60)
        print("🎉 Prueba del modelo TCN completada exitosamente!")
        
        # Resumen final
        print(f"\n📋 RESUMEN FINAL:")
        print(f"   - Modelo: {self.model_path}")
        print(f"   - Símbolo: {self.symbol}")
        print(f"   - Datos: {len(df)} períodos de {self.interval}")
        print(f"   - Features: 14 indicadores técnicos")
        print(f"   - Correlación test: {evaluation['test_corr']:.3f}")
        print(f"   - Precisión direccional: {evaluation['test_direction_acc']:.1f}%")
        print(f"   - Error promedio: ${evaluation['test_mae']:.2f}")
        
        return True


async def main():
    """Función principal"""
    print("🧪 TCN Model Tester - Validación con Datos Reales de Binance")
    print("="*60)
    
    tester = TCNModelTester()
    
    try:
        success = await tester.run_complete_test()
        
        if success:
            print("\n✅ Todas las pruebas completadas exitosamente!")
            print("📊 Revisa los gráficos generados para analizar el rendimiento")
        else:
            print("\n❌ Las pruebas fallaron. Revisa los errores anteriores.")
    
    except KeyboardInterrupt:
        print("\n⚠️ Pruebas interrumpidas por el usuario")
    except Exception as e:
        print(f"\n💥 Error crítico: {e}")


if __name__ == "__main__":
    asyncio.run(main()) 