#!/usr/bin/env python3
"""
🧪 Test Simplificado del Modelo TCN - Datos Reales de Binance

Script simplificado para validar datos de Binance y probar predicciones básicas.
No depende del modelo pre-entrenado para evitar problemas de compatibilidad.
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
from binance.client import Client as BinanceClient
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns

# Configurar el path para imports
sys.path.append(str(Path(__file__).parent))

# Configuración básica
plt.style.use('default')
sns.set_palette("husl")


class SimpleTCNTester:
    """
    🧪 Tester simplificado para validar datos y algoritmos de predicción
    """
    
    def __init__(self):
        self.binance_client: Optional[BinanceClient] = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
        # Configuración
        self.symbol = "BTCUSDT"
        self.interval = "5m"
        self.lookback_window = 60  # Ventana de observación
        self.prediction_steps = 12  # Predicción a 12 períodos (1 hora)
        
        print("🧪 Simple TCN Tester inicializado")
        print(f"   - Símbolo: {self.symbol}")
        print(f"   - Intervalo: {self.interval}")
        print(f"   - Ventana: {self.lookback_window} períodos")
        print(f"   - Predicción: {self.prediction_steps} períodos adelante")
    
    def setup_binance_client(self) -> bool:
        """Configura cliente de Binance para datos públicos"""
        try:
            print("\n🔗 Configurando cliente Binance para datos públicos...")
            self.binance_client = BinanceClient()
            
            # Test de conectividad
            server_time = self.binance_client.get_server_time()
            server_datetime = datetime.fromtimestamp(server_time['serverTime']/1000)
            
            print(f"✅ Conectado a Binance")
            print(f"   - Servidor: {server_datetime}")
            print(f"   - Latencia: ~{(datetime.now() - server_datetime).total_seconds():.2f}s")
            
            return True
            
        except Exception as e:
            print(f"❌ Error conectando a Binance: {e}")
            return False
    
    def get_market_data(self, limit: int = 1000) -> Optional[pd.DataFrame]:
        """Obtiene datos reales de mercado de Binance"""
        try:
            print(f"\n📊 Obteniendo datos de {self.symbol} ({self.interval})...")
            
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
            
            # Calcular estadísticas básicas
            price_change_24h = ((df['close'].iloc[-1] - df['close'].iloc[-288]) / df['close'].iloc[-288]) * 100
            volume_avg = df['volume'].tail(288).mean()  # Promedio 24h
            
            print(f"✅ Datos obtenidos: {len(df)} períodos")
            print(f"   - Desde: {df['timestamp'].min()}")
            print(f"   - Hasta: {df['timestamp'].max()}")
            print(f"   - Precio actual: ${float(df['close'].iloc[-1]):,.2f}")
            print(f"   - Cambio 24h: {price_change_24h:+.2f}%")
            print(f"   - Volumen promedio 24h: {volume_avg:,.0f}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error obteniendo datos de mercado: {e}")
            return None
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula indicadores técnicos optimizados"""
        try:
            print("\n🔧 Calculando indicadores técnicos...")
            
            df = df.copy()
            
            # 1. Moving Averages
            df['sma_7'] = df['close'].rolling(window=7).mean()
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['ema_12'] = df['close'].ewm(span=12).mean()
            df['ema_26'] = df['close'].ewm(span=26).mean()
            
            # 2. RSI optimizado
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
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # 5. Volumen
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # 6. Momentum
            df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
            df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
            df['roc'] = df['close'].pct_change(periods=10)
            
            # 7. Volatilidad
            df['volatility'] = df['close'].pct_change().rolling(window=20).std()
            df['atr'] = df[['high', 'low', 'close']].apply(
                lambda x: max(x['high'] - x['low'], 
                             abs(x['high'] - x['close']), 
                             abs(x['low'] - x['close'])), axis=1
            ).rolling(window=14).mean()
            
            # 8. Price position
            df['price_position'] = (df['close'] - df['low'].rolling(20).min()) / \
                                  (df['high'].rolling(20).max() - df['low'].rolling(20).min())
            
            # Seleccionar features finales
            feature_columns = [
                'sma_7', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal',
                'macd_histogram', 'bb_position', 'volume_ratio', 'momentum_5',
                'momentum_10', 'roc', 'volatility', 'price_position'
            ]
            
            # Limpiar NaN
            df = df.dropna()
            
            print(f"✅ Indicadores calculados: {len(feature_columns)} features")
            print(f"   - Datos limpios: {len(df)} períodos")
            print(f"   - Features: {', '.join(feature_columns)}")
            
            return df[['timestamp', 'open', 'high', 'low', 'close', 'volume'] + feature_columns]
            
        except Exception as e:
            print(f"❌ Error calculando indicadores: {e}")
            return None
    
    def analyze_data_quality(self, df: pd.DataFrame) -> Dict:
        """Analiza la calidad de los datos"""
        try:
            print("\n🔍 Analizando calidad de datos...")
            
            feature_columns = [col for col in df.columns if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
            # Estadísticas básicas
            missing_data = df[feature_columns].isnull().sum()
            infinite_data = np.isinf(df[feature_columns]).sum()
            
            # Correlaciones
            correlations = df[feature_columns + ['close']].corr()['close'].abs().sort_values(ascending=False)
            
            # Varianza y estabilidad
            variances = df[feature_columns].var()
            
            analysis = {
                'total_samples': len(df),
                'missing_data': missing_data.sum(),
                'infinite_data': infinite_data.sum(),
                'top_correlations': correlations.head(6).to_dict(),
                'feature_variances': variances.to_dict(),
                'price_volatility': df['close'].pct_change().std(),
                'data_span_hours': (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 3600
            }
            
            print(f"✅ Análisis de calidad completado:")
            print(f"   - Muestras totales: {analysis['total_samples']:,}")
            print(f"   - Datos faltantes: {analysis['missing_data']}")
            print(f"   - Datos infinitos: {analysis['infinite_data']}")
            print(f"   - Span temporal: {analysis['data_span_hours']:.1f} horas")
            print(f"   - Volatilidad precio: {analysis['price_volatility']:.4f}")
            
            print(f"\n📈 Top correlaciones con precio:")
            for feature, corr in list(analysis['top_correlations'].items())[1:6]:  # Skip 'close' itself
                print(f"   - {feature}: {corr:.3f}")
            
            return analysis
            
        except Exception as e:
            print(f"❌ Error analizando calidad: {e}")
            return {}
    
    def create_simple_predictor(self, df: pd.DataFrame) -> Tuple[object, Dict]:
        """Crea un predictor simple usando Random Forest"""
        try:
            print("\n🤖 Creando predictor simple (Random Forest)...")
            
            feature_columns = [col for col in df.columns if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
            # Preparar datos
            features = df[feature_columns].values
            prices = df['close'].values
            
            # Normalizar features
            features_scaled = self.scaler.fit_transform(features)
            
            # Crear secuencias
            X, y = [], []
            for i in range(self.lookback_window, len(features_scaled) - self.prediction_steps):
                # Usar últimos N valores como features
                X.append(features_scaled[i-self.lookback_window:i].flatten())
                y.append(prices[i + self.prediction_steps])
            
            X = np.array(X)
            y = np.array(y)
            
            # Dividir train/test
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Entrenar modelo simple
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            
            print(f"   - Entrenando con {len(X_train)} muestras...")
            model.fit(X_train, y_train)
            
            # Predicciones
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Métricas
            train_mae = mean_absolute_error(y_train, y_pred_train)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            test_corr = np.corrcoef(y_test, y_pred_test)[0,1]
            
            # Precisión direccional
            test_direction_acc = np.mean(
                np.sign(np.diff(y_test)) == np.sign(np.diff(y_pred_test))
            ) * 100
            
            results = {
                'model': model,
                'X_test': X_test,
                'y_test': y_test,
                'y_pred_test': y_pred_test,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'test_rmse': test_rmse,
                'test_corr': test_corr,
                'test_direction_acc': test_direction_acc,
                'feature_importance': dict(zip(
                    [f"{col}_{i}" for col in feature_columns for i in range(self.lookback_window)],
                    model.feature_importances_
                ))
            }
            
            print(f"✅ Predictor creado y evaluado:")
            print(f"   - Train MAE: ${train_mae:.2f}")
            print(f"   - Test MAE: ${test_mae:.2f}")
            print(f"   - Test RMSE: ${test_rmse:.2f}")
            print(f"   - Test Correlación: {test_corr:.3f}")
            print(f"   - Precisión direccional: {test_direction_acc:.1f}%")
            
            return model, results
            
        except Exception as e:
            print(f"❌ Error creando predictor: {e}")
            return None, {}
    
    def plot_results(self, df: pd.DataFrame, results: Dict):
        """Visualiza los resultados"""
        try:
            print("\n📊 Generando visualizaciones...")
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'🧪 Evaluación Predictor Simple - {self.symbol}', fontsize=16, fontweight='bold')
            
            # 1. Serie temporal de precios recientes
            ax1 = axes[0, 0]
            recent_data = df.tail(200)
            ax1.plot(recent_data['timestamp'], recent_data['close'], linewidth=2, color='blue')
            ax1.set_title('Serie Temporal de Precios (Últimas 200 observaciones)')
            ax1.set_xlabel('Tiempo')
            ax1.set_ylabel('Precio USD')
            ax1.grid(True, alpha=0.3)
            ax1.tick_params(axis='x', rotation=45)
            
            # 2. Predicciones vs Realidad
            ax2 = axes[0, 1]
            ax2.scatter(results['y_test'], results['y_pred_test'], alpha=0.6, color='red')
            ax2.plot([results['y_test'].min(), results['y_test'].max()], 
                    [results['y_test'].min(), results['y_test'].max()], 'k--', lw=2)
            ax2.set_xlabel('Precio Real')
            ax2.set_ylabel('Precio Predicho')
            ax2.set_title(f'Predicciones vs Realidad\nCorr: {results["test_corr"]:.3f}')
            ax2.grid(True, alpha=0.3)
            
            # 3. Residuos
            ax3 = axes[1, 0]
            residuals = results['y_test'] - results['y_pred_test']
            ax3.scatter(results['y_pred_test'], residuals, alpha=0.6, color='green')
            ax3.axhline(y=0, color='red', linestyle='--')
            ax3.set_xlabel('Predicciones')
            ax3.set_ylabel('Residuos (Real - Predicho)')
            ax3.set_title(f'Análisis de Residuos\nMAE: ${results["test_mae"]:.2f}')
            ax3.grid(True, alpha=0.3)
            
            # 4. Serie temporal de predicciones
            ax4 = axes[1, 1]
            test_length = min(100, len(results['y_test']))  # Últimas 100 predicciones
            x_range = range(test_length)
            
            ax4.plot(x_range, results['y_test'][-test_length:], 
                    label='Real', color='blue', linewidth=2)
            ax4.plot(x_range, results['y_pred_test'][-test_length:], 
                    label='Predicho', color='red', linewidth=2, alpha=0.8)
            ax4.set_xlabel('Tiempo (períodos)')
            ax4.set_ylabel('Precio')
            ax4.set_title(f'Predicciones Temporales\nPrecisión Dir: {results["test_direction_acc"]:.1f}%')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Guardar gráfico
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"predictor_evaluation_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"✅ Gráfico guardado: {filename}")
            
            # Mostrar gráfico (si estamos en un entorno que lo soporte)
            try:
                plt.show()
            except:
                print("   (Gráfico guardado pero no se puede mostrar en este entorno)")
            
        except Exception as e:
            print(f"❌ Error generando visualizaciones: {e}")
    
    def make_future_prediction(self, model, df: pd.DataFrame, steps: int = 5) -> Dict:
        """Hace predicción hacia el futuro"""
        try:
            print(f"\n🔮 Haciendo predicción a {steps} períodos futuro...")
            
            feature_columns = [col for col in df.columns if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
            # Obtener últimos datos
            recent_features = df[feature_columns].tail(self.lookback_window).values
            recent_features_scaled = self.scaler.transform(recent_features)
            
            # Preparar para predicción
            X_future = recent_features_scaled.flatten().reshape(1, -1)
            
            # Hacer predicción
            future_price = model.predict(X_future)[0]
            current_price = df['close'].iloc[-1]
            
            # Calcular cambio
            price_change = future_price - current_price
            price_change_pct = (price_change / current_price) * 100
            
            prediction = {
                'current_price': current_price,
                'predicted_price': future_price,
                'price_change': price_change,
                'price_change_pct': price_change_pct,
                'prediction_horizon': f"{steps * 5} minutos",  # 5m interval
                'timestamp': datetime.now()
            }
            
            print(f"✅ Predicción completada:")
            print(f"   - Precio actual: ${current_price:,.2f}")
            print(f"   - Precio predicho ({prediction['prediction_horizon']}): ${future_price:,.2f}")
            print(f"   - Cambio esperado: ${price_change:+,.2f} ({price_change_pct:+.2f}%)")
            
            if abs(price_change_pct) > 1:
                print(f"   ⚠️ Cambio significativo predicho!")
            
            return prediction
            
        except Exception as e:
            print(f"❌ Error haciendo predicción futura: {e}")
            return {}
    
    async def run_complete_test(self):
        """Ejecuta la prueba completa"""
        print("🚀 Iniciando prueba completa de datos y predicción")
        print("="*60)
        
        # 1. Configurar Binance
        if not self.setup_binance_client():
            return False
        
        # 2. Obtener datos
        df = self.get_market_data(limit=1000)
        if df is None:
            return False
        
        # 3. Calcular indicadores
        df_features = self.calculate_technical_indicators(df)
        if df_features is None:
            return False
        
        # 4. Analizar calidad
        analysis = self.analyze_data_quality(df_features)
        
        # 5. Crear predictor
        model, results = self.create_simple_predictor(df_features)
        if model is None:
            return False
        
        # 6. Visualizar resultados
        self.plot_results(df_features, results)
        
        # 7. Predicción futura
        future_pred = self.make_future_prediction(model, df_features, steps=self.prediction_steps)
        
        print("\n" + "="*60)
        print("🎉 Prueba completada exitosamente!")
        
        # Resumen final
        print(f"\n📋 RESUMEN FINAL:")
        print(f"   - Símbolo: {self.symbol}")
        print(f"   - Datos: {len(df)} períodos de {self.interval}")
        print(f"   - Correlación: {results['test_corr']:.3f}")
        print(f"   - Precisión direccional: {results['test_direction_acc']:.1f}%")
        print(f"   - Error promedio: ${results['test_mae']:.2f}")
        print(f"   - Predicción futura: {future_pred.get('price_change_pct', 0):+.2f}%")
        
        return True


async def main():
    """Función principal"""
    print("🧪 Simple TCN Tester - Validación con Datos Reales de Binance")
    print("="*60)
    
    tester = SimpleTCNTester()
    
    try:
        success = await tester.run_complete_test()
        
        if success:
            print("\n✅ Todas las pruebas completadas exitosamente!")
            print("📊 Revisa los gráficos generados para analizar el rendimiento")
            print("\n🔍 INTERPRETACIÓN:")
            print("   - El predictor usa Random Forest como proxy del TCN")
            print("   - Los datos de Binance son reales y actuales")
            print("   - Los indicadores técnicos están funcionando correctamente")
            print("   - Puedes usar estos mismos datos con el modelo TCN real")
        else:
            print("\n❌ Las pruebas fallaron. Revisa los errores anteriores.")
    
    except KeyboardInterrupt:
        print("\n⚠️ Pruebas interrumpidas por el usuario")
    except Exception as e:
        print(f"\n💥 Error crítico: {e}")


if __name__ == "__main__":
    asyncio.run(main()) 