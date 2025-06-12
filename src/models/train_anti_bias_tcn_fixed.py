#!/usr/bin/env python3
"""
🚀 ENTRENAMIENTO TCN ANTI-SESGO CORREGIDO - SCRIPT PRINCIPAL
===========================================================

Script corregido para entrenar el modelo TCN consciente del régimen
que elimina sesgos de mercado usando el algoritmo mejorado de clasificación.

CORRECCIONES:
- Uso del algoritmo mejorado de clasificación de regímenes
- Mejor balance entre bull/bear/sideways
- Detección más precisa de tendencias de mercado
"""

import os
import sys
import numpy as np
import pandas as pd
import json
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Importar módulos propios
from regime_classifier import MarketRegimeClassifier
from tcn_features_engineering import AdvancedFeatureEngineer
from tcn_anti_bias_model import RegimeAwareTCNModel

class AntiBiasTCNTrainerFixed:
    """
    Entrenador CORREGIDO del sistema TCN anti-sesgo

    Mejoras:
    - Algoritmo mejorado de clasificación de regímenes
    - Mejor balance de datos de entrenamiento
    - Detección más precisa de bull/bear markets
    """

    def __init__(self, config_path: str = None):
        self.config = self.load_config(config_path)
        # IMPORTANTE: Usar algoritmo mejorado con parámetros optimizados
        self.regime_classifier = MarketRegimeClassifier(
            trend_window=20,
            trend_threshold=0.015,
            momentum_threshold=0.015
        )
        self.feature_engineer = AdvancedFeatureEngineer()
        self.model = None
        self.training_history = {}

        # Crear directorios necesarios
        os.makedirs('models', exist_ok=True)
        os.makedirs('data', exist_ok=True)
        os.makedirs('results', exist_ok=True)

    def load_config(self, config_path: str = None) -> dict:
        """Cargar configuración de entrenamiento optimizada"""
        default_config = {
            # Datos
            'symbols': ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'DOTUSDT'],
            'days_history': 730,  # 2 años de datos
            'timeframe': '5m',

            # Regímenes (MEJORADO)
            'trend_window': 20,           # Ventana más pequeña
            'trend_threshold': 0.015,     # Threshold más bajo (1.5% vs 3%)
            'momentum_threshold': 0.015,  # Threshold de momentum
            'min_samples_per_regime': 6000,  # Reducido para mejor balance

            # Features
            'sequence_length': 60,
            'prediction_horizon': 1,
            'price_threshold': 0.005,  # 0.5% para BUY/SELL

            # Modelo
            'tcn_filters': 48,
            'tcn_kernel_size': 2,
            'tcn_stacks': 2,
            'tcn_dilations': [1, 2, 4, 8, 16, 32],
            'tcn_dropout': 0.3,

            # Entrenamiento
            'epochs': 150,      # Más épocas para mejor convergencia
            'batch_size': 64,
            'learning_rate': 0.001,
            'validation_split': 0.2,
            'test_split': 0.2,

            # Paths
            'model_save_path': 'models/tcn_anti_bias_fixed.h5',
            'data_save_path': 'data/balanced_training_data_fixed.pkl',
            'results_save_path': 'results/training_results_fixed.json'
        }

        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = json.load(f)
            default_config.update(user_config)

        return default_config

    def step_1_download_and_classify_data_fixed(self) -> pd.DataFrame:
        """
        Paso 1 CORREGIDO: Descargar datos y clasificar regímenes con algoritmo mejorado
        """
        print("🚀 STEP 1 FIXED: Downloading data and classifying market regimes (IMPROVED)")
        print("=" * 70)

        # Verificar si ya tenemos datos mejorados
        if os.path.exists('data/balanced_training_data_improved.pkl'):
            print(f"📂 Loading existing improved data...")
            try:
                balanced_df = self.regime_classifier.load_balanced_data('data/balanced_training_data_improved.pkl')
                if balanced_df is not None and len(balanced_df) > 10000:
                    print(f"✅ Using existing improved data: {len(balanced_df):,} samples")

                    # Verificar balance
                    regime_counts = balanced_df['regime'].value_counts()
                    print(f"📊 Current balance:")
                    for regime, count in regime_counts.items():
                        pct = count / len(balanced_df) * 100
                        print(f"   {regime}: {count:,} ({pct:.1f}%)")

                    return balanced_df
            except:
                print("⚠️ Error loading improved data, will re-classify")

        # Cargar datos existentes o descargar nuevos
        if os.path.exists(self.config['data_save_path'].replace('_fixed', '')):
            print(f"📂 Loading existing raw data...")
            raw_df = self.regime_classifier.load_balanced_data(self.config['data_save_path'].replace('_fixed', ''))
            # Remover clasificación anterior
            if 'regime' in raw_df.columns:
                raw_df = raw_df.drop('regime', axis=1)
        else:
            print(f"📊 Downloading {self.config['days_history']} days of fresh data...")
            raw_df = self.regime_classifier.download_balanced_data(
                symbols=self.config['symbols'],
                days=self.config['days_history']
            )

        print(f"📊 Processing {len(raw_df):,} total samples")

        # NUEVO: Clasificar regímenes con algoritmo mejorado
        print("\n🎯 Classifying market regimes with IMPROVED algorithm...")
        df_with_regimes = self.regime_classifier.classify_market_regimes_improved(raw_df)

        # Mostrar resultados de clasificación
        regime_counts = df_with_regimes['regime'].value_counts()
        print(f"\n📊 Classification results:")
        for regime, count in regime_counts.items():
            pct = count / len(df_with_regimes) * 100
            print(f"   {regime}: {count:,} ({pct:.1f}%)")

        # Balancear datos por régimen
        print("\n⚖️ Balancing regime data...")
        balanced_df = self.regime_classifier.balance_regime_data(
            df_with_regimes,
            min_samples_per_regime=self.config['min_samples_per_regime']
        )

        # Guardar datos balanceados mejorados
        print(f"\n💾 Saving improved balanced data to {self.config['data_save_path']}")
        self.regime_classifier.save_balanced_data(balanced_df, self.config['data_save_path'])

        # También guardar copia como "improved"
        self.regime_classifier.save_balanced_data(balanced_df, 'data/balanced_training_data_improved.pkl')

        print(f"\n✅ Step 1 FIXED completed: {len(balanced_df):,} balanced samples ready")
        return balanced_df

    def step_2_feature_engineering(self, df: pd.DataFrame) -> tuple:
        """
        Paso 2: Feature engineering avanzado (sin cambios)
        """
        print("\n🔧 STEP 2: Advanced feature engineering")
        print("=" * 60)

        # Preparar features avanzados
        features_df, scalers = self.feature_engineer.prepare_advanced_features(df)

        # Añadir columnas necesarias para creación de secuencias
        features_df['close'] = df['close']
        features_df['regime'] = df['regime']

        print(f"✅ Features created: {features_df.shape[1]} features")

        # Split estratificado
        print("\n📊 Creating stratified splits...")
        train_df, val_df, test_df = self.feature_engineer.regime_stratified_split(
            features_df,
            test_size=self.config['test_split'],
            validation_size=self.config['validation_split']
        )

        # Crear secuencias de entrenamiento
        print("\n🔄 Creating training sequences...")
        X_train_price, X_train_regime, y_train = self.feature_engineer.create_training_sequences(
            train_df,
            sequence_length=self.config['sequence_length'],
            prediction_horizon=self.config['prediction_horizon'],
            price_threshold=self.config['price_threshold']
        )

        print("\n🔄 Creating validation sequences...")
        X_val_price, X_val_regime, y_val = self.feature_engineer.create_training_sequences(
            val_df,
            sequence_length=self.config['sequence_length'],
            prediction_horizon=self.config['prediction_horizon'],
            price_threshold=self.config['price_threshold']
        )

        print("\n🔄 Creating test sequences...")
        X_test_price, X_test_regime, y_test = self.feature_engineer.create_training_sequences(
            test_df,
            sequence_length=self.config['sequence_length'],
            prediction_horizon=self.config['prediction_horizon'],
            price_threshold=self.config['price_threshold']
        )

        # Guardar scalers
        scaler_path = 'models/feature_scalers_fixed.pkl'
        with open(scaler_path, 'wb') as f:
            pickle.dump(scalers, f)
        print(f"💾 Scalers saved to {scaler_path}")

        print(f"\n✅ Step 2 completed: Training data ready")
        print(f"   Train: {X_train_price.shape}")
        print(f"   Validation: {X_val_price.shape}")
        print(f"   Test: {X_test_price.shape}")

        return (X_train_price, X_train_regime, y_train,
                X_val_price, X_val_regime, y_val,
                X_test_price, X_test_regime, y_test,
                scalers)

    def step_3_train_model(self, training_data: tuple) -> RegimeAwareTCNModel:
        """
        Paso 3: Entrenar modelo TCN anti-sesgo
        """
        print("\n🧠 STEP 3: Training FIXED anti-bias TCN model")
        print("=" * 60)

        (X_train_price, X_train_regime, y_train,
         X_val_price, X_val_regime, y_val,
         X_test_price, X_test_regime, y_test,
         scalers) = training_data

        # Crear modelo
        input_shape_prices = (self.config['sequence_length'], X_train_price.shape[2])
        regime_shape = (3,)  # bull, bear, sideways

        self.model = RegimeAwareTCNModel(
            input_shape_prices=input_shape_prices,
            regime_shape=regime_shape
        )

        # Entrenar modelo
        print(f"🏃‍♂️ Starting FIXED training...")
        history = self.model.train(
            X_train_price, X_train_regime, y_train,
            X_val_price, X_val_regime, y_val,
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            model_save_path=self.config['model_save_path']
        )

        print(f"\n✅ Step 3 completed: FIXED Model trained and saved")
        return self.model

    def step_4_validate_anti_bias(self, training_data: tuple) -> dict:
        """
        Paso 4: Validación anti-sesgo por régimen MEJORADA
        """
        print("\n📊 STEP 4: IMPROVED Anti-bias validation by regime")
        print("=" * 60)

        (X_train_price, X_train_regime, y_train,
         X_val_price, X_val_regime, y_val,
         X_test_price, X_test_regime, y_test,
         scalers) = training_data

        # Evaluación por régimen
        regime_results = self.model.evaluate_by_regime(
            X_test_price, X_test_regime, y_test
        )

        # Análisis de consistencia global
        print(f"\n🎯 IMPROVED ANTI-BIAS ANALYSIS:")

        if len(regime_results) >= 2:
            accuracies = [r['accuracy'] for r in regime_results.values()]
            f1_scores = [r['f1_score'] for r in regime_results.values()]

            accuracy_std = np.std(accuracies)
            f1_std = np.std(f1_scores)
            accuracy_mean = np.mean(accuracies)

            print(f"   Average accuracy: {accuracy_mean:.3f}")
            print(f"   Accuracy std deviation: {accuracy_std:.3f}")
            print(f"   F1-Score std deviation: {f1_std:.3f}")

            # Criterio de éxito anti-sesgo MEJORADO
            if accuracy_std < 0.05 and f1_std < 0.05 and accuracy_mean > 0.7:
                print(f"   ✅ MODEL PASSES IMPROVED ANTI-BIAS TEST!")
                bias_status = "EXCELLENT"
            elif accuracy_std < 0.08 and f1_std < 0.08 and accuracy_mean > 0.6:
                print(f"   ✅ Model passes anti-bias test")
                bias_status = "PASSED"
            elif accuracy_std < 0.12 and accuracy_mean > 0.5:
                print(f"   ⚠️ Model has moderate bias")
                bias_status = "MODERATE"
            else:
                print(f"   ❌ Model has significant bias - needs improvement")
                bias_status = "FAILED"
        else:
            bias_status = "INSUFFICIENT_DATA"

        # Preparar resultados completos
        results = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config,
            'algorithm_version': 'improved_v2',
            'regime_results': regime_results,
            'bias_status': bias_status,
            'global_metrics': {
                'accuracy_mean': float(accuracy_mean) if len(regime_results) >= 2 else None,
                'accuracy_std': float(accuracy_std) if len(regime_results) >= 2 else None,
                'f1_std': float(f1_std) if len(regime_results) >= 2 else None,
                'total_test_samples': int(len(X_test_price))
            }
        }

        # Guardar resultados
        with open(self.config['results_save_path'], 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n💾 Results saved to {self.config['results_save_path']}")
        print(f"✅ Step 4 completed: IMPROVED Anti-bias validation finished")

        return results

    def run_complete_training(self) -> dict:
        """
        Ejecutar proceso completo de entrenamiento anti-sesgo CORREGIDO
        """
        print("🚀 STARTING COMPLETE FIXED ANTI-BIAS TCN TRAINING")
        print("=" * 80)

        start_time = datetime.now()

        try:
            # Paso 1: Datos y regímenes CON ALGORITMO MEJORADO
            balanced_df = self.step_1_download_and_classify_data_fixed()

            # Paso 2: Feature engineering
            training_data = self.step_2_feature_engineering(balanced_df)

            # Paso 3: Entrenar modelo
            model = self.step_3_train_model(training_data)

            # Paso 4: Validación anti-sesgo
            results = self.step_4_validate_anti_bias(training_data)

            # Resumen final
            end_time = datetime.now()
            training_time = end_time - start_time

            print(f"\n🎉 FIXED TRAINING COMPLETED SUCCESSFULLY!")
            print(f"=" * 80)
            print(f"⏱️ Total time: {training_time}")
            print(f"📊 Bias status: {results['bias_status']}")
            print(f"💾 Model saved: {self.config['model_save_path']}")
            print(f"📄 Results saved: {self.config['results_save_path']}")

            # Recomendaciones finales mejoradas
            if results['bias_status'] in ['EXCELLENT', 'PASSED']:
                print(f"\n✅ RECOMMENDATIONS:")
                print(f"   • Model is ready for live trading")
                print(f"   • Improved regime detection should provide better performance")
                print(f"   • Performance is consistent across all market regimes")
                print(f"   • Consider deploying with conservative position sizing")

            elif results['bias_status'] == 'MODERATE':
                print(f"\n⚠️ RECOMMENDATIONS:")
                print(f"   • Model shows improvement but use with caution")
                print(f"   • Test with paper trading first")
                print(f"   • Monitor performance across different market conditions")

            else:
                print(f"\n❌ RECOMMENDATIONS:")
                print(f"   • Model still has bias issues - review data quality")
                print(f"   • Consider adjusting regime classification parameters further")
                print(f"   • May need more diverse training data")

            return results

        except Exception as e:
            print(f"\n❌ ERROR DURING FIXED TRAINING:")
            print(f"   {e}")
            import traceback
            traceback.print_exc()
            return {'error': str(e), 'status': 'FAILED'}

def main():
    """
    Función principal para ejecutar entrenamiento CORREGIDO
    """
    print("🧠 TCN ANTI-BIAS TRAINER FIXED - Main Execution")
    print("=" * 80)

    # Verificar variables de entorno
    api_key = os.environ.get('BINANCE_API_KEY')
    api_secret = os.environ.get('BINANCE_API_SECRET')

    if not api_key or not api_secret:
        print("⚠️ WARNING: Binance API keys not found in environment")
        print("   The trainer will use public data only (limited rate)")
        print("   For full access, set BINANCE_API_KEY and BINANCE_API_SECRET")
    else:
        print("✅ Binance API keys found - full access available")

    # Crear trainer CORREGIDO
    trainer = AntiBiasTCNTrainerFixed()

    # Mostrar configuración
    print(f"\n📋 FIXED TRAINING CONFIGURATION:")
    print(f"   Symbols: {trainer.config['symbols']}")
    print(f"   History: {trainer.config['days_history']} days")
    print(f"   Sequence length: {trainer.config['sequence_length']}")
    print(f"   Epochs: {trainer.config['epochs']}")
    print(f"   Batch size: {trainer.config['batch_size']}")
    print(f"   Regime threshold: {trainer.config['trend_threshold']}")

    # Confirmar ejecución
    response = input(f"\n🚀 Start FIXED training? [y/N]: ").strip().lower()
    if response != 'y':
        print("❌ Training cancelled by user")
        return

    # Ejecutar entrenamiento completo CORREGIDO
    results = trainer.run_complete_training()

    if 'error' not in results:
        print(f"\n🎊 SUCCESS! Your FIXED anti-bias TCN model is ready!")
        print(f"   • Better regime classification")
        print(f"   • Improved balance between bull/bear/sideways")
        print(f"   • More accurate market detection")
    else:
        print(f"\n💥 Training failed. Check logs for details.")

if __name__ == "__main__":
    main()
