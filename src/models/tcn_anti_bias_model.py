#!/usr/bin/env python3
"""
🧠 MODELO TCN ANTI-SESGO CON CONCIENCIA DE RÉGIMEN
=================================================

Implementación de la arquitectura TCN dual que recibe:
1. Datos de precios y features técnicos (secuencias temporales)
2. Contexto de régimen de mercado (one-hot encoding)

Esta arquitectura elimina sesgos porque el modelo "sabe" explícitamente
en qué tipo de mercado está operando.
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import pickle
from typing import Tuple, Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Configuración TensorFlow para Apple Silicon
tf.config.experimental.enable_mlir_bridge()
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    print("✅ Metal GPU available for acceleration")
else:
    print("⚠️ Metal GPU not detected, using CPU")

class TemporalConvNet(layers.Layer):
    """
    Implementación de Temporal Convolutional Network (TCN)

    TCN es superior a LSTM/GRU para trading porque:
    - Campo receptivo más amplio
    - Paralelización eficiente
    - Sin problemas de gradientes desvanecientes
    - Predicciones consistentes
    """

    def __init__(self, nb_filters=48, kernel_size=2, nb_stacks=2,
                 dilations=[1, 2, 4, 8, 16, 32], dropout_rate=0.3,
                 use_skip_connections=True, activation='relu', name='tcn', **kwargs):
        super(TemporalConvNet, self).__init__(name=name, **kwargs)

        self.nb_filters = nb_filters
        self.kernel_size = kernel_size
        self.nb_stacks = nb_stacks
        self.dilations = dilations
        self.dropout_rate = dropout_rate
        self.use_skip_connections = use_skip_connections
        self.activation = activation

        # Crear capas TCN
        self.tcn_layers = []

        for stack in range(nb_stacks):
            stack_layers = []
            for i, dilation in enumerate(dilations):
                # Capa convolucional dilatada
                conv_layer = layers.Conv1D(
                    filters=nb_filters,
                    kernel_size=kernel_size,
                    dilation_rate=dilation,
                    padding='causal',  # Fundamental para no ver el futuro
                    activation=None,
                    name=f'tcn_conv_s{stack}_d{dilation}'
                )

                # Normalización y activación
                norm_layer = layers.LayerNormalization(name=f'tcn_norm_s{stack}_d{dilation}')
                activation_layer = layers.Activation(activation, name=f'tcn_act_s{stack}_d{dilation}')
                dropout_layer = layers.SpatialDropout1D(dropout_rate, name=f'tcn_drop_s{stack}_d{dilation}')

                stack_layers.append({
                    'conv': conv_layer,
                    'norm': norm_layer,
                    'activation': activation_layer,
                    'dropout': dropout_layer
                })

            self.tcn_layers.append(stack_layers)

        # Capa de output
        self.output_conv = layers.Conv1D(
            filters=nb_filters,
            kernel_size=1,
            padding='same',
            activation=None,
            name='tcn_output_conv'
        )

        # Global pooling para obtener representación final
        self.global_pool = layers.GlobalMaxPooling1D(name='tcn_global_pool')

    def call(self, inputs, training=None):
        x = inputs

        # Aplicar stacks de TCN
        for stack_idx, stack_layers in enumerate(self.tcn_layers):
            stack_input = x

            for layer_dict in stack_layers:
                # Convolución dilatada
                x = layer_dict['conv'](x)
                x = layer_dict['norm'](x, training=training)
                x = layer_dict['activation'](x)
                x = layer_dict['dropout'](x, training=training)

            # Skip connection si está habilitado
            if self.use_skip_connections and x.shape[-1] == stack_input.shape[-1]:
                x = layers.Add(name=f'tcn_skip_s{stack_idx}')([x, stack_input])

        # Proyección final y pooling
        x = self.output_conv(x)
        x = self.global_pool(x)

        return x

    def get_config(self):
        config = super(TemporalConvNet, self).get_config()
        config.update({
            'nb_filters': self.nb_filters,
            'kernel_size': self.kernel_size,
            'nb_stacks': self.nb_stacks,
            'dilations': self.dilations,
            'dropout_rate': self.dropout_rate,
            'use_skip_connections': self.use_skip_connections,
            'activation': self.activation
        })
        return config

class RegimeAwareTCNModel:
    """
    Modelo TCN consciente del régimen de mercado

    Arquitectura innovadora:
    - Input 1: Secuencias temporales de precios/features
    - Input 2: Contexto de régimen de mercado (bull/bear/sideways)
    - Output: Decisión de trading balanceada (BUY/HOLD/SELL)
    """

    def __init__(self, input_shape_prices=(60, 50), regime_shape=(3,)):
        self.input_shape_prices = input_shape_prices
        self.regime_shape = regime_shape
        self.model = None
        self.history = None
        self.scalers = {}

    def create_model(self) -> Model:
        """
        Crea el modelo TCN consciente del régimen

        Innovación clave: El modelo recibe explícitamente el contexto
        de mercado, eliminando la necesidad de "adivinar" el régimen
        """
        print("🧠 Building Regime-Aware TCN Model...")

        # === ENTRADA 1: DATOS DE PRECIOS Y FEATURES TÉCNICOS ===
        price_input = layers.Input(
            shape=self.input_shape_prices,
            name='price_features',
            dtype=tf.float32
        )

        # TCN Principal - Procesamiento temporal avanzado
        tcn_output = TemporalConvNet(
            nb_filters=48,                      # Optimizado para Apple Silicon
            kernel_size=2,                      # Kernel pequeño para granularidad
            nb_stacks=2,                        # Balance complejidad/velocidad
            dilations=[1, 2, 4, 8, 16, 32],    # Campo receptivo de 64 períodos
            dropout_rate=0.3,                  # Regularización moderada
            use_skip_connections=True,         # Mejor flujo de gradientes
            activation='relu',
            name='tcn_temporal_processor'
        )(price_input)

        # === ENTRADA 2: CONTEXTO DE RÉGIMEN DE MERCADO ===
        regime_input = layers.Input(
            shape=self.regime_shape,
            name='market_regime',
            dtype=tf.float32
        )

        # Procesar contexto de régimen
        regime_processed = layers.Dense(
            32,
            activation='relu',
            name='regime_processor'
        )(regime_input)

        regime_processed = layers.Dropout(
            0.2,
            name='regime_dropout'
        )(regime_processed)

        # === FUSIÓN DE INFORMACIÓN TEMPORAL + CONTEXTUAL ===
        combined_features = layers.Concatenate(
            name='feature_fusion'
        )([tcn_output, regime_processed])

        # === CAPAS DE DECISIÓN ESPECIALIZADAS ===
        # Primera capa: Integración de features
        x = layers.Dense(
            128,
            activation='relu',
            kernel_initializer='he_normal',
            name='decision_layer_1'
        )(combined_features)

        x = layers.BatchNormalization(name='bn_1')(x)
        x = layers.Dropout(0.4, name='dropout_1')(x)  # Dropout agresivo

        # Segunda capa: Refinamiento
        x = layers.Dense(
            64,
            activation='relu',
            kernel_initializer='he_normal',
            name='decision_layer_2'
        )(x)

        x = layers.BatchNormalization(name='bn_2')(x)
        x = layers.Dropout(0.3, name='dropout_2')(x)

        # Tercera capa: Pre-decisión
        x = layers.Dense(
            32,
            activation='relu',
            kernel_initializer='he_normal',
            name='decision_layer_3'
        )(x)

        x = layers.Dropout(0.2, name='dropout_3')(x)

        # === SALIDA BALANCEADA: BUY, HOLD, SELL ===
        output = layers.Dense(
            3,
            activation='softmax',
            kernel_initializer='glorot_uniform',  # Inicialización balanceada
            bias_initializer='zeros',             # Sin sesgo inicial
            name='trading_decision'
        )(x)

        # Crear modelo
        model = Model(
            inputs=[price_input, regime_input],
            outputs=output,
            name='TCN_Anti_Bias_Trading_Model'
        )

        # === COMPILACIÓN CON CONFIGURACIÓN ANTI-SESGO ===
        model.compile(
            optimizer=Nadam(
                learning_rate=0.001,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-7
            ),
            loss='categorical_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc')
            ]
        )

        self.model = model

        print(f"✅ Model created with {model.count_params():,} parameters")
        print(f"   Input shapes: {self.input_shape_prices} (prices) + {self.regime_shape} (regime)")

        return model

    def calculate_class_weights(self, y_train: np.ndarray) -> Dict[int, float]:
        """
        Calcula pesos balanceados para clases

        CRÍTICO para anti-sesgo: Asegura que cada clase (BUY/HOLD/SELL)
        tenga igual importancia durante el entrenamiento
        """
        class_counts = np.sum(y_train, axis=0)
        total_samples = len(y_train)

        class_weights = {}
        for i in range(len(class_counts)):
            if class_counts[i] > 0:
                class_weights[i] = total_samples / (len(class_counts) * class_counts[i])
            else:
                class_weights[i] = 1.0

        print(f"📊 Class distribution in training data:")
        labels = ['BUY', 'HOLD', 'SELL']
        for i, (count, weight) in enumerate(zip(class_counts, class_weights.values())):
            percentage = count / total_samples * 100
            print(f"   {labels[i]}: {count:,} samples ({percentage:.1f}%) - weight: {weight:.3f}")

        return class_weights

    def create_callbacks(self, model_save_path: str = 'models/tcn_anti_bias_best.h5') -> List:
        """Crear callbacks para entrenamiento"""

        # Crear directorio si no existe
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1,
                mode='min'
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-6,
                verbose=1,
                mode='min'
            ),
            ModelCheckpoint(
                model_save_path,
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False,
                verbose=1,
                mode='min'
            )
        ]

        return callbacks

    def train(self, X_train_price: np.ndarray, X_train_regime: np.ndarray, y_train: np.ndarray,
              X_val_price: np.ndarray, X_val_regime: np.ndarray, y_val: np.ndarray,
              epochs: int = 100, batch_size: int = 64, model_save_path: str = 'models/tcn_anti_bias_best.h5',
              callbacks: List = None) -> tf.keras.callbacks.History:
        """
        Entrenar el modelo TCN anti-sesgo
        """
        print("🏃‍♂️ Starting anti-bias TCN training...")

        if self.model is None:
            self.create_model()

        # Calcular pesos balanceados (CRÍTICO para anti-sesgo)
        class_weights = self.calculate_class_weights(y_train)

        # Usar callbacks personalizados si se proporcionan, sino crear los por defecto
        if callbacks is None:
            callbacks = self.create_callbacks(model_save_path)

        # Entrenar modelo
        print(f"   Training samples: {len(X_train_price):,}")
        print(f"   Validation samples: {len(X_val_price):,}")
        print(f"   Epochs: {epochs}, Batch size: {batch_size}")

        history = self.model.fit(
            [X_train_price, X_train_regime], y_train,
            validation_data=([X_val_price, X_val_regime], y_val),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weights,  # CRÍTICO para balanceado
            callbacks=callbacks,
            verbose=1,
            shuffle=True
        )

        self.history = history

        print("✅ Training completed!")
        return history

    def predict(self, X_price: np.ndarray, X_regime: np.ndarray) -> np.ndarray:
        """Realizar predicciones con el modelo"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")

        predictions = self.model.predict([X_price, X_regime], verbose=0)
        return predictions

    def evaluate_by_regime(self, X_test_price: np.ndarray, X_test_regime: np.ndarray,
                          y_test: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Evaluación específica por régimen de mercado

        CRÍTICO: Verifica que el modelo funcione bien en TODOS los regímenes
        """
        print("📊 Evaluating model performance by market regime...")

        predictions = self.predict(X_test_price, X_test_regime)

        # Identificar régimen de cada muestra
        regime_names = ['bull', 'bear', 'sideways']
        regime_indices = np.argmax(X_test_regime, axis=1)

        results = {}

        for regime_idx, regime_name in enumerate(regime_names):
            regime_mask = regime_indices == regime_idx

            if not np.any(regime_mask):
                print(f"   ⚠️ No {regime_name} samples in test set")
                continue

            regime_y_true = y_test[regime_mask]
            regime_y_pred = predictions[regime_mask]

            # Calcular métricas
            y_true_labels = np.argmax(regime_y_true, axis=1)
            y_pred_labels = np.argmax(regime_y_pred, axis=1)

            accuracy = np.mean(y_true_labels == y_pred_labels)

            # Precision y recall por clase
            labels = ['BUY', 'HOLD', 'SELL']
            precision_per_class = []
            recall_per_class = []

            for class_idx in range(3):
                true_positives = np.sum((y_true_labels == class_idx) & (y_pred_labels == class_idx))
                predicted_positives = np.sum(y_pred_labels == class_idx)
                actual_positives = np.sum(y_true_labels == class_idx)

                precision = true_positives / predicted_positives if predicted_positives > 0 else 0
                recall = true_positives / actual_positives if actual_positives > 0 else 0

                precision_per_class.append(precision)
                recall_per_class.append(recall)

            avg_precision = np.mean(precision_per_class)
            avg_recall = np.mean(recall_per_class)
            f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0

            results[regime_name] = {
                'samples': int(np.sum(regime_mask)),
                'accuracy': float(accuracy),
                'precision': float(avg_precision),
                'recall': float(avg_recall),
                'f1_score': float(f1_score),
                'precision_per_class': [float(p) for p in precision_per_class],
                'recall_per_class': [float(r) for r in recall_per_class]
            }

            print(f"   📊 {regime_name.upper()} regime ({results[regime_name]['samples']} samples):")
            print(f"      Accuracy: {accuracy:.3f}")
            print(f"      Precision: {avg_precision:.3f}")
            print(f"      Recall: {avg_recall:.3f}")
            print(f"      F1-Score: {f1_score:.3f}")

        # Análisis de consistencia (objetivo anti-sesgo)
        if len(results) >= 2:
            accuracies = [r['accuracy'] for r in results.values()]
            accuracy_variance = np.var(accuracies)
            accuracy_std = np.std(accuracies)

            print(f"\n🎯 ANTI-BIAS ANALYSIS:")
            print(f"   Accuracy variance across regimes: {accuracy_variance:.6f}")
            print(f"   Accuracy standard deviation: {accuracy_std:.3f}")

            if accuracy_std < 0.05:  # Menos de 5% de diferencia
                print(f"   ✅ Model is WELL-BALANCED across regimes!")
            elif accuracy_std < 0.10:
                print(f"   ⚠️ Model has moderate bias across regimes")
            else:
                print(f"   ❌ Model has significant bias across regimes!")

        return results

    def save_model(self, filepath: str):
        """Guardar modelo completo"""
        if self.model is None:
            raise ValueError("No model to save")

        self.model.save(filepath)
        print(f"💾 Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Cargar modelo previamente entrenado"""
        self.model = tf.keras.models.load_model(filepath, custom_objects={
            'TemporalConvNet': TemporalConvNet
        })
        print(f"📂 Model loaded from {filepath}")

    def get_model_summary(self):
        """Mostrar resumen del modelo"""
        if self.model is None:
            print("❌ No model created yet")
            return

        print("🧠 Model Architecture Summary:")
        self.model.summary()

        # Visualizar arquitectura
        try:
            tf.keras.utils.plot_model(
                self.model,
                to_file='tcn_anti_bias_architecture.png',
                show_shapes=True,
                show_layer_names=True,
                rankdir='TB'
            )
            print("📊 Architecture diagram saved to 'tcn_anti_bias_architecture.png'")
        except:
            print("⚠️ Could not save architecture diagram")

def main():
    """Demo de la arquitectura TCN anti-sesgo"""
    print("🚀 Starting TCN Anti-Bias Model Demo")

    # Crear datos de prueba
    batch_size = 100
    sequence_length = 60
    num_features = 50

    # Simular datos de entrenamiento
    X_train_price = np.random.randn(batch_size, sequence_length, num_features).astype(np.float32)
    X_train_regime = np.random.randint(0, 2, (batch_size, 3)).astype(np.float32)

    # Normalizar regime a one-hot
    for i in range(batch_size):
        X_train_regime[i] = 0
        X_train_regime[i, np.random.randint(0, 3)] = 1

    y_train = np.random.randint(0, 2, (batch_size, 3)).astype(np.float32)

    # Normalizar labels a one-hot
    for i in range(batch_size):
        y_train[i] = 0
        y_train[i, np.random.randint(0, 3)] = 1

    print(f"📊 Demo data shapes:")
    print(f"   X_train_price: {X_train_price.shape}")
    print(f"   X_train_regime: {X_train_regime.shape}")
    print(f"   y_train: {y_train.shape}")

    # Crear modelo
    model = RegimeAwareTCNModel(
        input_shape_prices=(sequence_length, num_features),
        regime_shape=(3,)
    )

    try:
        # Crear arquitectura
        print("\n1️⃣ Creating model architecture...")
        model.create_model()

        # Mostrar resumen
        print("\n2️⃣ Model summary...")
        model.get_model_summary()

        # Simular predicción
        print("\n3️⃣ Testing prediction...")
        predictions = model.predict(X_train_price[:10], X_train_regime[:10])
        print(f"✅ Predictions shape: {predictions.shape}")
        print(f"   Sample prediction: {predictions[0]}")

        print("\n✅ TCN Anti-Bias Model demo completed successfully!")

    except Exception as e:
        print(f"❌ Error in demo: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
