#!/usr/bin/env python3
"""
🔍 ANALIZADOR DE REQUERIMIENTOS DE MODELOS TCN
==============================================

Script para analizar exactamente qué input shape y features
requieren los modelos TCN existentes.
"""

import os
import numpy as np
import tensorflow as tf
from pathlib import Path

def analyze_existing_models():
    """Analizar modelos TCN existentes"""
    print("🔍 ANALIZANDO MODELOS TCN EXISTENTES")
    print("=" * 50)

    # Buscar archivos de modelos de producción específicos
    production_models = [
        "production_model_BTCUSDT.h5",
        "production_model_ETHUSDT.h5",
        "production_model_BNBUSDT.h5",
        "production_tcn.h5"
    ]

    found_models = []

    # Buscar modelos de producción en directorio actual
    for model_file in production_models:
        if os.path.exists(model_file):
            found_models.append(model_file)

    # Buscar también otros modelos relevantes
    other_models = [
        "models/tcn_final_btcusdt.h5",
        "models/tcn_final_ethusdt.h5",
        "models/tcn_final_bnbusdt.h5"
    ]

    for model_file in other_models:
        if os.path.exists(model_file):
            found_models.append(model_file)

    if not found_models:
        print("❌ No se encontraron modelos TCN")
        return

    # Analizar cada modelo
    for model_path in found_models:
        print(f"\n📊 ANALIZANDO: {model_path}")
        print("-" * 40)

        try:
            # Cargar modelo
            model = tf.keras.models.load_model(model_path, compile=False)

            # Información básica
            print(f"✅ Modelo cargado exitosamente")
            print(f"   📐 Input shape: {model.input_shape}")
            print(f"   📤 Output shape: {model.output_shape}")
            print(f"   📊 Parámetros: {model.count_params():,}")

            # Detalles del input
            input_shape = model.input_shape
            if input_shape:
                batch_size = input_shape[0]
                timesteps = input_shape[1] if len(input_shape) > 1 else None
                features = input_shape[2] if len(input_shape) > 2 else None

                print(f"   🔢 Batch size: {batch_size}")
                print(f"   ⏰ Timesteps: {timesteps}")
                print(f"   🎯 Features: {features}")

                # Verificar si es compatible con datos dummy
                if timesteps and features:
                    print(f"\n🧪 Probando con datos dummy...")
                    try:
                        dummy_data = np.random.normal(0, 0.1, (1, timesteps, features)).astype(np.float32)
                        prediction = model.predict(dummy_data, verbose=0)

                        print(f"   ✅ Predicción exitosa")
                        print(f"   📊 Output shape: {prediction.shape}")
                        print(f"   🎯 Sample prediction: {prediction[0]}")

                        # Interpretar output
                        if prediction.shape[1] == 3:
                            classes = ['SELL', 'HOLD', 'BUY']
                            predicted_class = np.argmax(prediction[0])
                            confidence = float(prediction[0][predicted_class])
                            print(f"   🤖 Predicción: {classes[predicted_class]} ({confidence:.1%})")

                    except Exception as e:
                        print(f"   ❌ Error en predicción: {e}")

            # Arquitectura del modelo
            print(f"\n🏗️ ARQUITECTURA:")
            model.summary(print_fn=lambda x: print(f"     {x}"))

        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")

    print(f"\n🎯 RESUMEN DE ANÁLISIS")
    print("=" * 50)
    print(f"Modelos encontrados: {len(found_models)}")

    if found_models:
        print("\n📋 REQUERIMIENTOS PARA DATOS REALES:")
        for model_path in found_models:
            try:
                model = tf.keras.models.load_model(model_path, compile=False)
                input_shape = model.input_shape
                symbol = model_path.split('/')[-1].replace('_tcn_production.h5', '').replace('.h5', '')

                if len(input_shape) >= 3:
                    print(f"   • {symbol}:")
                    print(f"     - Timesteps: {input_shape[1]}")
                    print(f"     - Features: {input_shape[2]}")
                    print(f"     - Shape: {input_shape}")

            except:
                continue


def create_feature_mapping_guide():
    """Crear guía de mapeo de features"""
    print(f"\n📝 GUÍA PARA DATOS REALES")
    print("=" * 50)

    print("""
Para usar datos reales de mercado, necesitas:

1. 🔍 VERIFICAR FEATURES EXACTAS:
   - Revisar código de entrenamiento original
   - Buscar archivos de configuración (.pkl, .json)
   - Verificar feature engineering usado

2. 📊 ADAPTAR PROVEEDOR DE DATOS:
   - Crear exactamente las mismas features
   - Usar mismo preprocessing/normalización
   - Mantener mismo orden de features

3. ⏰ AJUSTAR TIMESTEPS:
   - Si modelo espera 40 timesteps, usar ventana de 40
   - Si modelo espera 32 timesteps, usar ventana de 32

4. 🔧 OPCIONES DE SOLUCIÓN:
   a) Modificar proveedor de datos para generar 159 features
   b) Re-entrenar modelos con 66 features (recomendado)
   c) Crear adaptador que mapee 66 -> 159 features
    """)


if __name__ == "__main__":
    analyze_existing_models()
    create_feature_mapping_guide()
