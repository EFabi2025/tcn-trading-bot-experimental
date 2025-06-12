#!/usr/bin/env python3
import os
import sys

# Configurar TensorFlow para macOS
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

try:
    print("Importando TensorFlow...")
    import tensorflow as tf
    print(f"✅ TensorFlow {tf.__version__} importado")

    # Probar cargar modelo
    model_path = 'tcn_anti_bias_fixed.h5'
    if os.path.exists(model_path):
        print(f"Intentando cargar {model_path}...")
        model = tf.keras.models.load_model(model_path)
        print(f"✅ Modelo cargado: {model.count_params()} parámetros")
        print("✅ Modelo NO está corrupto")
    else:
        print(f"❌ {model_path} no encontrado")

except Exception as e:
    print(f"❌ Error: {e}")
    print("Modelo probablemente corrupto o TensorFlow tiene problemas")
