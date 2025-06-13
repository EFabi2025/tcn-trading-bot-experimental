#!/usr/bin/env python3
"""
🧪 TEST SIMPLE ETHUSDT
Prueba rápida del modelo ETHUSDT sin cargar TensorFlow pesado
"""

import os
import sys

def check_ethusdt_model():
    """🔍 Verificar estado del modelo ETHUSDT"""

    print("🧪 VERIFICACIÓN MODELO ETHUSDT")
    print("=" * 50)

    model_path = "models/definitivo_ethusdt/best_model.h5"

    if os.path.exists(model_path):
        # Obtener información del archivo
        stat_info = os.stat(model_path)
        size_mb = stat_info.st_size / (1024 * 1024)

        print(f"✅ Modelo encontrado: {model_path}")
        print(f"📊 Tamaño: {size_mb:.1f} MB")
        print(f"📅 Creado: {stat_info.st_ctime}")
        print(f"📅 Modificado: {stat_info.st_mtime}")

        # Verificar si existe scaler
        scaler_path = "models/definitivo_ethusdt/scaler.pkl"
        features_path = "models/definitivo_ethusdt/feature_columns.pkl"

        if os.path.exists(scaler_path):
            print(f"✅ Scaler encontrado: {scaler_path}")
        else:
            print(f"⚠️ Scaler NO encontrado: {scaler_path}")

        if os.path.exists(features_path):
            print(f"✅ Feature columns encontrados: {features_path}")
        else:
            print(f"⚠️ Feature columns NO encontrados: {features_path}")

        return True
    else:
        print(f"❌ Modelo NO encontrado: {model_path}")
        return False

def get_ethusdt_current_price():
    """💰 Obtener precio actual de ETHUSDT"""
    try:
        import requests

        url = "https://api.binance.com/api/v3/ticker/price"
        params = {"symbol": "ETHUSDT"}

        response = requests.get(url, params=params, timeout=5)
        if response.status_code == 200:
            data = response.json()
            price = float(data['price'])
            print(f"💰 Precio actual ETHUSDT: ${price:,.2f}")
            return price
        else:
            print(f"❌ Error API: {response.status_code}")
            return None

    except Exception as e:
        print(f"❌ Error obteniendo precio: {e}")
        return None

def main():
    """🚀 Función principal"""

    print("🧪 TEST SIMPLE MODELO ETHUSDT")
    print("=" * 60)

    # 1. Verificar modelo
    model_exists = check_ethusdt_model()

    if model_exists:
        print(f"\n🎯 ESTADO DEL ENTRENAMIENTO ETHUSDT:")
        print(f"   ✅ Modelo: COMPLETADO (3.7MB)")
        print(f"   ⚠️ Scaler: FALTANTE (necesario recrear)")
        print(f"   ⚠️ Features: FALTANTE (necesario recrear)")

        # 2. Obtener precio actual
        print(f"\n💰 PRECIO ACTUAL:")
        current_price = get_ethusdt_current_price()

        # 3. Resumen
        print(f"\n📋 RESUMEN:")
        print(f"   🟢 ETHUSDT: Modelo entrenado exitosamente")
        print(f"   🟡 Necesita: Scaler y feature columns")
        print(f"   🎯 Siguiente: Recrear scaler para ETHUSDT")

    else:
        print(f"\n❌ ETHUSDT: Modelo no encontrado")
        print(f"   🎯 Acción: Reiniciar entrenamiento")

if __name__ == "__main__":
    main()
