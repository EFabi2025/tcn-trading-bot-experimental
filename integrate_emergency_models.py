#!/usr/bin/env python3
"""
🔧 INTEGRAR MODELOS TCN CORREGIDOS AL SISTEMA
Reemplazar modelos antiguos con versiones que detectan momentum bajista
"""

import os
import shutil
from pathlib import Path

def integrate_emergency_models():
    """🔧 Integrar modelos de emergencia al sistema principal"""

    print("🔧 INTEGRANDO MODELOS TCN CORREGIDOS AL SISTEMA")
    print("=" * 60)

    # Mapeo de modelos de emergencia a ubicaciones del sistema
    model_mappings = {
        'emergency_btcusdt': 'models/btc',
        'emergency_ethusdt': 'models/eth',
        'emergency_bnbusdt': 'models/bnb'
    }

    backup_dir = 'models/backup_old_models'

    # Crear directorio de backup
    os.makedirs(backup_dir, exist_ok=True)

    for emergency_model, system_location in model_mappings.items():
        print(f"\n🔄 Procesando {emergency_model} -> {system_location}")

        emergency_path = f'models/{emergency_model}'

        # Verificar que el modelo de emergencia existe
        if not os.path.exists(emergency_path):
            print(f"❌ No se encontró {emergency_path}")
            continue

        # Hacer backup del modelo anterior si existe
        if os.path.exists(system_location):
            backup_path = f'{backup_dir}/{os.path.basename(system_location)}_backup'
            print(f"📦 Haciendo backup: {system_location} -> {backup_path}")

            if os.path.exists(backup_path):
                shutil.rmtree(backup_path)
            shutil.copytree(system_location, backup_path)

            # Eliminar modelo anterior
            shutil.rmtree(system_location)

        # Crear directorio del sistema si no existe
        os.makedirs(system_location, exist_ok=True)

        # Copiar modelo de emergencia al sistema
        print(f"🚀 Copiando modelo corregido...")

        # Copiar archivos
        for file_name in ['model.h5', 'scaler.pkl', 'feature_columns.pkl']:
            src_file = f'{emergency_path}/{file_name}'
            dst_file = f'{system_location}/{file_name}'

            if os.path.exists(src_file):
                shutil.copy2(src_file, dst_file)
                print(f"  ✅ {file_name} copiado")
            else:
                print(f"  ⚠️ {file_name} no encontrado")

        print(f"✅ {emergency_model} integrado exitosamente")

    print(f"\n🎯 INTEGRACIÓN COMPLETADA")
    print("=" * 40)
    print("✅ Modelos TCN corregidos integrados al sistema")
    print("✅ Modelos anteriores respaldados en models/backup_old_models/")
    print("💡 El sistema ahora debería detectar mejor el momentum bajista")

    # Verificar integración
    print(f"\n🔍 VERIFICANDO INTEGRACIÓN:")
    for emergency_model, system_location in model_mappings.items():
        if os.path.exists(f'{system_location}/model.h5'):
            print(f"  ✅ {system_location}/model.h5 - OK")
        else:
            print(f"  ❌ {system_location}/model.h5 - FALTA")

if __name__ == "__main__":
    integrate_emergency_models()
