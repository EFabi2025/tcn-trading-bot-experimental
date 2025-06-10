#!/usr/bin/env python3
"""
🔧 SOLUCIÓN SIMPLE PARA MODELO TCN ANTI-BIAS FIXED
==================================================

Script simple que NO usa TensorFlow para verificar y crear un plan de acción.
"""

import os
import json
import shutil
from datetime import datetime

def check_files():
    """Verificar archivos relacionados al modelo"""
    print("🔍 VERIFICANDO ARCHIVOS DEL SISTEMA...")
    print("=" * 40)
    
    files_to_check = {
        'tcn_anti_bias_fixed.h5': 'Modelo principal',
        'feature_scalers_fixed.pkl': 'Scalers de features',
        'models/feature_scalers_fixed.pkl': 'Scalers en directorio models',
        'src/models/train_anti_bias_tcn_fixed.py': 'Script de entrenamiento original',
        'src/models/regime_classifier.py': 'Clasificador de regímenes',
        'src/models/tcn_features_engineering.py': 'Feature engineering',
        'src/models/tcn_anti_bias_model.py': 'Modelo TCN'
    }
    
    status = {}
    
    for file_path, description in files_to_check.items():
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            status[file_path] = {
                'exists': True,
                'size': size,
                'size_mb': size / (1024 * 1024),
                'description': description
            }
            print(f"✅ {file_path}: {size/1024:.1f} KB - {description}")
        else:
            status[file_path] = {
                'exists': False,
                'description': description
            }
            print(f"❌ {file_path}: No encontrado - {description}")
    
    return status

def create_backup():
    """Crear backup del modelo actual"""
    print("\n💾 CREANDO BACKUP...")
    print("=" * 20)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    files_to_backup = [
        'tcn_anti_bias_fixed.h5',
        'feature_scalers_fixed.pkl'
    ]
    
    backup_dir = f"backup_model_{timestamp}"
    os.makedirs(backup_dir, exist_ok=True)
    
    for file_path in files_to_backup:
        if os.path.exists(file_path):
            backup_path = os.path.join(backup_dir, file_path)
            os.makedirs(os.path.dirname(backup_path), exist_ok=True)
            shutil.copy2(file_path, backup_path)
            print(f"✅ {file_path} → {backup_path}")
        else:
            print(f"⚠️ {file_path}: No encontrado para backup")
    
    print(f"💾 Backup completado en: {backup_dir}")
    return backup_dir

def test_h5_file():
    """Probar si el archivo H5 está corrupto"""
    print("\n🧪 PROBANDO ARCHIVO H5...")
    print("=" * 25)
    
    model_path = 'tcn_anti_bias_fixed.h5'
    
    if not os.path.exists(model_path):
        print(f"❌ Archivo no existe: {model_path}")
        return False
    
    try:
        # Intentar abrir como archivo HDF5
        import h5py
        
        with h5py.File(model_path, 'r') as f:
            print("✅ Archivo HDF5 válido")
            print(f"   Grupos principales: {list(f.keys())}")
            
            # Verificar estructura básica de Keras
            if 'model_weights' in f.keys():
                print("✅ Estructura de modelo Keras detectada")
                return True
            else:
                print("⚠️ Estructura de modelo no estándar")
                return False
                
    except ImportError:
        print("⚠️ h5py no disponible - instalando...")
        os.system("pip install h5py")
        return test_h5_file()  # Reintentar
        
    except Exception as e:
        print(f"❌ Archivo H5 corrupto: {e}")
        return False

def create_tensorflow_test():
    """Crear script de prueba de TensorFlow"""
    test_script = '''#!/usr/bin/env python3
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
'''
    
    with open('test_tensorflow.py', 'w') as f:
        f.write(test_script)
    
    print("📝 Script de prueba creado: test_tensorflow.py")
    return 'test_tensorflow.py'

def create_action_plan(file_status, h5_valid):
    """Crear plan de acción basado en el diagnóstico"""
    print("\n📋 PLAN DE ACCIÓN RECOMENDADO")
    print("=" * 30)
    
    plan = []
    
    # Verificar dependencias del entrenador
    trainer_exists = file_status.get('src/models/train_anti_bias_tcn_fixed.py', {}).get('exists', False)
    
    if not trainer_exists:
        plan.append("❌ CRÍTICO: Script de entrenamiento original no encontrado")
        plan.append("   Solución: Restaurar src/models/train_anti_bias_tcn_fixed.py")
    
    if not h5_valid:
        plan.append("❌ Modelo H5 corrupto confirmado")
        plan.append("   Solución: Reentrenamiento obligatorio")
    else:
        plan.append("✅ Archivo H5 parece válido")
        plan.append("   Problema probablemente es TensorFlow")
    
    # Plan específico
    if h5_valid and trainer_exists:
        plan.append("\n🔧 PASOS RECOMENDADOS:")
        plan.append("1. Ejecutar: python test_tensorflow.py")
        plan.append("2. Si TensorFlow falla: pip uninstall tensorflow && pip install tensorflow-macos")
        plan.append("3. Si modelo se carga: No es necesario reentrenar")
        plan.append("4. Si modelo falla: Ejecutar reentrenamiento")
    else:
        plan.append("\n🚨 PASOS CRÍTICOS:")
        plan.append("1. Restaurar archivos faltantes del sistema")
        plan.append("2. Arreglar TensorFlow")
        plan.append("3. Ejecutar reentrenamiento completo")
    
    for step in plan:
        print(step)
    
    return plan

def main():
    """Función principal del diagnóstico"""
    print("🔧 DIAGNÓSTICO SIMPLE - MODELO TCN ANTI-BIAS FIXED")
    print("=" * 60)
    print("Diagnóstico SIN TensorFlow para evitar colgadas")
    print("")
    
    # 1. Verificar archivos
    file_status = check_files()
    
    # 2. Crear backup
    backup_dir = create_backup()
    
    # 3. Probar archivo H5
    h5_valid = test_h5_file()
    
    # 4. Crear script de prueba TensorFlow
    test_script = create_tensorflow_test()
    
    # 5. Crear plan de acción
    plan = create_action_plan(file_status, h5_valid)
    
    # 6. Resumen final
    print("\n🎯 RESUMEN EJECUTIVO")
    print("=" * 20)
    print(f"📂 Backup creado: {backup_dir}")
    print(f"🧪 Script de prueba: {test_script}")
    print(f"📊 Archivo H5 válido: {'Sí' if h5_valid else 'No'}")
    
    if h5_valid:
        print("\n✅ BUENAS NOTICIAS:")
        print("   El modelo parece no estar corrupto")
        print("   El problema probablemente es TensorFlow")
        print("   Ejecuta: python test_tensorflow.py")
    else:
        print("\n⚠️ CONFIRMADO:")
        print("   El modelo ESTÁ corrupto")
        print("   Requiere reentrenamiento completo")
    
    print(f"\nPróximo paso: python {test_script}")

if __name__ == "__main__":
    main()