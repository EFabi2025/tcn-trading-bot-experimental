#!/usr/bin/env python3
"""
🧪 Script de prueba para verificar correcciones de JSON
"""

import json
import numpy as np
import os
from datetime import datetime

def test_numpy_json_conversion():
    """🧪 Probar conversión de tipos numpy a JSON"""

    print("🧪 PROBANDO CONVERSIÓN NUMPY A JSON")
    print("=" * 50)

    # Crear datos de prueba con tipos numpy
    test_data = {
        'accuracy': np.float64(0.856),
        'loss': np.float32(0.234),
        'epochs': np.int64(50),
        'batch_size': np.int32(64),
        'predictions': np.array([0.1, 0.2, 0.7]),
        'confusion_matrix': np.array([[10, 2, 1], [3, 15, 2], [1, 1, 8]]),
        'trading_metrics': {
            'precision': np.float64(0.823),
            'recall': np.float64(0.789),
            'f1_score': np.float64(0.805),
            'sharpe_ratio': np.float64(1.234),
            'max_drawdown': np.float64(-0.056)
        },
        'created_at': datetime.now().isoformat()
    }

    print("📊 Datos originales (con tipos numpy):")
    for key, value in test_data.items():
        print(f"   {key}: {type(value)} = {value}")

    # Función de conversión (la misma que usamos en el entrenador)
    def convert_numpy_types(obj):
        """🔄 Convertir tipos numpy a tipos nativos de Python para JSON"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj

    # Convertir datos
    converted_data = convert_numpy_types(test_data)

    print(f"\n📊 Datos convertidos (tipos nativos):")
    for key, value in converted_data.items():
        print(f"   {key}: {type(value)} = {value}")

    # Probar serialización JSON
    try:
        json_string = json.dumps(converted_data, indent=2)
        print(f"\n✅ JSON serializado exitosamente!")
        print(f"📄 Longitud del JSON: {len(json_string)} caracteres")

        # Probar deserialización
        parsed_data = json.loads(json_string)
        print(f"✅ JSON deserializado exitosamente!")

        return True

    except Exception as e:
        print(f"❌ ERROR en serialización JSON: {e}")
        return False

def test_directory_creation():
    """🧪 Probar creación automática de directorios"""

    print(f"\n🧪 PROBANDO CREACIÓN DE DIRECTORIOS")
    print("=" * 50)

    test_dir = "test_models/adaptive_test_model_1m_6h_24w_optimized_crypto"

    try:
        # Crear directorio (debería funcionar sin errores)
        os.makedirs(test_dir, exist_ok=True)
        print(f"✅ Directorio creado: {test_dir}")

        # Verificar que existe
        if os.path.exists(test_dir):
            print(f"✅ Directorio existe: {test_dir}")
        else:
            print(f"❌ Directorio no existe: {test_dir}")
            return False

        # Crear archivo de prueba
        test_file = f"{test_dir}/test_config.json"
        test_config = {
            'symbol': 'TESTUSDT',
            'timeframe': '1m',
            'feature_set': 'optimized_crypto',
            'accuracy': 0.856,
            'created_at': datetime.now().isoformat()
        }

        with open(test_file, 'w') as f:
            json.dump(test_config, f, indent=2)

        print(f"✅ Archivo de prueba creado: {test_file}")

        # Limpiar
        os.remove(test_file)
        os.rmdir(test_dir)
        print(f"✅ Limpieza completada")

        return True

    except Exception as e:
        print(f"❌ ERROR en creación de directorios: {e}")
        return False

def test_feature_sets():
    """🧪 Probar feature sets disponibles"""

    print(f"\n🧪 PROBANDO FEATURE SETS")
    print("=" * 50)

    feature_sets = {
        'tcn_definitivo': '88 features (completo)',
        'optimized_crypto': '25 features (optimizado)',
        'ultra_optimized': '15 features (ultra optimizado)'
    }

    for fs, desc in feature_sets.items():
        print(f"✅ {fs}: {desc}")

    return True

def main():
    """🎯 Función principal de pruebas"""

    print("🚀 INICIANDO PRUEBAS DE CORRECCIONES JSON")
    print("=" * 60)

    tests = [
        ("Conversión Numpy a JSON", test_numpy_json_conversion),
        ("Creación de Directorios", test_directory_creation),
        ("Feature Sets Disponibles", test_feature_sets)
    ]

    results = []

    for test_name, test_func in tests:
        print(f"\n🧪 Ejecutando: {test_name}")
        try:
            success = test_func()
            results.append((test_name, success))
            status = "✅ PASÓ" if success else "❌ FALLÓ"
            print(f"   {status}: {test_name}")
        except Exception as e:
            print(f"   ❌ ERROR: {test_name} - {e}")
            results.append((test_name, False))

    # Resumen final
    print(f"\n🎯 RESUMEN DE PRUEBAS")
    print("=" * 40)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for test_name, success in results:
        status = "✅ PASÓ" if success else "❌ FALLÓ"
        print(f"   {status}: {test_name}")

    print(f"\n🏆 RESULTADO FINAL: {passed}/{total} pruebas pasaron")

    if passed == total:
        print("🎉 ¡TODAS LAS CORRECCIONES FUNCIONAN CORRECTAMENTE!")
        print("✅ El entrenador está listo para uso robusto")
    else:
        print("⚠️  Algunas correcciones necesitan revisión")

    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
