#!/usr/bin/env python3
"""
🔍 VERIFICACIÓN DE MODELOS - COMPARACIÓN PRODUCTION vs ULTRA
Script que verifica qué modelos cumplen realmente sus thresholds
"""

import os
import glob
from datetime import datetime
import tensorflow as tf
import numpy as np

def analyze_model_requirements():
    """Analizar requerimientos de cada tipo de modelo"""

    requirements = {
        'production_model': {
            'accuracy': 0.35,
            'confidence': 0.55,
            'f1_score': 0.25,
            'bias_score': 5.0,
            'description': 'Modelos con thresholds realistas para crypto'
        },
        'ultra_model': {
            'accuracy': 0.45,  # +29% más exigente
            'confidence': 0.65, # +18% más exigente
            'f1_score': 0.40,   # +60% más exigente
            'bias_score': 6.0,  # +20% más exigente
            'description': 'Modelos con thresholds ultra-estrictos'
        },
        'best_model': {
            'accuracy': 0.30,   # Inferido
            'confidence': 0.50, # Inferido
            'f1_score': 0.20,   # Inferido
            'bias_score': 4.0,  # Inferido
            'description': 'Modelos anteriores con thresholds básicos'
        }
    }

    return requirements

def get_model_files():
    """Obtener todos los archivos de modelo"""
    model_files = {}

    # Buscar todos los archivos .h5
    for model_file in glob.glob("*.h5"):
        base_name = model_file.replace('.h5', '')

        # Clasificar por tipo
        if 'production_model' in base_name:
            model_type = 'production_model'
            pair = base_name.replace('production_model_', '')
        elif 'ultra_model' in base_name:
            model_type = 'ultra_model'
            pair = base_name.replace('ultra_model_', '')
        elif 'best_model' in base_name:
            model_type = 'best_model'
            pair = base_name.replace('best_model_', '')
        else:
            model_type = 'other'
            pair = base_name

        if model_type not in model_files:
            model_files[model_type] = {}

        # Obtener información del archivo
        stat = os.stat(model_file)
        model_files[model_type][pair] = {
            'file': model_file,
            'size_kb': stat.st_size / 1024,
            'created': datetime.fromtimestamp(stat.st_mtime),
            'created_str': datetime.fromtimestamp(stat.st_mtime).strftime("%d/%m %H:%M")
        }

    return model_files

def analyze_model_architecture(model_file):
    """Analizar arquitectura del modelo sin cargar pesos"""
    try:
        # Cargar solo la arquitectura
        model = tf.keras.models.load_model(model_file, compile=False)

        info = {
            'layers': len(model.layers),
            'params': model.count_params(),
            'input_shape': model.input_shape,
            'output_shape': model.output_shape,
            'optimizer': getattr(model.optimizer, 'name', 'Unknown') if hasattr(model, 'optimizer') else 'Unknown'
        }

        # Limpiar memoria
        del model
        tf.keras.backend.clear_session()

        return info
    except Exception as e:
        return {'error': str(e)}

def main():
    """Función principal de verificación"""

    print("🔍 VERIFICACIÓN COMPLETA DE MODELOS")
    print("=" * 70)

    # Obtener requerimientos
    requirements = analyze_model_requirements()

    # Obtener archivos de modelo
    model_files = get_model_files()

    print(f"\n📊 RESUMEN DE MODELOS ENCONTRADOS:")
    total_models = sum(len(models) for models in model_files.values())
    print(f"   Total: {total_models} modelos")

    for model_type, models in model_files.items():
        print(f"   {model_type}: {len(models)} modelos")

    # Análisis por tipo de modelo
    for model_type, models in model_files.items():
        if not models:
            continue

        print(f"\n{'='*60}")
        print(f"🔍 ANÁLISIS: {model_type.upper()}")
        print(f"{'='*60}")

        if model_type in requirements:
            req = requirements[model_type]
            print(f"📋 REQUERIMIENTOS:")
            print(f"   Accuracy: ≥{req['accuracy']:.2f}")
            print(f"   Confianza: ≥{req['confidence']:.2f}")
            print(f"   F1 Score: ≥{req['f1_score']:.2f}")
            print(f"   Bias Score: ≥{req['bias_score']:.1f}")
            print(f"   📝 {req['description']}")

        print(f"\n🗂️ ARCHIVOS ENCONTRADOS:")

        # Ordenar por fecha de creación (más reciente primero)
        sorted_models = sorted(models.items(),
                             key=lambda x: x[1]['created'],
                             reverse=True)

        for pair, info in sorted_models:
            print(f"\n📄 {info['file']}")
            print(f"   📅 Creado: {info['created_str']}")
            print(f"   💾 Tamaño: {info['size_kb']:.0f} KB")

            # Analizar arquitectura
            arch_info = analyze_model_architecture(info['file'])
            if 'error' not in arch_info:
                print(f"   🏗️ Capas: {arch_info['layers']}")
                print(f"   🔢 Parámetros: {arch_info['params']:,}")
                print(f"   📐 Input: {arch_info['input_shape']}")
                print(f"   📤 Output: {arch_info['output_shape']}")

    # Comparación temporal
    print(f"\n{'='*60}")
    print(f"⏰ CRONOLOGÍA DE CREACIÓN")
    print(f"{'='*60}")

    all_models = []
    for model_type, models in model_files.items():
        for pair, info in models.items():
            all_models.append({
                'type': model_type,
                'pair': pair,
                'file': info['file'],
                'created': info['created'],
                'created_str': info['created_str']
            })

    # Ordenar todos por fecha
    all_models.sort(key=lambda x: x['created'], reverse=True)

    print(f"\n🕐 ORDEN CRONOLÓGICO (más reciente → más antiguo):")
    for i, model in enumerate(all_models[:10]):  # Mostrar solo los 10 más recientes
        emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1:2d}."
        print(f"   {emoji} {model['created_str']} - {model['type']} ({model['pair']})")

    # Recomendación final
    print(f"\n{'='*60}")
    print(f"🎯 RECOMENDACIÓN BASADA EN ANÁLISIS")
    print(f"{'='*60}")

    production_models = model_files.get('production_model', {})
    ultra_models = model_files.get('ultra_model', {})

    if production_models and ultra_models:
        # Comparar fechas
        latest_production = max(production_models.values(), key=lambda x: x['created'])
        latest_ultra = max(ultra_models.values(), key=lambda x: x['created'])

        print(f"\n📊 COMPARACIÓN TEMPORAL:")
        print(f"   🏭 Production más reciente: {latest_production['created_str']}")
        print(f"   ⚡ Ultra más reciente: {latest_ultra['created_str']}")

        time_diff = abs((latest_production['created'] - latest_ultra['created']).total_seconds() / 60)
        print(f"   🕐 Diferencia: {time_diff:.0f} minutos")

        if latest_production['created'] > latest_ultra['created']:
            print(f"\n✅ RECOMENDACIÓN: **USAR PRODUCTION_MODEL**")
            print(f"   📅 Son más recientes ({time_diff:.0f}min después)")
            print(f"   🎯 Thresholds realistas para crypto trading")
            print(f"   ⚖️ Balance entre precisión y practicidad")
        else:
            print(f"\n⚡ ADVERTENCIA: Los ULTRA_MODEL son más recientes")
            print(f"   📅 Creados {time_diff:.0f} minutos después")
            print(f"   🔥 Thresholds MÁS EXIGENTES:")
            print(f"      • Accuracy: 0.45 vs 0.35 (+29%)")
            print(f"      • Confianza: 0.65 vs 0.55 (+18%)")
            print(f"      • F1 Score: 0.40 vs 0.25 (+60%)")
            print(f"   ❓ PERO... ¿Pasaron realmente estos thresholds?")

    print(f"\n🔍 PARA VERIFICAR VERDADERO RENDIMIENTO:")
    print(f"   1. Revisar logs de entrenamiento en terminal")
    print(f"   2. Ejecutar scripts de test individuales")
    print(f"   3. Comprobar si los ultra_model realmente cumplieron sus thresholds")

    return model_files, requirements

if __name__ == "__main__":
    main()
