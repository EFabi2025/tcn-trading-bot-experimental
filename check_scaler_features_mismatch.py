#!/usr/bin/env python3
"""
🔍 VERIFICACIÓN DE DISCREPANCIA ENTRE FEATURES DE SCALER Y CALCULADAS
=====================================================================

Script para identificar si hay diferencias entre:
- Features guardadas en los scalers (del entrenamiento)
- Features calculadas actualmente por centralized_features_engine2.py

Este es el problema más probable: los modelos se entrenaron con un conjunto
de features, pero el predictor está calculando un conjunto diferente.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import pickle
from centralized_features_engine3 import CentralizedFeaturesEngine

def check_scaler_features_mismatch():
    """🔍 Verificar discrepancia entre features de scaler y calculadas"""
    
    print("🔍 VERIFICACIÓN DE DISCREPANCIA ENTRE FEATURES DE SCALER Y CALCULADAS")
    print("=" * 70)
    
    # 1. Crear engine de features
    engine = CentralizedFeaturesEngine()
    
    # 2. Obtener features calculadas actualmente
    current_features = engine.feature_sets['tcn_definitivo']
    print(f"\n📊 FEATURES CALCULADAS ACTUALMENTE:")
    print(f"   Total: {len(current_features)}")
    print(f"   Lista: {current_features}")
    
    # 3. Verificar scalers en el directorio models y subdirectorios
    models_dir = "models"
    if not os.path.exists(models_dir):
        print(f"\n❌ Directorio 'models' no encontrado")
        return
    
    print(f"\n🔍 BUSCANDO SCALERS EN {models_dir} Y SUBDIRECTORIOS...")
    
    scaler_files = []
    for root, dirs, files in os.walk(models_dir):
        for file in files:
            if file.endswith('scaler.pkl'):
                scaler_files.append(os.path.join(root, file))
    
    print(f"   Encontrados {len(scaler_files)} archivos de scaler")
    
    # 4. Analizar cada scaler
    all_scaler_features = set()
    scaler_analysis = {}
    
    for scaler_file in scaler_files:
        try:
            print(f"\n📁 Analizando: {scaler_file}")
            
            with open(scaler_file, 'rb') as f:
                scaler_data = pickle.load(f)
            
            # Extraer features del scaler
            if hasattr(scaler_data, 'feature_names_in_'):
                scaler_features = list(scaler_data.feature_names_in_)
            elif hasattr(scaler_data, 'feature_names'):
                scaler_features = list(scaler_data.feature_names)
            else:
                print(f"   ⚠️ No se pueden extraer features del scaler")
                continue
            
            print(f"   Features en scaler: {len(scaler_features)}")
            print(f"   Features: {scaler_features}")
            
            # Comparar con features actuales
            missing_in_scaler = set(current_features) - set(scaler_features)
            extra_in_scaler = set(scaler_features) - set(current_features)
            
            scaler_analysis[scaler_file] = {
                'features': scaler_features,
                'count': len(scaler_features),
                'missing_in_scaler': list(missing_in_scaler),
                'extra_in_scaler': list(extra_in_scaler)
            }
            
            all_scaler_features.update(scaler_features)
            
            if missing_in_scaler:
                print(f"   ❌ Faltantes en scaler: {missing_in_scaler}")
            if extra_in_scaler:
                print(f"   ➕ Extra en scaler: {extra_in_scaler}")
            if not missing_in_scaler and not extra_in_scaler:
                print(f"   ✅ Coincidencia perfecta")
                
        except Exception as e:
            print(f"   ❌ Error analizando {scaler_file}: {e}")
    
    # 5. Resumen general
    print(f"\n📋 RESUMEN GENERAL:")
    print(f"   Features calculadas actualmente: {len(current_features)}")
    print(f"   Features únicas en scalers: {len(all_scaler_features)}")
    
    # 6. Análisis detallado por scaler
    print(f"\n📊 ANÁLISIS DETALLADO POR SCALER:")
    for scaler_file, analysis in scaler_analysis.items():
        print(f"\n   📁 {os.path.basename(os.path.dirname(scaler_file))}/{os.path.basename(scaler_file)}:")
        print(f"      Features: {analysis['count']}")
        print(f"      Faltantes en scaler: {len(analysis['missing_in_scaler'])}")
        print(f"      Extra en scaler: {len(analysis['extra_in_scaler'])}")
        
        if analysis['missing_in_scaler']:
            print(f"      Faltantes: {analysis['missing_in_scaler']}")
        if analysis['extra_in_scaler']:
            print(f"      Extra: {analysis['extra_in_scaler']}")
    
    # 7. Verificar si hay patrones
    print(f"\n🔍 ANÁLISIS DE PATRONES:")
    
    # Features que faltan consistentemente en scalers
    missing_consistently = set(current_features)
    for analysis in scaler_analysis.values():
        missing_consistently = missing_consistently.intersection(set(analysis['missing_in_scaler']))
    
    if missing_consistently:
        print(f"   ❌ Features que faltan consistentemente en TODOS los scalers:")
        for feature in sorted(missing_consistently):
            print(f"      - {feature}")
    
    # Features extra consistentes en scalers
    extra_consistently = set()
    for analysis in scaler_analysis.values():
        extra_consistently = extra_consistently.union(set(analysis['extra_in_scaler']))
    
    if extra_consistently:
        print(f"   ➕ Features extra consistentes en scalers:")
        for feature in sorted(extra_consistently):
            print(f"      - {feature}")
    
    # 8. Recomendaciones
    print(f"\n💡 RECOMENDACIONES:")
    
    if missing_consistently or extra_consistently:
        print(f"   ❌ PROBLEMA IDENTIFICADO: Hay discrepancia entre features de entrenamiento y predicción")
        print(f"   1. Reentrenar modelos con el conjunto de features actual")
        print(f"   2. O modificar centralized_features_engine2.py para usar las features de entrenamiento")
        print(f"   3. Verificar que el entrenamiento use exactamente centralized_features_engine2.py")
    else:
        print(f"   ✅ No se detectaron discrepancias consistentes")
        print(f"   1. Verificar que el predictor esté usando el engine correcto")
        print(f"   2. Verificar que los modelos se carguen correctamente")
    
    return {
        'current_features': current_features,
        'scaler_analysis': scaler_analysis,
        'missing_consistently': list(missing_consistently),
        'extra_consistently': list(extra_consistently)
    }

if __name__ == "__main__":
    result = check_scaler_features_mismatch()
    print(f"\n🎯 RESUMEN FINAL:")
    print(f"   Features actuales: {len(result['current_features'])}")
    print(f"   Scalers analizados: {len(result['scaler_analysis'])}")
    print(f"   Features faltantes consistentemente: {len(result['missing_consistently'])}")
    print(f"   Features extra consistentes: {len(result['extra_consistently'])}") 