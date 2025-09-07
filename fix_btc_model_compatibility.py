#!/usr/bin/env python3
"""
🔧 SOLUCIÓN PARA COMPATIBILIDAD MODELO BTCUSDT
==============================================

Identifica y resuelve la incompatibilidad entre las 88 features calculadas
y las 84 features esperadas por el modelo BTCUSDT.
"""

import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from centralized_features_engine3 import CentralizedFeaturesEngine

def load_model_feature_columns(model_dir):
    """Cargar las columnas de features del modelo entrenado"""
    model_path = Path(model_dir)
    
    # Buscar archivo de columnas
    possible_files = [
        'feature_columns.pkl',
        'features.pkl',
        'feature_names.pkl'
    ]
    
    for filename in possible_files:
        file_path = model_path / filename
        if file_path.exists():
            print(f"✅ Cargando features desde: {filename}")
            with open(file_path, 'rb') as f:
                return pickle.load(f)
    
    print("⚠️ No se encontró archivo de features del modelo")
    return None

def identify_compatibility_issues():
    """Identificar el problema de compatibilidad"""
    print("🔍 IDENTIFICANDO PROBLEMA DE COMPATIBILIDAD")
    print("=" * 50)
    
    # Motor centralizado actual
    engine = CentralizedFeaturesEngine()
    current_features = engine.feature_sets['tcn_definitivo']
    print(f"📊 Features actuales (motor): {len(current_features)}")
    
    # Features del modelo entrenado
    model_dir = "models/adaptive_btcusdt_5m_6h_48w_tcn_definitivo"
    model_features = load_model_feature_columns(model_dir)
    
    if model_features is not None:
        print(f"📊 Features del modelo: {len(model_features)}")
        
        # Comparar
        current_set = set(current_features)
        model_set = set(model_features)
        
        extra_in_current = current_set - model_set
        missing_in_current = model_set - current_set
        
        print(f"\n🔍 ANÁLISIS DE DIFERENCIAS:")
        print(f"   ➕ Features extra en motor actual: {len(extra_in_current)}")
        print(f"   ❌ Features faltantes en motor actual: {len(missing_in_current)}")
        
        if extra_in_current:
            print(f"\n➕ FEATURES EXTRA (estas causan el problema):")
            for i, feature in enumerate(sorted(extra_in_current), 1):
                print(f"   {i}. {feature}")
                
        if missing_in_current:
            print(f"\n❌ FEATURES FALTANTES:")
            for i, feature in enumerate(sorted(missing_in_current), 1):
                print(f"   {i}. {feature}")
        
        return {
            'current_features': current_features,
            'model_features': model_features,
            'extra_features': list(extra_in_current),
            'missing_features': list(missing_in_current)
        }
    else:
        print("❌ No se pudo cargar features del modelo")
        return None

def create_compatibility_solution(analysis):
    """Crear solución de compatibilidad"""
    if not analysis:
        return None
        
    print(f"\n🔧 CREANDO SOLUCIÓN DE COMPATIBILIDAD")
    print("=" * 40)
    
    extra_features = analysis['extra_features']
    model_features = analysis['model_features']
    
    if len(extra_features) == 4:
        print("✅ Problema identificado: 4 features extra")
        print("🔧 Solución: Filtrar features para usar solo las del modelo")
        
        # Crear función de filtro
        filter_code = f"""
def fix_features_for_btcusdt_model(features_df):
    \"\"\"
    Filtrar features para compatibilidad con modelo BTCUSDT
    
    El modelo espera {len(model_features)} features, pero el motor calcula {len(analysis['current_features'])}.
    Esta función remueve las {len(extra_features)} features extra.
    \"\"\"
    
    # Features que el modelo espera (orden original del entrenamiento)
    expected_features = {model_features}
    
    # Filtrar solo las features que el modelo conoce
    available_features = [col for col in expected_features if col in features_df.columns]
    
    if len(available_features) != {len(model_features)}:
        missing = set(expected_features) - set(available_features)
        print(f"⚠️ Features faltantes: {{missing}}")
    
    # Retornar DataFrame filtrado en el orden correcto
    return features_df[available_features]

# Features extra que deben ser removidas:
# {extra_features}
"""
        
        return filter_code
    else:
        print(f"⚠️ Problema diferente: {len(extra_features)} features extra")
        return None

def create_patched_engine():
    """Crear motor de features con parche para BTCUSDT"""
    print(f"\n🔧 CREANDO MOTOR CON PARCHE PARA BTCUSDT")
    print("=" * 40)
    
    analysis = identify_compatibility_issues()
    if not analysis:
        return
        
    # Crear clase con parche
    patch_code = f'''
class PatchedFeaturesEngine(CentralizedFeaturesEngine):
    """Motor de features con parche para compatibilidad BTCUSDT"""
    
    def __init__(self):
        super().__init__()
        
        # Features originales del modelo BTCUSDT entrenado
        self.btcusdt_model_features = {analysis['model_features']}
        
    def calculate_features_for_btcusdt(self, df):
        """Calcular features compatibles con modelo BTCUSDT"""
        
        # Calcular todas las features normalmente
        all_features = self.calculate_features(df, feature_set='tcn_definitivo')
        
        # Filtrar solo las que el modelo conoce
        available_features = [col for col in self.btcusdt_model_features 
                            if col in all_features.columns]
        
        print(f"🔧 Filtrando features para BTCUSDT: {{len(all_features.columns)}} → {{len(available_features)}}")
        
        if len(available_features) != {len(analysis['model_features'])}:
            missing = set(self.btcusdt_model_features) - set(available_features)
            print(f"⚠️ Features faltantes: {{missing}}")
        
        return all_features[available_features]
'''
    
    return patch_code

def test_solution():
    """Test de la solución"""
    print(f"\n🧪 TESTING SOLUCIÓN")
    print("=" * 30)
    
    analysis = identify_compatibility_issues()
    if not analysis:
        return
        
    # Generar datos de prueba
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=100, freq='5T')
    
    base_price = 45000
    returns = np.random.normal(0, 0.02, 100)
    prices = base_price * np.exp(np.cumsum(returns))
    
    test_data = pd.DataFrame({
        'timestamp': dates,
        'open': prices * (1 + np.random.normal(0, 0.005, 100)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.005, 100))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.005, 100))),
        'close': prices,
        'volume': np.random.lognormal(16, 0.3, 100)
    })
    
    test_data['high'] = np.maximum.reduce([test_data['open'], test_data['high'], test_data['low'], test_data['close']])
    test_data['low'] = np.minimum.reduce([test_data['open'], test_data['high'], test_data['low'], test_data['close']])
    
    # Test con motor original
    engine = CentralizedFeaturesEngine()
    all_features = engine.calculate_features(test_data, feature_set='tcn_definitivo')
    print(f"📊 Motor original: {all_features.shape}")
    
    # Test con filtro
    model_features = analysis['model_features']
    available_features = [col for col in model_features if col in all_features.columns]
    filtered_features = all_features[available_features]
    print(f"📊 Features filtradas: {filtered_features.shape}")
    
    # Verificar resultado
    expected_count = len(analysis['model_features'])
    actual_count = filtered_features.shape[1]
    
    if actual_count == expected_count:
        print("✅ SOLUCIÓN FUNCIONA CORRECTAMENTE")
        print(f"   🎯 Esperadas: {expected_count}")
        print(f"   ✅ Obtenidas: {actual_count}")
        return True
    else:
        print("❌ SOLUCIÓN NO FUNCIONA")
        print(f"   🎯 Esperadas: {expected_count}")
        print(f"   ❌ Obtenidas: {actual_count}")
        return False

def main():
    """Función principal"""
    print("🚀 SOLUCIONANDO INCOMPATIBILIDAD MODELO BTCUSDT")
    print("=" * 60)
    
    # 1. Identificar problema
    analysis = identify_compatibility_issues()
    
    if analysis:
        # 2. Crear solución
        solution_code = create_compatibility_solution(analysis)
        
        if solution_code:
            # 3. Crear motor con parche
            patch_code = create_patched_engine()
            
            # 4. Test
            success = test_solution()
            
            # 5. Guardar soluciones
            if success:
                with open('btcusdt_compatibility_fix.py', 'w') as f:
                    f.write(solution_code)
                    
                with open('patched_features_engine.py', 'w') as f:
                    f.write("from centralized_features_engine2 import CentralizedFeaturesEngine\n\n")
                    f.write(patch_code)
                
                print(f"\n💾 ARCHIVOS GENERADOS:")
                print(f"   📄 btcusdt_compatibility_fix.py")
                print(f"   📄 patched_features_engine.py")
                
                print(f"\n✅ PROBLEMA RESUELTO")
                print(f"🔧 Usa el motor con parche para compatibilidad con BTCUSDT")
            else:
                print(f"\n❌ SOLUCIÓN FALLÓ")
        else:
            print(f"\n❌ NO SE PUDO CREAR SOLUCIÓN")
    else:
        print(f"\n❌ NO SE PUDO IDENTIFICAR EL PROBLEMA")

if __name__ == "__main__":
    main()
