#!/usr/bin/env python3
"""
🔧 PARCHE SIMPLE PARA COMPATIBILIDAD BTCUSDT
===========================================

Solución rápida para el problema de features incompatibles.
"""

def fix_features_for_btcusdt_model(features_df):
    """
    Filtrar features para compatibilidad con modelo BTCUSDT
    
    El modelo espera 84 features, pero el motor calcula 88.
    Esta función remueve las 4 features extra específicas.
    """
    
    # Features extra que deben ser removidas (agregadas después del entrenamiento del modelo)
    features_to_remove = [
        'higher_high',
        'lower_low', 
        'resistance_touch',
        'support_touch'
    ]
    
    # Remover solo las que existen en el DataFrame
    features_to_remove_existing = [f for f in features_to_remove if f in features_df.columns]
    
    if features_to_remove_existing:
        print(f"🔧 Removiendo {len(features_to_remove_existing)} features para compatibilidad BTCUSDT:")
        for feature in features_to_remove_existing:
            print(f"   - {feature}")
        
        filtered_df = features_df.drop(columns=features_to_remove_existing)
        print(f"📊 Features: {features_df.shape[1]} → {filtered_df.shape[1]}")
        return filtered_df
    else:
        print("✅ No se requiere filtrado")
        return features_df

# Test de la función
if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    from centralized_features_engine3 import CentralizedFeaturesEngine
    
    print("🧪 TESTING PARCHE DE COMPATIBILIDAD")
    print("=" * 40)
    
    # Generar datos de prueba
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=50, freq='5T')
    
    test_data = pd.DataFrame({
        'timestamp': dates,
        'open': 45000 + np.random.normal(0, 1000, 50),
        'high': 46000 + np.random.normal(0, 1000, 50),
        'low': 44000 + np.random.normal(0, 1000, 50),
        'close': 45000 + np.random.normal(0, 1000, 50),
        'volume': np.random.lognormal(16, 0.3, 50)
    })
    
    # Calcular features con motor centralizado
    engine = CentralizedFeaturesEngine()
    all_features = engine.calculate_features(test_data, feature_set='tcn_definitivo')
    
    print(f"📊 Features originales: {all_features.shape}")
    
    # Aplicar parche
    fixed_features = fix_features_for_btcusdt_model(all_features)
    
    print(f"📊 Features después del parche: {fixed_features.shape}")
    
    if fixed_features.shape[1] == 84:
        print("✅ PARCHE FUNCIONA CORRECTAMENTE - 84 features obtenidas")
    else:
        print(f"❌ PARCHE FALLÓ - {fixed_features.shape[1]} features en lugar de 84")
