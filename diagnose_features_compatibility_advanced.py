#!/usr/bin/env python3
"""
🔍 DIAGNÓSTICO AVANZADO DE INCOMPATIBILIDAD DE FEATURES
====================================================

Script para identificar las 4 features FALTANTES que causan el problema dimensional
Motor debería calcular 88, pero esta PC solo calcula 84

Creado específicamente para resolver problema PC Windows con features inconsistentes.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import json
import os
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Importar el motor centralizado
from centralized_features_engine3 import CentralizedFeaturesEngine

class FeatureCompatibilityDiagnostic:
    """Diagnóstico avanzado de compatibilidad de features"""
    
    def __init__(self):
        """Inicializar diagnóstico"""
        self.features_engine = CentralizedFeaturesEngine()
        self.models_dir = Path("models")
        self.btc_model_path = None
        
        # Buscar modelo de BTCUSDT-5m específico
        self._find_btc_model()
        
    def _find_btc_model(self):
        """Encontrar el modelo de BTCUSDT-5m que falla"""
        print("🔍 Buscando modelo de BTCUSDT-5m...")
        
        # Buscar modelos que contengan btcusdt y 5m en el nombre
        btc_patterns = [
            "*btcusdt*5m*",
            "*adaptive_btcusdt*5m*",
            "*tcn_definitivo*btcusdt*"
        ]
        
        for pattern in btc_patterns:
            for model_dir in self.models_dir.glob(pattern):
                if model_dir.is_dir():
                    model_file = model_dir / "best_model.h5"
                    if model_file.exists():
                        self.btc_model_path = model_dir
                        print(f"✅ Modelo encontrado: {model_dir}")
                        return
                        
        print("⚠️ No se encontró modelo específico, usando primer modelo de BTCUSDT disponible")
        for model_dir in self.models_dir.glob("*btcusdt*"):
            if model_dir.is_dir():
                model_file = model_dir / "best_model.h5"
                if model_file.exists():
                    self.btc_model_path = model_dir
                    print(f"✅ Modelo encontrado: {model_dir}")
                    return
                    
    def generate_test_data(self, periods: int = 200) -> pd.DataFrame:
        """Generar datos de prueba para BTCUSDT"""
        print(f"📊 Generando {periods} periodos de datos de prueba para BTCUSDT...")
        
        # Datos realistas para BTCUSDT
        dates = pd.date_range(start='2024-01-01', periods=periods, freq='5T')
        np.random.seed(42)  # Semilla fija para consistencia
        
        base_price = 45000
        volatility = 0.02
        
        # Simulación de caminata aleatoria con tendencia
        returns = np.random.normal(0, volatility, periods)
        prices = base_price * np.exp(np.cumsum(returns))
        
        # Generar OHLCV realista
        noise_factor = 0.005
        high_noise = np.abs(np.random.normal(0, noise_factor, periods))
        low_noise = np.abs(np.random.normal(0, noise_factor, periods))
        
        df = pd.DataFrame({
            'timestamp': dates,
            'open': prices * (1 + np.random.normal(0, noise_factor, periods)),
            'high': prices * (1 + high_noise),
            'low': prices * (1 - low_noise),
            'close': prices,
            'volume': np.random.lognormal(16, 0.3, periods)  # Volúmenes realistas
        })
        
        # Asegurar consistencia OHLC
        df['high'] = np.maximum.reduce([df['open'], df['high'], df['low'], df['close']])
        df['low'] = np.minimum.reduce([df['open'], df['high'], df['low'], df['close']])
        
        print("✅ Datos de prueba generados")
        return df
        
    def analyze_model_requirements(self) -> Dict:
        """Analizar qué features espera el modelo"""
        if not self.btc_model_path:
            print("❌ No se encontró modelo de BTCUSDT")
            return {}
            
        print(f"🔍 Analizando modelo: {self.btc_model_path}")
        
        model_info = {}
        
        try:
            # Cargar modelo
            model_file = self.btc_model_path / "best_model.h5"
            if not model_file.exists():
                model_file = self.btc_model_path / "model.h5"
                
            model = tf.keras.models.load_model(model_file)
            
            # Obtener información de input
            input_shape = model.input_shape
            model_info['input_shape'] = input_shape
            model_info['expected_features'] = input_shape[-1] if input_shape else None
            model_info['expected_sequence_length'] = input_shape[1] if len(input_shape) > 2 else None
            
            print(f"📊 Input shape del modelo: {input_shape}")
            print(f"📊 Features esperadas: {model_info['expected_features']}")
            print(f"📊 Longitud de secuencia: {model_info['expected_sequence_length']}")
            
            # Cargar configuración si existe
            config_file = self.btc_model_path / "config.json"
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    model_info['config'] = config
                    print(f"📋 Config cargada: feature_set = {config.get('feature_set', 'NO ESPECIFICADO')}")
            else:
                print("⚠️ No se encontró config.json")
                
        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
            
        return model_info
        
    def calculate_current_features(self) -> Tuple[pd.DataFrame, List[str]]:
        """Calcular features actuales con motor centralizado"""
        print("🔄 Calculando features con motor centralizado...")
        
        # Generar datos de prueba
        test_data = self.generate_test_data()
        
        # Calcular features con tcn_definitivo
        features_df = self.features_engine.calculate_features(test_data, feature_set='tcn_definitivo')
        feature_names = list(features_df.columns)
        
        print(f"📊 Features calculadas: {len(feature_names)}")
        print(f"📊 Shape resultante: {features_df.shape}")
        
        return features_df, feature_names
        
    def identify_missing_features(self, expected_count: int, actual_features: List[str]) -> Dict:
        """Identificar qué features están faltando"""
        actual_count = len(actual_features)
        missing_count = expected_count - actual_count
        
        print(f"\n🔍 ANÁLISIS DE FEATURES FALTANTES:")
        print(f"   📊 Esperadas: {expected_count}")
        print(f"   📊 Calculadas: {actual_count}")
        print(f"   ❌ Faltantes: {missing_count}")
        
        if missing_count <= 0:
            return {'status': 'ok', 'missing_features': []}
            
        # Obtener features del conjunto tcn_definitivo esperado
        expected_features = self.features_engine.feature_sets['tcn_definitivo']
        
        # Comparar con features realmente calculadas
        missing_from_calculation = set(expected_features) - set(actual_features)
        extra_in_calculation = set(actual_features) - set(expected_features)
        
        analysis = {
            'status': 'mismatch',
            'expected_count': expected_count,
            'actual_count': actual_count,
            'missing_count': missing_count,
            'expected_features': expected_features,
            'actual_features': actual_features,
            'missing_from_calculation': list(missing_from_calculation),
            'extra_in_calculation': list(extra_in_calculation),
            'missing_features_identified': []
        }
        
        # Analizar features faltantes
        print(f"\n📋 FEATURES FALTANTES EN CÁLCULO: {len(missing_from_calculation)}")
        for feature in missing_from_calculation:
            print(f"   ❌ {feature}")
            
        print(f"\n📋 FEATURES EXTRA EN CÁLCULO: {len(extra_in_calculation)}")
        for feature in extra_in_calculation:
            print(f"   ➕ {feature}")
            
        # Si hay exactamente 4 features faltantes, identificarlas específicamente
        if missing_count == 4:
            print(f"\n🎯 IDENTIFICANDO LAS 4 FEATURES FALTANTES...")
            
            # Las features que faltan son exactamente las que están en expected pero no en actual
            missing_features_list = list(missing_from_calculation)
            
            analysis['missing_features_identified'] = missing_features_list
            
            print(f"🔍 Features faltantes identificadas:")
            for i, feature in enumerate(missing_features_list, 1):
                print(f"   {i}. {feature}")
                
            # Analizar por categorías las features faltantes
            if missing_features_list:
                print(f"\n📊 ANÁLISIS POR CATEGORÍAS:")
                categories = {
                    'volatility': [f for f in missing_features_list if 'volatility' in f],
                    'momentum': [f for f in missing_features_list if 'momentum' in f],
                    'price': [f for f in missing_features_list if 'price' in f],
                    'technical': [f for f in missing_features_list if any(x in f for x in ['rsi', 'macd', 'bb', 'sma', 'ema'])],
                    'volume': [f for f in missing_features_list if 'volume' in f or 'ad' in f or 'obv' in f],
                    'other': []
                }
                
                # Clasificar features no categorizadas
                categorized = []
                for cat_features in categories.values():
                    categorized.extend(cat_features)
                categories['other'] = [f for f in missing_features_list if f not in categorized]
                
                for category, features in categories.items():
                    if features:
                        print(f"   📈 {category.upper()}: {len(features)} features")
                        for feature in features:
                            print(f"      - {feature}")
                
        return analysis
        
    def suggest_solutions(self, analysis: Dict) -> List[str]:
        """Sugerir soluciones al problema"""
        if analysis['status'] == 'ok':
            return ["✅ No hay problemas de compatibilidad"]
            
        solutions = []
        
        if analysis['missing_count'] == 4:
            solutions.extend([
                "🔧 SOLUCIÓN 1: Verificar instalación y versión de TA-Lib",
                "🔧 SOLUCIÓN 2: Verificar que todas las dependencias estén disponibles",
                "🔧 SOLUCIÓN 3: Revisar errores silenciosos en el cálculo de features",
                "🔧 SOLUCIÓN 4: Comparar con PC funcionando para identificar diferencias",
                "🔧 SOLUCIÓN 5: Implementar fallback para features faltantes"
            ])
            
        if analysis['missing_from_calculation']:
            missing_features = analysis['missing_from_calculation']
            solutions.append(f"🔧 PRIORITARIO: Asegurar cálculo de features faltantes: {missing_features}")
            
            # Análisis específico por tipo de feature faltante
            if any('volatility' in f for f in missing_features):
                solutions.append("🔧 ESPECÍFICO: Revisar cálculo de features de volatilidad")
                
            if any('momentum' in f for f in missing_features):
                solutions.append("🔧 ESPECÍFICO: Revisar cálculo de features de momentum")
                
            if any(x in str(missing_features) for x in ['rsi', 'macd', 'bb']):
                solutions.append("🔧 ESPECÍFICO: Verificar TA-Lib instalación y funcionamiento")
            
        if analysis['extra_in_calculation']:
            solutions.append(f"🔧 BONUS: Remover features extra no esperadas: {analysis['extra_in_calculation']}")
            
        return solutions
        
    def create_compatibility_fix(self, analysis: Dict) -> str:
        """Crear un parche de compatibilidad"""
        if analysis['status'] == 'ok':
            return "# No se requiere parche"
            
        fix_code = """
def fix_features_compatibility(features_df, expected_features_list, target_count=88):
    \"\"\"
    Parche para ajustar features faltantes al conjunto tcn_definitivo completo
    \"\"\"
    current_count = len(features_df.columns)
    
    if current_count == target_count:
        return features_df
        
    if current_count < target_count:
        # Agregar features faltantes con valores por defecto
        missing_features = [
"""
        
        if analysis.get('missing_features_identified'):
            for feature in analysis['missing_features_identified']:
                fix_code += f'            "{feature}",\n'
                
        fix_code += """        ]
        
        print(f"⚠️ Agregando {len(missing_features)} features faltantes con valores por defecto")
        
        for feature in missing_features:
            if feature not in features_df.columns:
                # Valor por defecto según tipo de feature
                if 'rsi' in feature:
                    default_value = 50.0  # RSI neutral
                elif 'momentum' in feature:
                    default_value = 0.0   # Sin momentum
                elif 'volatility' in feature:
                    default_value = 0.01  # Volatilidad mínima
                elif 'ratio' in feature:
                    default_value = 1.0   # Ratio neutro
                elif 'position' in feature:
                    default_value = 0.5   # Posición neutral
                else:
                    default_value = 0.0   # Valor neutro genérico
                    
                features_df[feature] = default_value
                print(f"   ✅ {feature} = {default_value}")
    
    # Reordenar columnas según expected_features_list
    if expected_features_list:
        available_features = [f for f in expected_features_list if f in features_df.columns]
        features_df = features_df[available_features]
        
    return features_df
"""
        
        return fix_code
        
    def run_full_diagnosis(self):
        """Ejecutar diagnóstico completo"""
        print("🚀 INICIANDO DIAGNÓSTICO AVANZADO DE COMPATIBILIDAD")
        print("=" * 60)
        
        # 1. Analizar modelo
        model_info = self.analyze_model_requirements()
        expected_features = model_info.get('expected_features', 88)  # tcn_definitivo debería ser 88
        
        # 2. Calcular features actuales
        features_df, feature_names = self.calculate_current_features()
        
        # 3. Identificar diferencias (features faltantes)
        analysis = self.identify_missing_features(expected_features, feature_names)
        
        # 4. Sugerir soluciones
        solutions = self.suggest_solutions(analysis)
        
        # 5. Crear parche
        fix_code = self.create_compatibility_fix(analysis)
        
        # Mostrar resultados
        print(f"\n📊 RESUMEN DEL DIAGNÓSTICO")
        print("=" * 40)
        print(f"🎯 Modelo analizado: {self.btc_model_path}")
        print(f"📊 Features esperadas: {expected_features}")
        print(f"📊 Features calculadas: {len(feature_names)}")
        print(f"⚠️ Estado: {analysis['status']}")
        
        print(f"\n💡 SOLUCIONES RECOMENDADAS:")
        for solution in solutions:
            print(f"   {solution}")
            
        # Guardar resultados
        results = {
            'model_info': model_info,
            'analysis': analysis,
            'solutions': solutions,
            'fix_code': fix_code,
            'feature_names': feature_names
        }
        
        with open('diagnosis_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
            
        with open('features_compatibility_fix.py', 'w') as f:
            f.write(fix_code)
            
        print(f"\n💾 Resultados guardados:")
        print(f"   📄 diagnosis_results.json")
        print(f"   🔧 features_compatibility_fix.py")
        
        return results


def main():
    """Función principal"""
    diagnostic = FeatureCompatibilityDiagnostic()
    results = diagnostic.run_full_diagnosis()
    
    print(f"\n✅ DIAGNÓSTICO COMPLETADO")
    print("=" * 40)
    
    if results['analysis']['status'] == 'ok':
        print("🎉 No se detectaron problemas de compatibilidad")
    else:
        print("⚠️ Se detectaron problemas de compatibilidad")
        print("🔧 Revisa los archivos generados para las soluciones")


if __name__ == "__main__":
    main()
