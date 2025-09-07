#!/usr/bin/env python3
"""
🎯 DETECTOR DE DIFERENCIAS ESPECÍFICAS: DOTUSDT vs OTROS
========================================================

Script especializado para detectar exactamente qué hace que DOTUSDT funcione
mientras que otros modelos fallan en esta PC Windows.

ENFOQUE:
1. Análisis profundo de DOTUSDT exitoso
2. Comparación línea por línea con otros modelos
3. Detección de diferencias en configuración
4. Identificación de problemas específicos por par
5. Soluciones personalizadas por modelo
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import json
import os
import hashlib
from typing import Dict, List, Tuple, Optional, Any, Set
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Importar componentes del sistema
from centralized_features_engine3 import CentralizedFeaturesEngine

class DotusSpecificDifferenceDetector:
    """Detector de diferencias específicas que hacen que DOTUSDT funcione"""
    
    def __init__(self):
        """Inicializar detector"""
        self.features_engine = CentralizedFeaturesEngine()
        self.models_dir = Path("models")
        
        # Pares a analizar (DOTUSDT primero como referencia)
        self.symbols = ['DOTUSDT', 'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT']
        
        # Almacenar diferencias detectadas
        self.differences = {}
        self.dotus_baseline = {}
        
    def analyze_dotus_baseline(self) -> Dict[str, Any]:
        """Analizar DOTUSDT como línea base de funcionamiento"""
        print("🎯 ANALIZANDO DOTUSDT COMO LÍNEA BASE DE FUNCIONAMIENTO")
        print("=" * 60)
        
        dotus_info = {
            'model_path': None,
            'model_exists': False,
            'input_shape': None,
            'expected_features': None,
            'feature_set': None,
            'config': None,
            'model_hash': None,
            'features_calculated': 0,
            'features_list': [],
            'working_config': {}
        }
        
        # Buscar modelo DOTUSDT
        dotus_patterns = [
            "*dotusdt*",
            "*adaptive_dotusdt*",
            "*tcn_definitivo*dotusdt*"
        ]
        
        for pattern in dotus_patterns:
            for model_dir in self.models_dir.glob(pattern):
                if model_dir.is_dir():
                    model_file = model_dir / "best_model.h5"
                    if not model_file.exists():
                        model_file = model_dir / "model.h5"
                        
                    if model_file.exists():
                        dotus_info['model_path'] = str(model_dir)
                        dotus_info['model_exists'] = True
                        
                        # Calcular hash del modelo
                        with open(model_file, 'rb') as f:
                            model_content = f.read()
                            dotus_info['model_hash'] = hashlib.md5(model_content).hexdigest()
                        
                        # Cargar información del modelo
                        try:
                            model = tf.keras.models.load_model(model_file)
                            dotus_info['input_shape'] = model.input_shape
                            dotus_info['expected_features'] = model.input_shape[-1] if model.input_shape else None
                        except Exception as e:
                            print(f"⚠️ Error cargando modelo DOTUSDT: {e}")
                            
                        # Cargar configuración
                        config_file = model_dir / "config.json"
                        if config_file.exists():
                            try:
                                with open(config_file, 'r') as f:
                                    config = json.load(f)
                                    dotus_info['config'] = config
                                    dotus_info['feature_set'] = config.get('feature_set', 'NO ESPECIFICADO')
                                    dotus_info['working_config'] = config
                            except Exception as e:
                                print(f"⚠️ Error cargando config DOTUSDT: {e}")
                                
                        break
                        
        # Calcular features para DOTUSDT
        if dotus_info['model_exists']:
            test_data = self.generate_test_data_for_symbol('DOTUSDT')
            try:
                features_df = self.features_engine.calculate_features(test_data, feature_set='tcn_definitivo')
                dotus_info['features_calculated'] = len(features_df.columns)
                dotus_info['features_list'] = list(features_df.columns)
                print(f"✅ DOTUSDT: {dotus_info['features_calculated']} features calculadas exitosamente")
            except Exception as e:
                print(f"❌ Error calculando features DOTUSDT: {e}")
                
        # Mostrar información de DOTUSDT
        print(f"\n📊 INFORMACIÓN DE DOTUSDT (LÍNEA BASE):")
        print(f"   🎯 Modelo existe: {dotus_info['model_exists']}")
        if dotus_info['model_exists']:
            print(f"   📁 Ruta: {dotus_info['model_path']}")
            print(f"   📏 Input shape: {dotus_info['input_shape']}")
            print(f"   🔢 Features esperadas: {dotus_info['expected_features']}")
            print(f"   ⚙️ Feature set: {dotus_info['feature_set']}")
            print(f"   🔐 Hash modelo: {dotus_info['model_hash'][:16]}...")
            print(f"   📊 Features calculadas: {dotus_info['features_calculated']}")
            
        self.dotus_baseline = dotus_info
        return dotus_info
        
    def generate_test_data_for_symbol(self, symbol: str, periods: int = 200) -> pd.DataFrame:
        """Generar datos de prueba específicos para un símbolo"""
        # Precios base realistas por símbolo
        base_prices = {
            'DOTUSDT': 7.0,
            'BTCUSDT': 45000.0,
            'ETHUSDT': 2800.0,
            'BNBUSDT': 320.0,
            'XRPUSDT': 0.55
        }
        
        base_price = base_prices.get(symbol, 100.0)
        volatility = 0.02
        
        # Simulación de caminata aleatoria
        dates = pd.date_range(start='2024-01-01', periods=periods, freq='5T')
        np.random.seed(42)  # Semilla fija para consistencia
        
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
            'volume': np.random.lognormal(16, 0.3, periods)
        })
        
        # Asegurar consistencia OHLC
        df['high'] = np.maximum.reduce([df['open'], df['high'], df['low'], df['close']])
        df['low'] = np.minimum.reduce([df['open'], df['high'], df['low'], df['close']])
        
        return df
        
    def detect_differences_for_symbol(self, symbol: str) -> Dict[str, Any]:
        """Detectar diferencias específicas para un símbolo vs DOTUSDT"""
        print(f"\n🔍 DETECTANDO DIFERENCIAS PARA {symbol}")
        print("-" * 40)
        
        differences = {
            'symbol': symbol,
            'model_exists': False,
            'model_path': None,
            'input_shape_match': False,
            'feature_set_match': False,
            'config_match': False,
            'features_calculated': 0,
            'features_match': False,
            'specific_issues': [],
            'differences_from_dotus': [],
            'recommendations': []
        }
        
        # Buscar modelo del símbolo
        model_patterns = [
            f"*{symbol.lower()}*",
            f"*adaptive_{symbol.lower()}*",
            f"*tcn_definitivo*{symbol.lower()}*"
        ]
        
        for pattern in model_patterns:
            for model_dir in self.models_dir.glob(pattern):
                if model_dir.is_dir():
                    model_file = model_dir / "best_model.h5"
                    if not model_file.exists():
                        model_file = model_dir / "model.h5"
                        
                    if model_file.exists():
                        differences['model_exists'] = True
                        differences['model_path'] = str(model_dir)
                        
                        # Comparar input shape
                        try:
                            model = tf.keras.models.load_model(model_file)
                            symbol_input_shape = model.input_shape
                            differences['input_shape_match'] = (symbol_input_shape == self.dotus_baseline.get('input_shape'))
                            
                            if not differences['input_shape_match']:
                                differences['differences_from_dotus'].append(
                                    f"Input shape diferente: {symbol_input_shape} vs {self.dotus_baseline.get('input_shape')}"
                                )
                                
                        except Exception as e:
                            differences['specific_issues'].append(f"Error cargando modelo: {e}")
                            
                        # Comparar configuración
                        config_file = model_dir / "config.json"
                        if config_file.exists():
                            try:
                                with open(config_file, 'r') as f:
                                    config = json.load(f)
                                    symbol_feature_set = config.get('feature_set', 'NO ESPECIFICADO')
                                    dotus_feature_set = self.dotus_baseline.get('feature_set', 'NO ESPECIFICADO')
                                    
                                    differences['feature_set_match'] = (symbol_feature_set == dotus_feature_set)
                                    differences['config_match'] = (config == self.dotus_baseline.get('config', {}))
                                    
                                    if not differences['feature_set_match']:
                                        differences['differences_from_dotus'].append(
                                            f"Feature set diferente: {symbol_feature_set} vs {dotus_feature_set}"
                                        )
                                        
                            except Exception as e:
                                differences['specific_issues'].append(f"Error cargando config: {e}")
                        else:
                            differences['differences_from_dotus'].append("No tiene archivo config.json")
                            
                        break
                        
        # Calcular features para comparar
        if differences['model_exists']:
            test_data = self.generate_test_data_for_symbol(symbol)
            try:
                features_df = self.features_engine.calculate_features(test_data, feature_set='tcn_definitivo')
                differences['features_calculated'] = len(features_df.columns)
                differences['features_match'] = (differences['features_calculated'] == self.dotus_baseline.get('features_calculated', 0))
                
                if not differences['features_match']:
                    differences['differences_from_dotus'].append(
                        f"Features calculadas diferentes: {differences['features_calculated']} vs {self.dotus_baseline.get('features_calculated', 0)}"
                    )
                    
            except Exception as e:
                differences['specific_issues'].append(f"Error calculando features: {e}")
                
        # Generar recomendaciones específicas
        if not differences['model_exists']:
            differences['recommendations'].append("🔧 PRIORIDAD ALTA: Crear o restaurar modelo para este símbolo")
        elif not differences['input_shape_match']:
            differences['recommendations'].append("🔧 PRIORIDAD ALTA: Reentrenar modelo con input shape compatible")
        elif not differences['feature_set_match']:
            differences['recommendations'].append("🔧 PRIORIDAD MEDIA: Sincronizar feature set con DOTUSDT")
        elif not differences['features_match']:
            differences['recommendations'].append("🔧 PRIORIDAD ALTA: Verificar cálculo de features")
            
        # Mostrar diferencias detectadas
        print(f"   🎯 Modelo existe: {differences['model_exists']}")
        print(f"   📏 Input shape coincide: {differences['input_shape_match']}")
        print(f"   ⚙️ Feature set coincide: {differences['feature_set_match']}")
        print(f"   📊 Features calculadas: {differences['features_calculated']}")
        print(f"   ✅ Features coinciden: {differences['features_match']}")
        
        if differences['differences_from_dotus']:
            print(f"   ❌ Diferencias detectadas: {len(differences['differences_from_dotus'])}")
            for diff in differences['differences_from_dotus'][:3]:
                print(f"      - {diff}")
                
        if differences['recommendations']:
            print(f"   💡 Recomendaciones: {len(differences['recommendations'])}")
            for rec in differences['recommendations']:
                print(f"      {rec}")
                
        return differences
        
    def generate_difference_report(self) -> str:
        """Generar reporte de diferencias detectadas"""
        report = f"""
🎯 REPORTE DE DIFERENCIAS ESPECÍFICAS: DOTUSDT vs OTROS MODELOS
================================================================
Fecha: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

📊 LÍNEA BASE DOTUSDT (FUNCIONANDO)
-----------------------------------
• Modelo existe: {self.dotus_baseline.get('model_exists', False)}
• Input shape: {self.dotus_baseline.get('input_shape', 'N/A')}
• Features esperadas: {self.dotus_baseline.get('expected_features', 'N/A')}
• Feature set: {self.dotus_baseline.get('feature_set', 'N/A')}
• Features calculadas: {self.dotus_baseline.get('features_calculated', 0)}
• Hash modelo: {self.dotus_baseline.get('model_hash', 'N/A')[:16] if self.dotus_baseline.get('model_hash') else 'N/A'}...

🔍 ANÁLISIS DE DIFERENCIAS POR SÍMBOLO
---------------------------------------
"""
        
        for symbol in self.symbols:
            if symbol != 'DOTUSDT' and symbol in self.differences:
                diff = self.differences[symbol]
                report += f"\n{symbol}:\n"
                report += f"  • Modelo existe: {diff['model_exists']}\n"
                report += f"  • Input shape coincide: {diff['input_shape_match']}\n"
                report += f"  • Feature set coincide: {diff['feature_set_match']}\n"
                report += f"  • Features calculadas: {diff['features_calculated']}\n"
                report += f"  • Features coinciden: {diff['features_match']}\n"
                
                if diff['differences_from_dotus']:
                    report += f"  • Diferencias: {len(diff['differences_from_dotus'])}\n"
                    for i, diff_desc in enumerate(diff['differences_from_dotus'][:5], 1):
                        report += f"    {i}. {diff_desc}\n"
                        
                if diff['recommendations']:
                    report += f"  • Recomendaciones: {len(diff['recommendations'])}\n"
                    for i, rec in enumerate(diff['recommendations'], 1):
                        report += f"    {i}. {rec}\n"
                        
        # Resumen de problemas
        total_models = len(self.symbols) - 1  # Excluir DOTUSDT
        working_models = sum(1 for diff in self.differences.values() if diff.get('features_match', False))
        
        report += f"""
📊 RESUMEN DE PROBLEMAS
-----------------------
• Total modelos analizados: {total_models}
• Modelos funcionando: {working_models}
• Modelos con problemas: {total_models - working_models}
• Tasa de éxito: {(working_models/total_models)*100:.1f}%

🔧 SOLUCIONES PRIORITARIAS
---------------------------
"""
        
        # Agrupar recomendaciones por prioridad
        all_recommendations = []
        for diff in self.differences.values():
            all_recommendations.extend(diff.get('recommendations', []))
            
        # Contar recomendaciones por tipo
        priority_counts = {}
        for rec in all_recommendations:
            if 'PRIORIDAD ALTA' in rec:
                priority_counts['alta'] = priority_counts.get('alta', 0) + 1
            elif 'PRIORIDAD MEDIA' in rec:
                priority_counts['media'] = priority_counts.get('media', 0) + 1
            else:
                priority_counts['baja'] = priority_counts.get('baja', 0) + 1
                
        for priority, count in priority_counts.items():
            report += f"• {priority.upper()}: {count} acciones requeridas\n"
            
        return report
        
    def run_difference_detection(self) -> Dict[str, Any]:
        """Ejecutar detección completa de diferencias"""
        print("🚀 INICIANDO DETECCIÓN DE DIFERENCIAS ESPECÍFICAS")
        print("=" * 60)
        
        # 1. Analizar DOTUSDT como línea base
        dotus_baseline = self.analyze_dotus_baseline()
        
        if not dotus_baseline['model_exists']:
            print("❌ ERROR: DOTUSDT no funciona como línea base")
            return {}
            
        # 2. Detectar diferencias para cada símbolo
        for symbol in self.symbols:
            if symbol != 'DOTUSDT':
                differences = self.detect_differences_for_symbol(symbol)
                self.differences[symbol] = differences
                
        # 3. Generar reporte
        report = self.generate_difference_report()
        
        # 4. Guardar resultados
        results = {
            'dotus_baseline': dotus_baseline,
            'differences': self.differences,
            'report': report,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        with open('dotus_differences_analysis.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str, ensure_ascii=False)
            
        with open('dotus_differences_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)
            
        print(f"\n💾 Resultados guardados:")
        print(f"   📄 dotus_differences_analysis.json")
        print(f"   📄 dotus_differences_report.txt")
        
        # Mostrar reporte
        print(f"\n{report}")
        
        return results


def main():
    """Función principal"""
    detector = DotusSpecificDifferenceDetector()
    results = detector.run_difference_detection()
    
    print(f"\n✅ DETECCIÓN DE DIFERENCIAS COMPLETADA")
    print("=" * 50)
    
    if results:
        # Resumen final
        working_models = sum(1 for diff in results['differences'].values() if diff.get('features_match', False))
        total_models = len(results['differences'])
        
        print(f"📊 Resumen:")
        print(f"   🎯 DOTUSDT: Funcionando ✅")
        print(f"   📊 Otros modelos: {working_models}/{total_models} funcionando")
        
        if working_models < total_models:
            print(f"🔧 Se detectaron diferencias específicas. Revisa el reporte para soluciones.")
        else:
            print(f"🎉 Todos los modelos funcionan correctamente")
    else:
        print("❌ No se pudo completar el análisis")


if __name__ == "__main__":
    main()