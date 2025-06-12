#!/usr/bin/env python3
"""
🔍 VERIFICADOR DE USO REAL DE MODELOS TCN
Script para comprobar si el sistema usa TCN reales o fallbacks
"""

import asyncio
import numpy as np
from simple_professional_manager import SimpleProfessionalTradingManager

async def verify_tcn_usage():
    """Verificar uso real de modelos TCN"""
    print("🔍 VERIFICANDO USO REAL DE MODELOS TCN")
    print("=" * 50)

    try:
        # Inicializar manager
        manager = SimpleProfessionalTradingManager()
        await manager._initialize_tcn_models()

        print(f"\n📊 ESTADO DE MODELOS:")
        print(f"   Modelos activos: {manager.tcn_models_active}")
        print(f"   Modelos disponibles: {list(manager.tcn_models.keys())}")

        # Verificar cada modelo
        for pair, model_info in manager.tcn_models.items():
            if model_info is not None:
                print(f"\n🔍 ANÁLISIS {pair}:")
                print(f"   📊 Parámetros: {model_info['params']:,}")
                print(f"   📐 Input shape: {model_info['input_shape']}")
                print(f"   📤 Output shape: {model_info['output_shape']}")
                print(f"   📅 Cargado: {model_info['loaded_at']}")
                print(f"   🔧 TF version: {model_info['tf_version']}")

                # Probar predicción con datos dummy
                model = model_info['model']
                input_shape = model_info['input_shape']

                # Crear datos de prueba
                batch_size = 1
                timesteps = input_shape[1]
                features = input_shape[2]

                dummy_data = np.random.random((batch_size, timesteps, features))

                try:
                    prediction = model.predict(dummy_data, verbose=0)
                    pred_class = np.argmax(prediction[0])
                    confidence = np.max(prediction[0])

                    classes = ['SELL', 'HOLD', 'BUY']

                    print(f"   🎯 Predicción de prueba:")
                    print(f"      Clase: {classes[pred_class]}")
                    print(f"      Confianza: {confidence:.3f}")
                    print(f"      Raw output: {prediction[0]}")

                    # Verificar si es modelo entrenado o vacío
                    # Los modelos vacíos tienden a dar outputs muy uniformes
                    variance = np.var(prediction[0])
                    if variance < 0.001:
                        print(f"   ⚠️ POSIBLE MODELO VACÍO (varianza muy baja: {variance:.6f})")
                        print(f"   🤖 Estado: FALLBACK SIN PESOS ENTRENADOS")
                    else:
                        print(f"   ✅ MODELO CON PESOS ENTRENADOS (varianza: {variance:.6f})")
                        print(f"   🎯 Estado: TCN REAL FUNCIONANDO")

                except Exception as e:
                    print(f"   ❌ Error en predicción: {e}")

        # Verificar dónde vienen las señales en el sistema real
        print(f"\n🔍 VERIFICANDO FUENTE DE SEÑALES:")

        # Simular obtención de señales
        dummy_prices = {'BTCUSDT': 109000.0, 'ETHUSDT': 2800.0, 'BNBUSDT': 670.0}
        signals = await manager._generate_simple_signals(dummy_prices)

        for symbol, signal in signals.items():
            print(f"   {symbol}: {signal['action']} ({signal['confidence']:.1f}%)")

        # Buscar en el código si usa TCN o señales aleatorias
        print(f"\n🕵️ ANÁLISIS DEL CÓDIGO DE SEÑALES:")
        try:
            import inspect
            source = inspect.getsource(manager._generate_simple_signals)

            if 'tcn_models' in source.lower() or 'model.predict' in source.lower():
                print("   ✅ El código de señales SÍ usa modelos TCN")
            else:
                print("   ⚠️ El código de señales NO usa modelos TCN")
                print("   🔄 Usa señales aleatorias/básicas")

            if 'random' in source.lower():
                print("   ⚠️ DETECTADO: Uso de señales aleatorias")

        except Exception as e:
            print(f"   ❌ No se pudo analizar el código: {e}")

    except Exception as e:
        print(f"❌ Error en verificación: {e}")

if __name__ == "__main__":
    asyncio.run(verify_tcn_usage())
