#!/usr/bin/env python3
"""
🧪 TEST: Verificar uso real de modelos TCN para señales
"""

import asyncio
from simple_professional_manager import SimpleProfessionalTradingManager

async def test_tcn_signals():
    """🧪 Test completo de señales TCN"""
    print("🧪 TESTING SEÑALES CON MODELOS TCN")
    print("=" * 50)
    
    try:
        # Inicializar manager completo
        manager = SimpleProfessionalTradingManager()
        await manager.initialize()  # ✅ Inicialización completa
        
        print(f"\n📊 ESTADO DE MODELOS TCN:")
        print(f"   Activos: {manager.tcn_models_active}")
        print(f"   Modelos: {list(manager.tcn_models.keys())}")
        print(f"   Balance actual: ${manager.current_balance:.2f}")
        
        # Datos de prueba
        test_prices = {
            'BTCUSDT': 109500.0,
            'ETHUSDT': 2800.0,
            'BNBUSDT': 670.0
        }
        
        print(f"\n🔍 GENERANDO SEÑALES DE PRUEBA:")
        
        # Generar señales
        signals = await manager._generate_simple_signals(test_prices)
        
        print(f"\n📊 RESULTADO DE SEÑALES:")
        print(f"   Total señales: {len(signals)}")
        
        for symbol, signal_data in signals.items():
            print(f"\n   🔍 {symbol}:")
            print(f"      Acción: {signal_data['signal']}")
            print(f"      Confianza: {signal_data['confidence']:.1%}")
            print(f"      Precio: ${signal_data['current_price']:.4f}")
            print(f"      Fuente: {signal_data['reason']}")
            
            if 'tcn_details' in signal_data:
                tcn = signal_data['tcn_details']
                print(f"      🤖 TCN usado: {tcn['model_used']}")
                print(f"      🎯 Raw prediction: {[f'{x:.3f}' for x in tcn['raw_prediction']]}")
                print(f"      ✅ USANDO MODELO TCN REAL")
            else:
                print(f"      ⚠️ Usando señal básica/fallback")
        
        # Test directo de predicción TCN
        print(f"\n🤖 TEST DIRECTO DE PREDICCIONES TCN:")
        
        for symbol in test_prices:
            if symbol in manager.tcn_models:
                tcn_pred = await manager._get_tcn_prediction(symbol, test_prices[symbol])
                
                if tcn_pred:
                    print(f"   ✅ {symbol}: {tcn_pred['action']} ({tcn_pred['confidence']:.1%})")
                    print(f"      Raw: {[f'{x:.3f}' for x in tcn_pred['raw_prediction']]}")
                else:
                    print(f"   ❌ {symbol}: Predicción falló")
            else:
                print(f"   ⚠️ {symbol}: Modelo no disponible")
        
        # Verificar diferencia entre TCN y fallback
        print(f"\n🔍 COMPARANDO TCN vs FALLBACK:")
        
        # Temporal: desactivar TCN
        manager.tcn_models_active = False
        fallback_signals = await manager._generate_simple_signals(test_prices)
        
        # Reactivar TCN
        manager.tcn_models_active = True
        tcn_signals = await manager._generate_simple_signals(test_prices)
        
        print(f"   Señales TCN: {len(tcn_signals)}")
        print(f"   Señales Fallback: {len(fallback_signals)}")
        
        # Comparar fuentes
        for symbol in test_prices:
            tcn_reason = tcn_signals.get(symbol, {}).get('reason', 'No signal')
            fallback_reason = fallback_signals.get(symbol, {}).get('reason', 'No signal')
            
            print(f"   {symbol}:")
            print(f"      TCN: {tcn_reason}")
            print(f"      Fallback: {fallback_reason}")
            
            if tcn_reason == 'tcn_model_prediction':
                print(f"      ✅ TCN funcionando correctamente")
            elif tcn_reason == 'basic_fallback_signal':
                print(f"      ⚠️ Usando fallback en lugar de TCN")
            else:
                print(f"      ❌ Sin señal generada")
        
        print(f"\n🎯 RESUMEN:")
        tcn_working = any('tcn_model_prediction' in signals.get(s, {}).get('reason', '') for s in test_prices)
        
        if tcn_working:
            print(f"   ✅ MODELOS TCN ESTÁN FUNCIONANDO PARA SEÑALES")
            print(f"   🎯 El sistema usa predicciones reales de TCN")
        else:
            print(f"   ⚠️ MODELOS TCN NO SE USAN PARA SEÑALES")
            print(f"   🔄 Sistema usa solo fallbacks/señales básicas")
        
    except Exception as e:
        print(f"❌ Error en test: {e}")

if __name__ == "__main__":
    asyncio.run(test_tcn_signals()) 