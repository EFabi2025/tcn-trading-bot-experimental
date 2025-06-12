#!/usr/bin/env python3
"""
₿ BTC MODEL - Modelo específico para Bitcoin basado en éxito ETH
"""

from simple_pair_models import train_pair

if __name__ == "__main__":
    print("₿ ENTRENANDO MODELO BITCOIN")
    print("="*50)
    print("🔧 Usando configuración optimizada para BTC")
    print("🎯 Threshold: 0.15% (menos volátil que ETH)")
    print("⏰ Predicción: 4 períodos (20 minutos)")
    
    success = train_pair('BTCUSDT')
    
    if success:
        print("\n🎉 ¡MODELO BTC COMPLETADO!")
        print("✅ Ahora tenemos modelos para ETH y BTC")
    else:
        print("\n❌ Error en modelo BTC") 