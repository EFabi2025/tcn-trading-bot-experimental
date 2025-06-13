#!/usr/bin/env python3
"""
🎯 ENTRENADOR BNBUSDT DEFINITIVO
Entrena solo el modelo de BNBUSDT con técnicas anti-sesgo
"""

import asyncio
from tcn_definitivo_trainer import DefinitiveTCNTrainer

async def main():
    """🚀 Entrenar solo BNBUSDT"""

    print("🎯 ENTRENAMIENTO DEFINITIVO - BNBUSDT ÚNICAMENTE")
    print("=" * 70)

    try:
        # Crear trainer
        trainer = DefinitiveTCNTrainer()

        # Entrenar solo BNBUSDT
        print("🚀 Iniciando entrenamiento de BNBUSDT...")
        print("📊 Usando mismo proceso exitoso que BTCUSDT y ETHUSDT")
        print("⏱️ Tiempo estimado: ~1.5 horas")
        print("💾 Guardará: modelo + scaler + features + checkpoints")
        print("🎯 Thresholds BNBUSDT: -0.15%/-0.07%/+0.07%/+0.15%")

        success = await trainer.train_definitive_model("BNBUSDT")

        if success:
            print(f"\n🎉 ¡BNBUSDT entrenado exitosamente!")
            print(f"🎯 Archivos guardados en: models/definitivo_bnbusdt/")
            print(f"📁 Incluye: best_model.h5, scaler.pkl, feature_columns.pkl")
            print(f"\n🏆 ¡TODOS LOS MODELOS COMPLETADOS!")
            print(f"   ✅ BTCUSDT: LISTO")
            print(f"   ✅ ETHUSDT: LISTO")
            print(f"   ✅ BNBUSDT: LISTO")
            print(f"   🎯 Progreso: 100% - ¡PROYECTO COMPLETADO!")
        else:
            print(f"\n❌ Error entrenando BNBUSDT")

    except Exception as e:
        print(f"❌ Error general: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
