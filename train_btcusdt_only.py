#!/usr/bin/env python3
"""
🎯 ENTRENADOR BTCUSDT DEFINITIVO
Entrena solo el modelo de BTCUSDT desde cero con técnicas anti-sesgo
"""

import asyncio
from tcn_definitivo_trainer import DefinitiveTCNTrainer

async def main():
    """🚀 Entrenar solo BTCUSDT desde cero"""

    print("🎯 ENTRENAMIENTO DEFINITIVO - BTCUSDT DESDE CERO")
    print("=" * 70)

    try:
        # Crear trainer
        trainer = DefinitiveTCNTrainer()

        # Entrenar solo BTCUSDT
        print("🚀 Iniciando entrenamiento de BTCUSDT desde cero...")
        print("📊 Usando mismo proceso exitoso que ETHUSDT")
        print("⏱️ Tiempo estimado: ~1.5 horas")
        print("💾 Guardará: modelo + scaler + features + checkpoints")

        success = await trainer.train_definitive_model("BTCUSDT")

        if success:
            print(f"\n✅ BTCUSDT entrenado exitosamente desde cero")
            print(f"🎯 Archivos guardados en: models/definitivo_btcusdt/")
            print(f"📁 Incluye: best_model.h5, scaler.pkl, feature_columns.pkl")
        else:
            print(f"\n❌ Error entrenando BTCUSDT")

    except Exception as e:
        print(f"❌ Error general: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
