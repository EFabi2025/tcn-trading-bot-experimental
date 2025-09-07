#!/usr/bin/env python3
"""
🎯 ENTRENADOR DOTUSDT DEFINITIVO
Entrena solo el modelo de DOTUSDT desde cero con técnicas anti-sesgo
"""

import asyncio
from tcn_adaptative_trainer_v2 import AdaptiveTCNTrainer

async def main():
    """🚀 Entrenar solo DOTUSDT desde cero"""

    print("🎯 ENTRENAMIENTO DEFINITIVO - DOTUSDT DESDE CERO")
    print("=" * 70)

    try:
        # Crear trainer
        trainer = AdaptiveTCNTrainer()

        # Entrenar solo DOTUSDT
        print("🚀 Iniciando entrenamiento de DOTUSDT desde cero...")
        print("📊 Usando mismo proceso exitoso que ETHUSDT")
        print("⏱️ Tiempo estimado: ~1.5 horas")
        print("💾 Guardará: modelo + scaler + features + checkpoints")

        success = await trainer.train_adaptive_model("DOTUSDT")

        if success:
            print(f"\n✅ DOTUSDT entrenado exitosamente desde cero")
            print(f"🎯 Archivos guardados en: models/adaptive_dotusdt_5m_6h_48w_tcn_definitivo/")
            print(f"📁 Incluye: best_model.h5, scaler.pkl, feature_columns.pkl")
        else:
            print(f"\n❌ Error entrenando DOTUSDT")

    except Exception as e:
        print(f"❌ Error general: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
