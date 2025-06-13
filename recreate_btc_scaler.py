#!/usr/bin/env python3
"""
🔧 RECREAR SCALER BTCUSDT
Recrea el scaler exacto usado en el entrenamiento del modelo definitivo
"""

import asyncio
import pickle
from tcn_definitivo_trainer import DefinitiveTCNTrainer

async def recreate_btc_scaler():
    """🔧 Recrear scaler para BTCUSDT"""

    print("🔧 RECREANDO SCALER PARA BTCUSDT")
    print("=" * 50)

    try:
        # Crear trainer
        trainer = DefinitiveTCNTrainer()
        symbol = "BTCUSDT"

        print(f"📊 Obteniendo datos de entrenamiento para {symbol}...")

        # 1. Obtener los mismos datos que se usaron en entrenamiento
        df = await trainer.get_real_market_data(symbol, days=45)

        # 2. Crear las mismas 66 features
        features = trainer.create_66_features(df)

        # 3. Crear las mismas etiquetas
        df_labeled = trainer.create_balanced_labels(df, features, symbol)

        # 4. Preparar datos exactamente igual que en entrenamiento
        X, y, scaler, feature_columns, class_weights = trainer.prepare_training_data(df_labeled, features)

        # 5. Guardar scaler y metadata
        import os
        os.makedirs(f'models/definitivo_{symbol.lower()}', exist_ok=True)

        scaler_path = f'models/definitivo_{symbol.lower()}/scaler.pkl'
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"💾 Scaler guardado: {scaler_path}")

        features_path = f'models/definitivo_{symbol.lower()}/feature_columns.pkl'
        with open(features_path, 'wb') as f:
            pickle.dump(feature_columns, f)
        print(f"💾 Feature columns guardados: {features_path}")

        # 6. Verificar
        print(f"\n✅ SCALER RECREADO EXITOSAMENTE")
        print(f"   - Features: {len(feature_columns)}")
        print(f"   - Datos de entrenamiento: {X.shape}")
        print(f"   - Distribución de clases: {dict(zip(*np.unique(y, return_counts=True)))}")

        return True

    except Exception as e:
        print(f"❌ Error recreando scaler: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import numpy as np
    asyncio.run(recreate_btc_scaler())
