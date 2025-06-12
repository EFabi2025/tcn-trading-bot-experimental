# 📦 Modelos Archivados

Esta carpeta contiene modelos TCN que **NO están en uso actualmente** en el sistema de trading.

## 🎯 Modelos Activos (en uso)

Los modelos que SÍ se están usando están en `/models/`:

- ✅ `models/tcn_final_btcusdt.h5` - Modelo TCN para BTC/USDT
- ✅ `models/tcn_final_ethusdt.h5` - Modelo TCN para ETH/USDT
- ✅ `models/tcn_final_bnbusdt.h5` - Modelo TCN para BNB/USDT

**Input Shape:** `(50, 21)` - 50 timesteps, 21 features técnicas
**Arquitectura:** TCN (Temporal Convolutional Network)
**TensorFlow:** 2.15.0

## 📦 Modelos Archivados (aquí)

Los modelos en esta carpeta son versiones anteriores o experimentales:

- `production_model_*.h5` - Versiones anteriores con shape (40, 159)
- `tcn_anti_bias_*.h5` - Experimentos anti-sesgo
- `eth_*.h5` - Modelos experimentales de ETH
- `btc_*.h5` - Modelos experimentales de BTC
- `model_fold_*.h5` - Modelos de validación cruzada
- Otros modelos experimentales

## ⚠️ Importante

**NO eliminar esta carpeta** - contiene el historial de desarrollo y puede ser útil para:
- Debugging
- Comparación de rendimiento
- Recuperación de versiones anteriores
- Investigación y desarrollo

## 🔄 Restaurar un modelo

Si necesitas usar un modelo archivado:

1. Copia el archivo desde `archived_models/` a `models/`
2. Actualiza `simple_professional_manager.py` si es necesario
3. Verifica que el input shape sea compatible

---
**Generado:** $(date)
**Sistema:** Professional Trading Bot con TCN
