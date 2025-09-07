#!/usr/bin/env python3
"""
🎯 GESTOR DE CONFIGURACIONES DE BACKTEST
Sistema avanzado para gestionar y optimizar configuraciones de backtesting

CARACTERÍSTICAS:
- Configuraciones predefinidas para diferentes estrategias
- Optimización automática de parámetros
- Validación de configuraciones
- Templates para diferentes escenarios
- Exportación/importación de configuraciones
"""

import json
import yaml
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import itertools
import random
from backtest_system import BacktestConfig, RiskLevel

class StrategyType(Enum):
    """Tipos de estrategias predefinidas"""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    SCALPING = "scalping"
    SWING = "swing"
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"

class OptimizationMethod(Enum):
    """Métodos de optimización"""
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN = "bayesian"
    GENETIC = "genetic"

@dataclass
class ParameterRange:
    """Rango de parámetros para optimización"""
    name: str
    min_value: float
    max_value: float
    step: float = None
    values: List[Any] = None
    param_type: str = "float"  # float, int, choice
    
    def get_values(self) -> List[Any]:
        """Obtener lista de valores para el parámetro"""
        if self.values:
            return self.values
        
        if self.param_type == "int":
            if self.step:
                return list(range(int(self.min_value), int(self.max_value) + 1, int(self.step)))
            else:
                return [int(self.min_value), int(self.max_value)]
        else:
            if self.step:
                values = []
                current = self.min_value
                while current <= self.max_value:
                    values.append(round(current, 6))
                    current += self.step
                return values
            else:
                return [self.min_value, self.max_value]

@dataclass
class OptimizationConfig:
    """Configuración para optimización de parámetros"""
    method: OptimizationMethod = OptimizationMethod.GRID_SEARCH
    max_iterations: int = 100
    cv_folds: int = 3
    scoring_metric: str = "sharpe_ratio"  # sharpe_ratio, total_return, profit_factor, etc.
    parameter_ranges: List[ParameterRange] = field(default_factory=list)
    
    def add_parameter_range(self, param_range: ParameterRange):
        """Añadir rango de parámetros"""
        self.parameter_ranges.append(param_range)
    
    def get_parameter_combinations(self) -> List[Dict[str, Any]]:
        """Generar todas las combinaciones de parámetros"""
        if not self.parameter_ranges:
            return [{}]
        
        param_values = {}
        for param_range in self.parameter_ranges:
            param_values[param_range.name] = param_range.get_values()
        
        # Generar todas las combinaciones
        keys = list(param_values.keys())
        values = list(param_values.values())
        combinations = []
        
        for combo in itertools.product(*values):
            combinations.append(dict(zip(keys, combo)))
        
        return combinations

class BacktestConfigManager:
    """Gestor de configuraciones de backtest"""
    
    def __init__(self):
        self.strategy_templates = self._load_strategy_templates()
        self.optimization_configs = {}
    
    def _load_strategy_templates(self) -> Dict[StrategyType, Dict[str, Any]]:
        """Cargar templates de estrategias predefinidas"""
        return {
            StrategyType.CONSERVATIVE: {
                "position_size_pct": 0.05,
                "stop_loss_pct": 0.015,
                "take_profit_pct": 0.03,
                "max_positions": 2,
                "min_confidence": 70.0,
                "min_signal_strength": 0.7,
                "risk_level_filter": [RiskLevel.LOW],
                "ensemble_weights": {"1m": 0.3, "3m": 0.5, "5m": 0.2},
                "description": "Estrategia conservadora con bajo riesgo y alta confianza"
            },
            
            StrategyType.MODERATE: {
                "position_size_pct": 0.1,
                "stop_loss_pct": 0.02,
                "take_profit_pct": 0.04,
                "max_positions": 3,
                "min_confidence": 60.0,
                "min_signal_strength": 0.6,
                "risk_level_filter": [RiskLevel.LOW, RiskLevel.MEDIUM],
                "ensemble_weights": {"1m": 0.4, "3m": 0.4, "5m": 0.2},
                "description": "Estrategia moderada con balance riesgo/retorno"
            },
            
            StrategyType.AGGRESSIVE: {
                "position_size_pct": 0.2,
                "stop_loss_pct": 0.03,
                "take_profit_pct": 0.06,
                "max_positions": 5,
                "min_confidence": 50.0,
                "min_signal_strength": 0.5,
                "risk_level_filter": [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH],
                "ensemble_weights": {"1m": 0.5, "3m": 0.3, "5m": 0.2},
                "description": "Estrategia agresiva con mayor exposición al riesgo"
            },
            
            StrategyType.SCALPING: {
                "position_size_pct": 0.15,
                "stop_loss_pct": 0.005,
                "take_profit_pct": 0.01,
                "max_positions": 10,
                "min_confidence": 55.0,
                "min_signal_strength": 0.55,
                "risk_level_filter": [RiskLevel.LOW, RiskLevel.MEDIUM],
                "ensemble_weights": {"1m": 0.7, "3m": 0.2, "5m": 0.1},
                "description": "Estrategia de scalping con trades rápidos y frecuentes"
            },
            
            StrategyType.SWING: {
                "position_size_pct": 0.08,
                "stop_loss_pct": 0.04,
                "take_profit_pct": 0.08,
                "max_positions": 2,
                "min_confidence": 65.0,
                "min_signal_strength": 0.65,
                "risk_level_filter": [RiskLevel.LOW, RiskLevel.MEDIUM],
                "ensemble_weights": {"1m": 0.2, "3m": 0.3, "5m": 0.5},
                "description": "Estrategia de swing trading con trades de mediano plazo"
            },
            
            StrategyType.MOMENTUM: {
                "position_size_pct": 0.12,
                "stop_loss_pct": 0.025,
                "take_profit_pct": 0.05,
                "max_positions": 4,
                "min_confidence": 58.0,
                "min_signal_strength": 0.58,
                "risk_level_filter": [RiskLevel.LOW, RiskLevel.MEDIUM],
                "ensemble_weights": {"1m": 0.3, "3m": 0.4, "5m": 0.3},
                "description": "Estrategia de momentum que sigue tendencias fuertes"
            },
            
            StrategyType.MEAN_REVERSION: {
                "position_size_pct": 0.06,
                "stop_loss_pct": 0.02,
                "take_profit_pct": 0.03,
                "max_positions": 3,
                "min_confidence": 68.0,
                "min_signal_strength": 0.68,
                "risk_level_filter": [RiskLevel.LOW],
                "ensemble_weights": {"1m": 0.2, "3m": 0.4, "5m": 0.4},
                "description": "Estrategia de reversión a la media"
            }
        }
    
    def create_config_from_template(
        self, 
        strategy_type: StrategyType,
        symbol: str = 'BTCUSDT',
        timeframe: str = '1m',
        start_date: str = '2024-01-01',
        end_date: str = '2024-12-31',
        initial_balance: float = 10000.0,
        **overrides
    ) -> BacktestConfig:
        """Crear configuración desde template de estrategia"""
        
        template = self.strategy_templates[strategy_type]
        
        # Crear configuración base
        config_dict = {
            "symbol": symbol,
            "timeframe": timeframe,
            "start_date": start_date,
            "end_date": end_date,
            "initial_balance": initial_balance,
            "predictors": ["1m", "3m", "5m"],
            "use_ensemble": True,
            **template
        }
        
        # Aplicar overrides
        config_dict.update(overrides)
        
        # Crear configuración
        config = BacktestConfig(**config_dict)
        
        print(f"✅ Configuración creada para estrategia {strategy_type.value}")
        print(f"📊 {template['description']}")
        
        return config
    
    def create_custom_config(
        self,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
        initial_balance: float = 10000.0,
        **params
    ) -> BacktestConfig:
        """Crear configuración personalizada"""
        
        config_dict = {
            "symbol": symbol,
            "timeframe": timeframe,
            "start_date": start_date,
            "end_date": end_date,
            "initial_balance": initial_balance,
            **params
        }
        
        return BacktestConfig(**config_dict)
    
    def create_optimization_config(
        self,
        base_config: BacktestConfig,
        optimization_method: OptimizationMethod = OptimizationMethod.GRID_SEARCH,
        max_iterations: int = 100,
        scoring_metric: str = "sharpe_ratio"
    ) -> OptimizationConfig:
        """Crear configuración de optimización"""
        
        opt_config = OptimizationConfig(
            method=optimization_method,
            max_iterations=max_iterations,
            scoring_metric=scoring_metric
        )
        
        # Añadir rangos de parámetros comunes
        opt_config.add_parameter_range(ParameterRange(
            name="position_size_pct",
            min_value=0.01,
            max_value=0.3,
            step=0.01,
            param_type="float"
        ))
        
        opt_config.add_parameter_range(ParameterRange(
            name="stop_loss_pct",
            min_value=0.005,
            max_value=0.05,
            step=0.005,
            param_type="float"
        ))
        
        opt_config.add_parameter_range(ParameterRange(
            name="take_profit_pct",
            min_value=0.01,
            max_value=0.1,
            step=0.01,
            param_type="float"
        ))
        
        opt_config.add_parameter_range(ParameterRange(
            name="min_confidence",
            min_value=40.0,
            max_value=80.0,
            step=5.0,
            param_type="float"
        ))
        
        opt_config.add_parameter_range(ParameterRange(
            name="max_positions",
            min_value=1,
            max_value=10,
            step=1,
            param_type="int"
        ))
        
        self.optimization_configs[base_config.symbol] = opt_config
        return opt_config
    
    def generate_config_variations(
        self,
        base_config: BacktestConfig,
        param_variations: Dict[str, List[Any]]
    ) -> List[BacktestConfig]:
        """Generar variaciones de configuración"""
        
        variations = []
        
        # Generar todas las combinaciones
        keys = list(param_variations.keys())
        values = list(param_variations.values())
        
        for combo in itertools.product(*values):
            config_dict = base_config.__dict__.copy()
            config_dict.update(dict(zip(keys, combo)))
            
            try:
                config = BacktestConfig(**config_dict)
                variations.append(config)
            except Exception as e:
                print(f"⚠️ Error creando variación: {e}")
                continue
        
        print(f"✅ Generadas {len(variations)} variaciones de configuración")
        return variations
    
    def save_config(self, config: BacktestConfig, filepath: str):
        """Guardar configuración en archivo"""
        config_dict = config.__dict__.copy()
        
        # Convertir enums a strings
        for key, value in config_dict.items():
            if hasattr(value, 'value'):
                config_dict[key] = value.value
            elif isinstance(value, list) and value and hasattr(value[0], 'value'):
                config_dict[key] = [v.value for v in value]
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
        
        print(f"💾 Configuración guardada en: {filepath}")
    
    def load_config(self, filepath: str) -> BacktestConfig:
        """Cargar configuración desde archivo"""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        # Convertir strings a enums
        if 'risk_level_filter' in config_dict:
            config_dict['risk_level_filter'] = [RiskLevel(level) for level in config_dict['risk_level_filter']]
        
        config = BacktestConfig(**config_dict)
        print(f"📂 Configuración cargada desde: {filepath}")
        return config
    
    def validate_config(self, config: BacktestConfig) -> Tuple[bool, List[str]]:
        """Validar configuración"""
        errors = []
        
        # Validaciones básicas
        if config.position_size_pct <= 0 or config.position_size_pct > 1:
            errors.append("position_size_pct debe estar entre 0 y 1")
        
        if config.stop_loss_pct <= 0 or config.stop_loss_pct > 1:
            errors.append("stop_loss_pct debe estar entre 0 y 1")
        
        if config.take_profit_pct <= 0 or config.take_profit_pct > 1:
            errors.append("take_profit_pct debe estar entre 0 y 1")
        
        if config.min_confidence < 0 or config.min_confidence > 100:
            errors.append("min_confidence debe estar entre 0 y 100")
        
        if config.max_positions <= 0:
            errors.append("max_positions debe ser mayor que 0")
        
        if config.initial_balance <= 0:
            errors.append("initial_balance debe ser mayor que 0")
        
        # Validar pesos del ensemble
        if config.use_ensemble:
            total_weight = sum(config.ensemble_weights.values())
            if abs(total_weight - 1.0) > 0.01:
                errors.append(f"Los pesos del ensemble deben sumar 1.0, actual: {total_weight}")
        
        # Validar fechas
        try:
            from datetime import datetime
            start_dt = datetime.strptime(config.start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(config.end_date, '%Y-%m-%d')
            if start_dt >= end_dt:
                errors.append("start_date debe ser anterior a end_date")
        except ValueError:
            errors.append("Formato de fecha inválido, usar YYYY-MM-DD")
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def get_recommended_configs(
        self,
        symbol: str,
        timeframe: str,
        risk_tolerance: str = "moderate"
    ) -> List[BacktestConfig]:
        """Obtener configuraciones recomendadas"""
        
        risk_mapping = {
            "low": [StrategyType.CONSERVATIVE, StrategyType.MEAN_REVERSION],
            "moderate": [StrategyType.MODERATE, StrategyType.SWING],
            "high": [StrategyType.AGGRESSIVE, StrategyType.SCALPING, StrategyType.MOMENTUM]
        }
        
        strategies = risk_mapping.get(risk_tolerance, [StrategyType.MODERATE])
        configs = []
        
        for strategy in strategies:
            config = self.create_config_from_template(
                strategy_type=strategy,
                symbol=symbol,
                timeframe=timeframe
            )
            configs.append(config)
        
        return configs
    
    def create_parameter_sweep_configs(
        self,
        base_config: BacktestConfig,
        param_name: str,
        param_values: List[Any]
    ) -> List[BacktestConfig]:
        """Crear configuraciones para barrido de parámetros"""
        
        configs = []
        
        for value in param_values:
            config_dict = base_config.__dict__.copy()
            config_dict[param_name] = value
            
            try:
                config = BacktestConfig(**config_dict)
                configs.append(config)
            except Exception as e:
                print(f"⚠️ Error creando configuración con {param_name}={value}: {e}")
                continue
        
        print(f"✅ Creadas {len(configs)} configuraciones para barrido de {param_name}")
        return configs

# === FUNCIONES DE UTILIDAD ===

def create_quick_config(
    symbol: str = 'BTCUSDT',
    timeframe: str = '1m',
    strategy: str = 'moderate',
    days: int = 30
) -> BacktestConfig:
    """Crear configuración rápida para testing"""
    
    manager = BacktestConfigManager()
    
    # Mapear estrategia
    strategy_mapping = {
        'conservative': StrategyType.CONSERVATIVE,
        'moderate': StrategyType.MODERATE,
        'aggressive': StrategyType.AGGRESSIVE,
        'scalping': StrategyType.SCALPING,
        'swing': StrategyType.SWING,
        'momentum': StrategyType.MOMENTUM,
        'mean_reversion': StrategyType.MEAN_REVERSION
    }
    
    strategy_type = strategy_mapping.get(strategy, StrategyType.MODERATE)
    
    # Calcular fechas
    from datetime import datetime, timedelta
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    return manager.create_config_from_template(
        strategy_type=strategy_type,
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d')
    )

def create_comparison_configs(
    symbol: str = 'BTCUSDT',
    timeframes: List[str] = ['1m', '3m', '5m'],
    strategies: List[str] = ['conservative', 'moderate', 'aggressive']
) -> List[BacktestConfig]:
    """Crear configuraciones para comparación"""
    
    manager = BacktestConfigManager()
    configs = []
    
    for timeframe in timeframes:
        for strategy in strategies:
            config = create_quick_config(symbol, timeframe, strategy)
            configs.append(config)
    
    print(f"✅ Creadas {len(configs)} configuraciones para comparación")
    return configs

# === EJEMPLO DE USO ===
if __name__ == "__main__":
    # Crear gestor
    manager = BacktestConfigManager()
    
    # Crear configuración desde template
    config = manager.create_config_from_template(
        strategy_type=StrategyType.MODERATE,
        symbol='BTCUSDT',
        timeframe='1m',
        start_date='2024-01-01',
        end_date='2024-01-31'
    )
    
    # Validar configuración
    is_valid, errors = manager.validate_config(config)
    if is_valid:
        print("✅ Configuración válida")
    else:
        print("❌ Errores en configuración:")
        for error in errors:
            print(f"  - {error}")
    
    # Guardar configuración
    manager.save_config(config, 'config_moderate_btc.json')
    
    # Crear configuraciones de comparación
    comparison_configs = create_comparison_configs()
    print(f"📊 Configuraciones de comparación creadas: {len(comparison_configs)}")
