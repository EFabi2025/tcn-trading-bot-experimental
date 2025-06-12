"""
🎯 PORTFOLIO DIVERSIFICATION MANAGER
=====================================
Gestiona la diversificación del portafolio sin liquidar posiciones existentes.
Implementa optimización gradual y respeta las posiciones actuales.
"""

import asyncio
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import aiohttp
from config.trading_config import TradingConfig

@dataclass
class PortfolioPosition:
    """📊 Representación de una posición en el portafolio"""
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    value_usd: float
    percentage: float
    category: str
    age_minutes: int
    pnl_percent: float

@dataclass
class DiversificationAnalysis:
    """📈 Análisis de diversificación del portafolio"""
    total_value: float
    symbol_concentrations: Dict[str, float]
    category_concentrations: Dict[str, float]
    over_concentrated_symbols: List[str]
    over_concentrated_categories: List[str]
    diversification_score: float  # 0-100
    recommendations: List[str]

class PortfolioDiversificationManager:
    """🎯 Gestor de Diversificación de Portafolio"""

    def __init__(self):
        # Acceder directamente a la configuración de diversificación
        self.diversification_config = TradingConfig.PORTFOLIO_DIVERSIFICATION
        self.last_analysis = None
        self.correlation_cache = {}
        self.last_correlation_update = None

    async def analyze_portfolio_diversification(self, positions: List[PortfolioPosition]) -> DiversificationAnalysis:
        """📊 Analizar diversificación actual del portafolio"""

        if not positions:
            return DiversificationAnalysis(
                total_value=0.0,
                symbol_concentrations={},
                category_concentrations={},
                over_concentrated_symbols=[],
                over_concentrated_categories=[],
                diversification_score=0.0,
                recommendations=["No hay posiciones en el portafolio"]
            )

        total_value = sum(pos.value_usd for pos in positions)

        # Calcular concentraciones por símbolo
        symbol_concentrations = {}
        for pos in positions:
            if pos.symbol not in symbol_concentrations:
                symbol_concentrations[pos.symbol] = 0.0
            symbol_concentrations[pos.symbol] += (pos.value_usd / total_value) * 100

        # Calcular concentraciones por categoría
        category_concentrations = {}
        for pos in positions:
            category = self.diversification_config['SYMBOL_CATEGORIES'].get(pos.symbol, 'UNKNOWN')
            if category not in category_concentrations:
                category_concentrations[category] = 0.0
            category_concentrations[category] += (pos.value_usd / total_value) * 100

        # Identificar sobre-concentraciones
        max_symbol_conc = self.diversification_config['MAX_SYMBOL_CONCENTRATION_PERCENT']
        max_category_conc = self.diversification_config['MAX_CATEGORY_CONCENTRATION_PERCENT']

        over_concentrated_symbols = [
            symbol for symbol, conc in symbol_concentrations.items()
            if conc > max_symbol_conc
        ]

        over_concentrated_categories = [
            category for category, conc in category_concentrations.items()
            if conc > max_category_conc
        ]

        # Calcular score de diversificación
        diversification_score = self._calculate_diversification_score(
            symbol_concentrations, category_concentrations
        )

        # Generar recomendaciones
        recommendations = self._generate_diversification_recommendations(
            positions, symbol_concentrations, category_concentrations,
            over_concentrated_symbols, over_concentrated_categories
        )

        analysis = DiversificationAnalysis(
            total_value=total_value,
            symbol_concentrations=symbol_concentrations,
            category_concentrations=category_concentrations,
            over_concentrated_symbols=over_concentrated_symbols,
            over_concentrated_categories=over_concentrated_categories,
            diversification_score=diversification_score,
            recommendations=recommendations
        )

        self.last_analysis = analysis
        return analysis

    def _calculate_diversification_score(self, symbol_conc: Dict[str, float],
                                       category_conc: Dict[str, float]) -> float:
        """📈 Calcular score de diversificación (0-100)"""

        # Penalizar concentraciones altas
        symbol_penalty = 0
        for conc in symbol_conc.values():
            if conc > 40:
                symbol_penalty += (conc - 40) * 2
            elif conc > 30:
                symbol_penalty += (conc - 30) * 1

        category_penalty = 0
        for conc in category_conc.values():
            if conc > 60:
                category_penalty += (conc - 60) * 1.5

        # Bonificar diversidad
        num_symbols = len(symbol_conc)
        num_categories = len(category_conc)

        diversity_bonus = min(num_symbols * 10, 30) + min(num_categories * 15, 30)

        # Score base
        base_score = 50
        final_score = base_score + diversity_bonus - symbol_penalty - category_penalty

        return max(0, min(100, final_score))

    def _generate_diversification_recommendations(self, positions: List[PortfolioPosition],
                                                symbol_conc: Dict[str, float],
                                                category_conc: Dict[str, float],
                                                over_symbols: List[str],
                                                over_categories: List[str]) -> List[str]:
        """💡 Generar recomendaciones de diversificación"""

        recommendations = []

        # Alertas de sobre-concentración
        for symbol in over_symbols:
            conc = symbol_conc[symbol]
            recommendations.append(
                f"⚠️ ALTA CONCENTRACIÓN: {symbol} representa {conc:.1f}% del portafolio "
                f"(límite: {self.diversification_config['MAX_SYMBOL_CONCENTRATION_PERCENT']}%)"
            )

        for category in over_categories:
            conc = category_conc[category]
            recommendations.append(
                f"⚠️ CATEGORÍA CONCENTRADA: {category} representa {conc:.1f}% "
                f"(límite: {self.diversification_config['MAX_CATEGORY_CONCENTRATION_PERCENT']}%)"
            )

        # Recomendaciones de diversificación
        if len(symbol_conc) < self.diversification_config['MIN_SYMBOLS_IN_PORTFOLIO']:
            recommendations.append(
                f"💡 DIVERSIFICAR: Considerar agregar más símbolos "
                f"(actual: {len(symbol_conc)}, mínimo: {self.diversification_config['MIN_SYMBOLS_IN_PORTFOLIO']})"
            )

        # Contar posiciones por símbolo
        symbol_position_count = {}
        for pos in positions:
            symbol_position_count[pos.symbol] = symbol_position_count.get(pos.symbol, 0) + 1

        for symbol, count in symbol_position_count.items():
            if count > self.diversification_config['MAX_POSITIONS_PER_SYMBOL']:
                recommendations.append(
                    f"⚠️ MÚLTIPLES POSICIONES: {symbol} tiene {count} posiciones "
                    f"(límite: {self.diversification_config['MAX_POSITIONS_PER_SYMBOL']})"
                )

        if not recommendations:
            recommendations.append("✅ Portafolio bien diversificado")

        return recommendations

    async def should_allow_new_position(self, symbol: str, position_size_usd: float,
                                      current_positions: List[PortfolioPosition]) -> Tuple[bool, str]:
        """🎯 Determinar si se debe permitir una nueva posición"""

        # Analizar portafolio actual
        analysis = await self.analyze_portfolio_diversification(current_positions)

        # Simular nueva posición
        total_value_after = analysis.total_value + position_size_usd

        # Calcular nueva concentración del símbolo
        current_symbol_value = sum(pos.value_usd for pos in current_positions if pos.symbol == symbol)
        new_symbol_concentration = ((current_symbol_value + position_size_usd) / total_value_after) * 100

        # Verificar límite de concentración por símbolo
        max_symbol_conc = self.diversification_config['MAX_SYMBOL_CONCENTRATION_PERCENT']
        if new_symbol_concentration > max_symbol_conc:
            return False, f"Nueva posición excedería límite de concentración para {symbol}: {new_symbol_concentration:.1f}% > {max_symbol_conc}%"

        # Verificar límite de posiciones por símbolo
        current_positions_count = len([pos for pos in current_positions if pos.symbol == symbol])
        max_positions = self.diversification_config['MAX_POSITIONS_PER_SYMBOL']
        if current_positions_count >= max_positions:
            return False, f"Límite de posiciones alcanzado para {symbol}: {current_positions_count}/{max_positions}"

        # Verificar concentración por categoría
        category = self.diversification_config['SYMBOL_CATEGORIES'].get(symbol, 'UNKNOWN')
        current_category_value = sum(
            pos.value_usd for pos in current_positions
            if self.diversification_config['SYMBOL_CATEGORIES'].get(pos.symbol, 'UNKNOWN') == category
        )
        new_category_concentration = ((current_category_value + position_size_usd) / total_value_after) * 100

        max_category_conc = self.diversification_config['MAX_CATEGORY_CONCENTRATION_PERCENT']
        if new_category_concentration > max_category_conc:
            return False, f"Nueva posición excedería límite de categoría {category}: {new_category_concentration:.1f}% > {max_category_conc}%"

        return True, "Posición permitida por diversificación"

    def calculate_diversification_adjusted_size(self, symbol: str, original_size_usd: float,
                                              current_positions: List[PortfolioPosition]) -> float:
        """📏 Ajustar tamaño de posición considerando diversificación"""

        if not current_positions:
            return original_size_usd

        total_value = sum(pos.value_usd for pos in current_positions)
        current_symbol_value = sum(pos.value_usd for pos in current_positions if pos.symbol == symbol)

        # Calcular tamaño máximo permitido
        max_symbol_conc = self.diversification_config['MAX_SYMBOL_CONCENTRATION_PERCENT']
        max_symbol_value = (total_value + original_size_usd) * (max_symbol_conc / 100)
        max_new_position = max_symbol_value - current_symbol_value

        # Aplicar factor de diversificación
        diversification_factor = 1.0 - self.diversification_config['DIVERSIFICATION_PRIORITY']

        # Si hay sobre-concentración, reducir tamaño
        current_concentration = (current_symbol_value / total_value) * 100 if total_value > 0 else 0
        warning_threshold = self.diversification_config['CONCENTRATION_WARNING_THRESHOLD']

        if current_concentration > warning_threshold:
            # Reducir tamaño gradualmente
            reduction_factor = max(0.3, 1.0 - ((current_concentration - warning_threshold) / 20))
            diversification_factor *= reduction_factor

        adjusted_size = min(original_size_usd * diversification_factor, max_new_position)

        return max(0, adjusted_size)

    def get_diversification_priority_symbols(self, available_symbols: List[str],
                                           current_positions: List[PortfolioPosition]) -> List[str]:
        """🎯 Obtener símbolos prioritarios para diversificación"""

        if not current_positions:
            return available_symbols

        # Obtener símbolos y categorías actuales
        current_symbols = set(pos.symbol for pos in current_positions)
        current_categories = set(
            self.diversification_config['SYMBOL_CATEGORIES'].get(pos.symbol, 'UNKNOWN')
            for pos in current_positions
        )

        # Priorizar símbolos de nuevas categorías
        priority_symbols = []
        for symbol in available_symbols:
            category = self.diversification_config['SYMBOL_CATEGORIES'].get(symbol, 'UNKNOWN')

            # Alta prioridad: nueva categoría
            if category not in current_categories:
                priority_symbols.insert(0, symbol)
            # Media prioridad: nuevo símbolo en categoría existente
            elif symbol not in current_symbols:
                priority_symbols.append(symbol)

        # Agregar símbolos existentes al final
        for symbol in available_symbols:
            if symbol not in priority_symbols:
                priority_symbols.append(symbol)

        return priority_symbols

    async def generate_diversification_report(self, positions: List[PortfolioPosition]) -> str:
        """📊 Generar reporte de diversificación"""

        analysis = await self.analyze_portfolio_diversification(positions)

        report = f"""
🎯 REPORTE DE DIVERSIFICACIÓN DE PORTAFOLIO
==========================================
💰 Valor Total: ${analysis.total_value:.2f}
📊 Score de Diversificación: {analysis.diversification_score:.1f}/100

📈 CONCENTRACIÓN POR SÍMBOLO:
"""

        for symbol, conc in sorted(analysis.symbol_concentrations.items(), key=lambda x: x[1], reverse=True):
            status = "🔴" if conc > self.diversification_config['MAX_SYMBOL_CONCENTRATION_PERCENT'] else \
                    "🟡" if conc > self.diversification_config['CONCENTRATION_WARNING_THRESHOLD'] else "🟢"
            report += f"   {status} {symbol}: {conc:.1f}%\n"

        report += f"\n🏷️ CONCENTRACIÓN POR CATEGORÍA:\n"
        for category, conc in sorted(analysis.category_concentrations.items(), key=lambda x: x[1], reverse=True):
            status = "🔴" if conc > self.diversification_config['MAX_CATEGORY_CONCENTRATION_PERCENT'] else "🟢"
            report += f"   {status} {category}: {conc:.1f}%\n"

        report += f"\n💡 RECOMENDACIONES:\n"
        for rec in analysis.recommendations:
            report += f"   {rec}\n"

        return report

    def is_diversification_enabled(self) -> bool:
        """✅ Verificar si la diversificación está habilitada"""
        return self.diversification_config.get('RESPECT_EXISTING_POSITIONS', True)
