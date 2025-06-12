#!/usr/bin/env python3
"""
🗄️ TRADING DATABASE LAYER
Sistema de persistencia para historial de trades, métricas y logs
"""

import asyncio
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any
import json
import os
from dataclasses import dataclass, asdict
import sqlite3
import aiosqlite
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Boolean, Float, Numeric
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import UUID
import uuid
from dotenv import load_dotenv

# Import condicional de asyncpg solo para PostgreSQL
try:
    import asyncpg
    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False
    print("⚠️ asyncpg no disponible - solo SQLite soportado")

load_dotenv()

Base = declarative_base()

class Trade(Base):
    """💼 Tabla de trades ejecutados"""
    __tablename__ = 'trades'

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    symbol = Column(String(20), nullable=False, index=True)
    side = Column(String(4), nullable=False)  # BUY/SELL
    quantity = Column(Numeric(18, 8), nullable=False)
    entry_price = Column(Numeric(18, 8), nullable=False)
    exit_price = Column(Numeric(18, 8), nullable=True)
    entry_time = Column(DateTime(timezone=True), nullable=False)
    exit_time = Column(DateTime(timezone=True), nullable=True)
    pnl_percent = Column(Float, nullable=True)
    pnl_usd = Column(Numeric(18, 8), nullable=True)
    stop_loss = Column(Numeric(18, 8), nullable=True)
    take_profit = Column(Numeric(18, 8), nullable=True)
    exit_reason = Column(String(50), nullable=True)  # STOP_LOSS, TAKE_PROFIT, MANUAL, etc.
    confidence = Column(Float, nullable=False)
    strategy = Column(String(50), nullable=False, default='TCN_ML')
    is_active = Column(Boolean, default=True)
    metadata_json = Column(Text, nullable=True)  # JSON con datos adicionales
    created_at = Column(DateTime(timezone=True), default=datetime.now)

class PerformanceMetric(Base):
    """📊 Tabla de métricas de performance"""
    __tablename__ = 'performance_metrics'

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    timestamp = Column(DateTime(timezone=True), nullable=False)
    total_balance = Column(Numeric(18, 8), nullable=False)
    daily_pnl = Column(Numeric(18, 8), nullable=False)
    total_pnl = Column(Numeric(18, 8), nullable=False)
    daily_return_percent = Column(Float, nullable=False)
    total_return_percent = Column(Float, nullable=False)
    current_drawdown = Column(Float, nullable=False)
    max_drawdown = Column(Float, nullable=False)
    sharpe_ratio = Column(Float, nullable=True)
    win_rate = Column(Float, nullable=False)
    profit_factor = Column(Float, nullable=True)
    active_positions_count = Column(Integer, nullable=False)
    total_exposure_usd = Column(Numeric(18, 8), nullable=False)
    exposure_percent = Column(Float, nullable=False)
    trades_today = Column(Integer, nullable=False)
    avg_trade_duration_minutes = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), default=datetime.now)

class SystemLog(Base):
    """📝 Tabla de logs del sistema"""
    __tablename__ = 'system_logs'

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    timestamp = Column(DateTime(timezone=True), nullable=False)
    level = Column(String(10), nullable=False)  # INFO, WARNING, ERROR, CRITICAL
    category = Column(String(50), nullable=False)  # TRADING, RISK, API, SYSTEM
    message = Column(Text, nullable=False)
    symbol = Column(String(20), nullable=True, index=True)
    metadata_json = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=datetime.now)

class RiskEvent(Base):
    """⚠️ Tabla de eventos de riesgo"""
    __tablename__ = 'risk_events'

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    timestamp = Column(DateTime(timezone=True), nullable=False)
    event_type = Column(String(50), nullable=False)  # CIRCUIT_BREAKER, STOP_LOSS, etc.
    severity = Column(String(20), nullable=False)  # LOW, MEDIUM, HIGH, CRITICAL
    description = Column(Text, nullable=False)
    symbol = Column(String(20), nullable=True)
    action_taken = Column(Text, nullable=True)
    resolved = Column(Boolean, default=False)
    resolved_at = Column(DateTime(timezone=True), nullable=True)
    metadata_json = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=datetime.now)

class MarketDataCache(Base):
    """📈 Cache de datos de mercado"""
    __tablename__ = 'market_data_cache'

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    symbol = Column(String(20), nullable=False, index=True)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    price = Column(Numeric(18, 8), nullable=False)
    volume = Column(Numeric(18, 8), nullable=True)
    data_type = Column(String(20), nullable=False)  # KLINE, TICKER, etc.
    data_json = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=datetime.now)

class TradingDatabase:
    """🗄️ Gestor de base de datos para trading"""

    def __init__(self, database_url: str = None):
        self.database_url = database_url or os.getenv('DATABASE_URL', 'sqlite:///trading_bot.db')
        self.engine = create_engine(self.database_url, echo=False)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

        # Crear tablas si no existen
        Base.metadata.create_all(bind=self.engine)

        print(f"🗄️ Base de datos inicializada: {self.database_url}")

    def get_session(self):
        """🔌 Obtener sesión de base de datos"""
        return self.SessionLocal()

    async def save_trade(self, trade_data: Dict) -> str:
        """💼 Guardar trade en base de datos"""
        try:
            session = self.get_session()

            trade = Trade(
                symbol=trade_data['symbol'],
                side=trade_data['side'],
                quantity=Decimal(str(trade_data['quantity'])),
                entry_price=Decimal(str(trade_data['entry_price'])),
                exit_price=Decimal(str(trade_data.get('exit_price', 0))) if trade_data.get('exit_price') else None,
                entry_time=trade_data['entry_time'],
                exit_time=trade_data.get('exit_time'),
                pnl_percent=trade_data.get('pnl_percent'),
                pnl_usd=Decimal(str(trade_data.get('pnl_usd', 0))) if trade_data.get('pnl_usd') else None,
                stop_loss=Decimal(str(trade_data.get('stop_loss', 0))) if trade_data.get('stop_loss') else None,
                take_profit=Decimal(str(trade_data.get('take_profit', 0))) if trade_data.get('take_profit') else None,
                exit_reason=trade_data.get('exit_reason'),
                confidence=trade_data.get('confidence', 0.0),
                strategy=trade_data.get('strategy', 'TCN_ML'),
                is_active=trade_data.get('is_active', True),
                metadata_json=json.dumps(trade_data.get('metadata', {}))
            )

            session.add(trade)
            session.commit()
            trade_id = trade.id
            session.close()

            print(f"💼 Trade guardado: {trade_data['symbol']} {trade_data['side']} - ID: {trade_id}")
            return trade_id

        except Exception as e:
            print(f"❌ Error guardando trade: {e}")
            if 'session' in locals():
                session.rollback()
                session.close()
            return None

    async def update_trade_exit(self, trade_id: str, exit_data: Dict) -> bool:
        """📉 Actualizar trade al cerrar posición"""
        try:
            session = self.get_session()

            trade = session.query(Trade).filter(Trade.id == trade_id).first()
            if trade:
                trade.exit_price = Decimal(str(exit_data['exit_price']))
                trade.exit_time = exit_data['exit_time']
                trade.pnl_percent = exit_data['pnl_percent']
                trade.pnl_usd = Decimal(str(exit_data['pnl_usd']))
                trade.exit_reason = exit_data['exit_reason']
                trade.is_active = False

                session.commit()
                session.close()

                print(f"📉 Trade actualizado al cerrar: {trade_id}")
                return True

            session.close()
            return False

        except Exception as e:
            print(f"❌ Error actualizando trade: {e}")
            if 'session' in locals():
                session.rollback()
                session.close()
            return False

    async def save_performance_metrics(self, metrics: Dict) -> str:
        """📊 Guardar métricas de performance"""
        try:
            session = self.get_session()

            metric = PerformanceMetric(
                timestamp=datetime.now(timezone.utc),
                total_balance=Decimal(str(metrics['total_balance'])),
                daily_pnl=Decimal(str(metrics['daily_pnl'])),
                total_pnl=Decimal(str(metrics['total_pnl'])),
                daily_return_percent=metrics['daily_return_percent'],
                total_return_percent=metrics['total_return_percent'],
                current_drawdown=metrics['current_drawdown'],
                max_drawdown=metrics.get('max_drawdown', 0.0),
                sharpe_ratio=metrics.get('sharpe_ratio'),
                win_rate=metrics['win_rate'],
                profit_factor=metrics.get('profit_factor'),
                active_positions_count=metrics['active_positions_count'],
                total_exposure_usd=Decimal(str(metrics['total_exposure_usd'])),
                exposure_percent=metrics['exposure_percent'],
                trades_today=metrics['trades_today'],
                avg_trade_duration_minutes=metrics.get('avg_trade_duration_minutes')
            )

            session.add(metric)
            session.commit()
            metric_id = metric.id
            session.close()

            return metric_id

        except Exception as e:
            print(f"❌ Error guardando métricas: {e}")
            if 'session' in locals():
                session.rollback()
                session.close()
            return None

    async def log_event(self, level: str, category: str, message: str, symbol: str = None, metadata: Dict = None):
        """📝 Guardar evento en logs"""
        try:
            session = self.get_session()

            log = SystemLog(
                timestamp=datetime.now(timezone.utc),
                level=level.upper(),
                category=category.upper(),
                message=message,
                symbol=symbol,
                metadata_json=json.dumps(metadata or {})
            )

            session.add(log)
            session.commit()
            session.close()

            # También imprimir en consola
            timestamp = datetime.now().strftime('%H:%M:%S')
            print(f"[{timestamp}] {level} {category}: {message}")

        except Exception as e:
            print(f"❌ Error guardando log: {e}")
            if 'session' in locals():
                session.rollback()
                session.close()

    async def log_risk_event(self, event_type: str, severity: str, description: str,
                           symbol: str = None, action_taken: str = None, metadata: Dict = None):
        """⚠️ Guardar evento de riesgo"""
        try:
            session = self.get_session()

            risk_event = RiskEvent(
                timestamp=datetime.now(timezone.utc),
                event_type=event_type.upper(),
                severity=severity.upper(),
                description=description,
                symbol=symbol,
                action_taken=action_taken,
                metadata_json=json.dumps(metadata or {})
            )

            session.add(risk_event)
            session.commit()
            session.close()

            # Log crítico también en consola
            if severity.upper() in ['HIGH', 'CRITICAL']:
                print(f"🚨 RISK EVENT: {event_type} - {description}")

        except Exception as e:
            print(f"❌ Error guardando evento de riesgo: {e}")
            if 'session' in locals():
                session.rollback()
                session.close()

    async def get_trades_history(self, symbol: str = None, days: int = 30,
                               is_active: bool = None) -> List[Dict]:
        """📈 Obtener historial de trades"""
        try:
            session = self.get_session()

            query = session.query(Trade)

            if symbol:
                query = query.filter(Trade.symbol == symbol)

            if is_active is not None:
                query = query.filter(Trade.is_active == is_active)

            if days:
                cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
                query = query.filter(Trade.entry_time >= cutoff_date)

            trades = query.order_by(Trade.entry_time.desc()).limit(1000).all()

            result = []
            for trade in trades:
                trade_dict = {
                    'id': trade.id,
                    'symbol': trade.symbol,
                    'side': trade.side,
                    'quantity': float(trade.quantity),
                    'entry_price': float(trade.entry_price),
                    'exit_price': float(trade.exit_price) if trade.exit_price else None,
                    'entry_time': trade.entry_time.isoformat(),
                    'exit_time': trade.exit_time.isoformat() if trade.exit_time else None,
                    'pnl_percent': trade.pnl_percent,
                    'pnl_usd': float(trade.pnl_usd) if trade.pnl_usd else None,
                    'exit_reason': trade.exit_reason,
                    'confidence': trade.confidence,
                    'strategy': trade.strategy,
                    'is_active': trade.is_active,
                    'metadata': json.loads(trade.metadata_json) if trade.metadata_json else {}
                }
                result.append(trade_dict)

            session.close()
            return result

        except Exception as e:
            print(f"❌ Error obteniendo historial: {e}")
            if 'session' in locals():
                session.close()
            return []

    async def get_performance_summary(self, days: int = 30) -> Dict:
        """📊 Obtener resumen de performance"""
        try:
            session = self.get_session()

            # Trades cerrados en los últimos días
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
            trades = session.query(Trade).filter(
                Trade.is_active == False,
                Trade.exit_time >= cutoff_date
            ).all()

            if not trades:
                session.close()
                return {'error': 'No hay trades en el período especificado'}

            # Calcular métricas
            total_trades = len(trades)
            winning_trades = [t for t in trades if t.pnl_usd and t.pnl_usd > 0]
            losing_trades = [t for t in trades if t.pnl_usd and t.pnl_usd <= 0]

            win_rate = len(winning_trades) / total_trades * 100

            total_pnl = sum(float(t.pnl_usd) for t in trades if t.pnl_usd)
            avg_win = sum(float(t.pnl_usd) for t in winning_trades) / len(winning_trades) if winning_trades else 0
            avg_loss = sum(float(t.pnl_usd) for t in losing_trades) / len(losing_trades) if losing_trades else 0

            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0

            # Duración promedio
            durations = []
            for trade in trades:
                if trade.exit_time and trade.entry_time:
                    duration = (trade.exit_time - trade.entry_time).total_seconds() / 60
                    durations.append(duration)

            avg_duration = sum(durations) / len(durations) if durations else 0

            summary = {
                'period_days': days,
                'total_trades': total_trades,
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'avg_duration_minutes': avg_duration,
                'best_trade': max((float(t.pnl_usd) for t in trades if t.pnl_usd), default=0),
                'worst_trade': min((float(t.pnl_usd) for t in trades if t.pnl_usd), default=0)
            }

            session.close()
            return summary

        except Exception as e:
            print(f"❌ Error calculando performance: {e}")
            if 'session' in locals():
                session.close()
            return {'error': str(e)}

    async def cleanup_old_data(self, days_to_keep: int = 90):
        """🧹 Limpiar datos antiguos"""
        try:
            session = self.get_session()
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_to_keep)

            # Limpiar logs antiguos
            old_logs = session.query(SystemLog).filter(SystemLog.created_at < cutoff_date).count()
            session.query(SystemLog).filter(SystemLog.created_at < cutoff_date).delete()

            # Limpiar métricas antiguas (mantener solo 1 por día)
            old_metrics = session.query(PerformanceMetric).filter(PerformanceMetric.created_at < cutoff_date).count()
            session.query(PerformanceMetric).filter(PerformanceMetric.created_at < cutoff_date).delete()

            # Limpiar cache de market data
            old_cache = session.query(MarketDataCache).filter(MarketDataCache.created_at < cutoff_date).count()
            session.query(MarketDataCache).filter(MarketDataCache.created_at < cutoff_date).delete()

            session.commit()
            session.close()

            print(f"🧹 Datos antiguos eliminados:")
            print(f"   📝 Logs: {old_logs}")
            print(f"   📊 Métricas: {old_metrics}")
            print(f"   📈 Cache: {old_cache}")

        except Exception as e:
            print(f"❌ Error limpiando datos: {e}")
            if 'session' in locals():
                session.rollback()
                session.close()

    async def backup_database(self, backup_path: str = None) -> str:
        """💾 Crear backup de la base de datos"""
        try:
            if not backup_path:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                backup_path = f"backup_trading_db_{timestamp}.sql"

            # Para SQLite
            if 'sqlite' in self.database_url:
                import shutil
                db_path = self.database_url.replace('sqlite:///', '')
                shutil.copy2(db_path, backup_path)
                print(f"💾 Backup creado: {backup_path}")
                return backup_path

            # Para PostgreSQL (requiere pg_dump)
            elif 'postgresql' in self.database_url:
                import subprocess
                cmd = f"pg_dump {self.database_url} > {backup_path}"
                subprocess.run(cmd, shell=True, check=True)
                print(f"💾 Backup PostgreSQL creado: {backup_path}")
                return backup_path

        except Exception as e:
            print(f"❌ Error creando backup: {e}")
            return None
