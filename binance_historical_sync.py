#!/usr/bin/env python3
"""
🔄 BINANCE HISTORICAL SYNC - Sincronizador de Datos Históricos
Sincroniza datos históricos de la cuenta de Binance con la base de datos local
"""

import asyncio
import aiohttp
import hmac
import hashlib
import time
import json
import os
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import pandas as pd
from trading_database import TradingDatabase
from dotenv import load_dotenv

load_dotenv()

@dataclass
class BinanceConfig:
    """🔧 Configuración de Binance API"""
    api_key: str = ""
    secret_key: str = ""
    testnet: bool = True
    base_url: str = ""
    
    def __post_init__(self):
        self.api_key = self.api_key or os.getenv('BINANCE_API_KEY', '').strip('"')
        self.secret_key = self.secret_key or os.getenv('BINANCE_API_SECRET', '').strip('"') or os.getenv('BINANCE_SECRET_KEY', '').strip('"')
        
        if self.testnet:
            self.base_url = "https://testnet.binance.vision"
        else:
            self.base_url = "https://api.binance.com"

class BinanceHistoricalSync:
    """🔄 Sincronizador de datos históricos de Binance"""
    
    def __init__(self, config: BinanceConfig = None):
        self.config = config or BinanceConfig()
        self.db = TradingDatabase()
        self.session = None
        
        print(f"🔄 Inicializando sincronizador de Binance")
        print(f"   📡 URL: {self.config.base_url}")
        print(f"   🔑 API Key: {'✅ Configurada' if self.config.api_key else '❌ No configurada'}")
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def _generate_signature(self, query_string: str) -> str:
        """🔐 Generar firma HMAC para autenticación"""
        return hmac.new(
            self.config.secret_key.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    def _get_headers(self) -> Dict[str, str]:
        """📋 Headers para requests autenticados"""
        return {
            'X-MBX-APIKEY': self.config.api_key,
            'Content-Type': 'application/json'
        }
    
    async def get_account_info(self) -> Dict:
        """💰 Obtener información completa de la cuenta"""
        print("💰 Obteniendo información de la cuenta...")
        
        timestamp = int(time.time() * 1000)
        query_string = f"timestamp={timestamp}"
        signature = self._generate_signature(query_string)
        
        url = f"{self.config.base_url}/api/v3/account"
        params = {
            'timestamp': timestamp,
            'signature': signature
        }
        
        try:
            async with self.session.get(url, headers=self._get_headers(), params=params) as response:
                if response.status == 200:
                    account_data = await response.json()
                    print(f"✅ Información de cuenta obtenida")
                    print(f"   📅 Can Trade: {account_data.get('canTrade', False)}")
                    print(f"   📈 Can Withdraw: {account_data.get('canWithdraw', False)}")
                    print(f"   💎 Account Type: {account_data.get('accountType', 'UNKNOWN')}")
                    
                    # Mostrar balances principales
                    balances = account_data.get('balances', [])
                    significant_balances = [b for b in balances if float(b['free']) > 0 or float(b['locked']) > 0]
                    
                    print(f"💰 Balances activos encontrados: {len(significant_balances)}")
                    for balance in significant_balances[:10]:  # Mostrar solo los primeros 10
                        total = float(balance['free']) + float(balance['locked'])
                        print(f"   💵 {balance['asset']}: {total:.8f} (libre: {balance['free']}, bloqueado: {balance['locked']})")
                    
                    return account_data
                else:
                    error_text = await response.text()
                    print(f"❌ Error obteniendo cuenta: {response.status} - {error_text}")
                    return {}
        except Exception as e:
            print(f"❌ Excepción obteniendo cuenta: {e}")
            return {}
    
    async def get_historical_trades(self, symbol: str, days_back: int = 30) -> List[Dict]:
        """📈 Obtener historial de trades de un símbolo"""
        print(f"📈 Obteniendo historial de trades para {symbol} ({days_back} días)...")
        
        # Calcular timestamps
        end_time = int(time.time() * 1000)
        start_time = end_time - (days_back * 24 * 60 * 60 * 1000)
        
        timestamp = int(time.time() * 1000)
        query_string = f"symbol={symbol}&startTime={start_time}&endTime={end_time}&timestamp={timestamp}"
        signature = self._generate_signature(query_string)
        
        url = f"{self.config.base_url}/api/v3/myTrades"
        params = {
            'symbol': symbol,
            'startTime': start_time,
            'endTime': end_time,
            'timestamp': timestamp,
            'signature': signature
        }
        
        try:
            async with self.session.get(url, headers=self._get_headers(), params=params) as response:
                if response.status == 200:
                    trades = await response.json()
                    print(f"✅ {len(trades)} trades históricos obtenidos para {symbol}")
                    return trades
                else:
                    error_text = await response.text()
                    print(f"❌ Error obteniendo trades de {symbol}: {response.status} - {error_text}")
                    return []
        except Exception as e:
            print(f"❌ Excepción obteniendo trades de {symbol}: {e}")
            return []
    
    async def get_deposit_history(self, days_back: int = 90) -> List[Dict]:
        """💳 Obtener historial de depósitos"""
        print(f"💳 Obteniendo historial de depósitos ({days_back} días)...")
        
        # Calcular timestamps
        end_time = int(time.time() * 1000)
        start_time = end_time - (days_back * 24 * 60 * 60 * 1000)
        
        timestamp = int(time.time() * 1000)
        query_string = f"startTime={start_time}&endTime={end_time}&timestamp={timestamp}"
        signature = self._generate_signature(query_string)
        
        url = f"{self.config.base_url}/sapi/v1/capital/deposit/hisrec"
        params = {
            'startTime': start_time,
            'endTime': end_time,
            'timestamp': timestamp,
            'signature': signature
        }
        
        try:
            async with self.session.get(url, headers=self._get_headers(), params=params) as response:
                if response.status == 200:
                    deposits = await response.json()
                    print(f"✅ {len(deposits)} depósitos históricos obtenidos")
                    return deposits
                else:
                    error_text = await response.text()
                    print(f"❌ Error obteniendo depósitos: {response.status} - {error_text}")
                    return []
        except Exception as e:
            print(f"❌ Excepción obteniendo depósitos: {e}")
            return []
    
    async def sync_balance_to_database(self, account_info: Dict) -> bool:
        """💾 Sincronizar balance actual con la base de datos"""
        print("💾 Sincronizando balance con base de datos...")
        
        try:
            balances = account_info.get('balances', [])
            usdt_balance = 0.0
            total_balance_usd = 0.0
            
            # Calcular balance USDT principal
            for balance in balances:
                if balance['asset'] == 'USDT':
                    usdt_balance = float(balance['free']) + float(balance['locked'])
                    total_balance_usd = usdt_balance  # Por simplicidad, usar USDT como base
                    break
            
            # Crear métrica de performance inicial
            performance_data = {
                'timestamp': datetime.now(timezone.utc),
                'total_balance': total_balance_usd,
                'daily_pnl': 0.0,
                'total_pnl': 0.0,
                'daily_return_percent': 0.0,
                'total_return_percent': 0.0,
                'current_drawdown': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': None,
                'win_rate': 0.0,
                'profit_factor': None,
                'active_positions_count': 0,
                'total_exposure_usd': 0.0,
                'exposure_percent': 0.0,
                'trades_today': 0,
                'avg_trade_duration_minutes': None
            }
            
            await self.db.save_performance_metrics(performance_data)
            
            # Log del evento
            await self.db.log_event(
                level="INFO",
                category="SYNC",
                message=f"Balance inicial sincronizado: ${total_balance_usd:.2f} USDT",
                metadata={
                    'sync_type': 'initial_balance',
                    'total_assets': len([b for b in balances if float(b['free']) > 0 or float(b['locked']) > 0]),
                    'account_type': account_info.get('accountType', 'UNKNOWN'),
                    'can_trade': account_info.get('canTrade', False)
                }
            )
            
            print(f"✅ Balance sincronizado: ${total_balance_usd:.2f} USDT")
            return True
            
        except Exception as e:
            print(f"❌ Error sincronizando balance: {e}")
            return False
    
    async def sync_historical_trades_to_database(self, symbol: str, historical_trades: List[Dict]) -> int:
        """📊 Sincronizar trades históricos con la base de datos"""
        print(f"📊 Sincronizando trades históricos de {symbol}...")
        
        synced_count = 0
        
        try:
            for trade in historical_trades:
                # Convertir trade de Binance a formato interno
                trade_data = {
                    'symbol': trade['symbol'],
                    'side': 'BUY' if trade['isBuyer'] else 'SELL',
                    'quantity': float(trade['qty']),
                    'entry_price': float(trade['price']),
                    'entry_time': datetime.fromtimestamp(trade['time'] / 1000, timezone.utc),
                    'exit_time': None,  # Los trades de Binance son instantáneos
                    'pnl_percent': None,
                    'pnl_usd': None,
                    'stop_loss': None,
                    'take_profit': None,
                    'exit_reason': 'FILLED',
                    'confidence': 1.0,  # Trade histórico ya ejecutado
                    'strategy': 'HISTORICAL_SYNC',
                    'is_active': False,  # Trade ya completado
                    'metadata': {
                        'binance_trade_id': trade['id'],
                        'commission': trade['commission'],
                        'commission_asset': trade['commissionAsset'],
                        'is_maker': trade['isMaker'],
                        'quote_qty': trade['quoteQty'],
                        'sync_source': 'binance_historical'
                    }
                }
                
                # Guardar en base de datos
                trade_id = await self.db.save_trade(trade_data)
                if trade_id:
                    synced_count += 1
                
                # Pequeña pausa para evitar sobrecarga
                await asyncio.sleep(0.1)
            
            print(f"✅ {synced_count}/{len(historical_trades)} trades históricos sincronizados para {symbol}")
            return synced_count
            
        except Exception as e:
            print(f"❌ Error sincronizando trades históricos de {symbol}: {e}")
            return synced_count
    
    async def calculate_initial_pnl(self, historical_trades: List[Dict]) -> Dict:
        """💹 Calcular PnL inicial basado en trades históricos"""
        print("💹 Calculando PnL inicial...")
        
        try:
            if not historical_trades:
                return {'total_pnl': 0.0, 'total_volume': 0.0, 'trade_count': 0}
            
            total_buy_value = 0.0
            total_sell_value = 0.0
            total_commission = 0.0
            total_volume = 0.0
            
            for trade in historical_trades:
                qty = float(trade['qty'])
                price = float(trade['price'])
                commission = float(trade['commission'])
                value = qty * price
                total_volume += value
                total_commission += commission
                
                if trade['isBuyer']:
                    total_buy_value += value
                else:
                    total_sell_value += value
            
            # PnL simple (ventas - compras - comisiones)
            total_pnl = total_sell_value - total_buy_value - total_commission
            
            pnl_summary = {
                'total_pnl': total_pnl,
                'total_volume': total_volume,
                'total_commission': total_commission,
                'trade_count': len(historical_trades),
                'buy_value': total_buy_value,
                'sell_value': total_sell_value
            }
            
            print(f"💹 PnL calculado: ${total_pnl:.2f} (volumen: ${total_volume:.2f})")
            return pnl_summary
            
        except Exception as e:
            print(f"❌ Error calculando PnL: {e}")
            return {'total_pnl': 0.0, 'total_volume': 0.0, 'trade_count': 0}
    
    async def full_sync(self, trading_symbols: List[str] = None, days_back: int = 30) -> Dict:
        """🔄 Sincronización completa de datos históricos"""
        if trading_symbols is None:
            trading_symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
        
        print("🔄 INICIANDO SINCRONIZACIÓN COMPLETA")
        print("=" * 50)
        
        sync_results = {
            'success': False,
            'balance_synced': False,
            'symbols_synced': [],
            'total_trades_synced': 0,
            'total_pnl': 0.0,
            'errors': []
        }
        
        try:
            # 1. Verificar API keys
            if not self.config.api_key or not self.config.secret_key:
                error_msg = "❌ API keys no configuradas"
                print(error_msg)
                sync_results['errors'].append(error_msg)
                return sync_results
            
            # 2. Obtener información de cuenta
            print("\n📋 PASO 1: Información de cuenta")
            account_info = await self.get_account_info()
            if not account_info:
                error_msg = "❌ No se pudo obtener información de cuenta"
                sync_results['errors'].append(error_msg)
                return sync_results
            
            # 3. Sincronizar balance
            print("\n💰 PASO 2: Sincronización de balance")
            balance_synced = await self.sync_balance_to_database(account_info)
            sync_results['balance_synced'] = balance_synced
            
            # 4. Sincronizar trades históricos por símbolo
            print("\n📈 PASO 3: Sincronización de trades históricos")
            all_historical_trades = []
            
            for symbol in trading_symbols:
                print(f"\n🔍 Procesando {symbol}...")
                historical_trades = await self.get_historical_trades(symbol, days_back)
                
                if historical_trades:
                    synced_count = await self.sync_historical_trades_to_database(symbol, historical_trades)
                    sync_results['symbols_synced'].append({
                        'symbol': symbol,
                        'trades_count': len(historical_trades),
                        'synced_count': synced_count
                    })
                    sync_results['total_trades_synced'] += synced_count
                    all_historical_trades.extend(historical_trades)
                else:
                    print(f"⚠️  No se encontraron trades históricos para {symbol}")
            
            # 5. Calcular métricas iniciales
            print("\n💹 PASO 4: Cálculo de métricas")
            pnl_summary = await self.calculate_initial_pnl(all_historical_trades)
            sync_results['total_pnl'] = pnl_summary['total_pnl']
            
            # 6. Log de finalización
            await self.db.log_event(
                level="INFO",
                category="SYNC",
                message=f"Sincronización histórica completada: {sync_results['total_trades_synced']} trades, PnL: ${pnl_summary['total_pnl']:.2f}",
                metadata={
                    'sync_type': 'full_historical',
                    'symbols_processed': len(trading_symbols),
                    'days_back': days_back,
                    'total_volume': pnl_summary['total_volume'],
                    'total_commission': pnl_summary.get('total_commission', 0)
                }
            )
            
            sync_results['success'] = True
            
            print("\n✅ SINCRONIZACIÓN COMPLETA FINALIZADA")
            print("=" * 50)
            print(f"💰 Balance sincronizado: {'✅' if balance_synced else '❌'}")
            print(f"📊 Symbols procesados: {len(sync_results['symbols_synced'])}")
            print(f"📈 Trades sincronizados: {sync_results['total_trades_synced']}")
            print(f"💹 PnL total calculado: ${sync_results['total_pnl']:.2f}")
            
            return sync_results
            
        except Exception as e:
            error_msg = f"❌ Error en sincronización completa: {e}"
            print(error_msg)
            sync_results['errors'].append(error_msg)
            return sync_results

async def main():
    """🚀 Función principal de sincronización"""
    print("🔄 BINANCE HISTORICAL SYNC")
    print("=" * 50)
    
    # Configuración - USANDO PRODUCCIÓN
    config = BinanceConfig(
        testnet=False,  # 🔴 PRODUCCIÓN - Cuenta real
    )
    
    # Símbolos a sincronizar
    trading_symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
    days_back = 30  # Últimos 30 días
    
    print(f"🎯 Configuración:")
    print(f"   📡 Entorno: {'Testnet' if config.testnet else 'Producción'}")
    print(f"   💱 Símbolos: {', '.join(trading_symbols)}")
    print(f"   📅 Días atrás: {days_back}")
    print(f"   🔑 API configurada: {'✅' if config.api_key else '❌'}")
    
    if not config.api_key:
        print("\n❌ ERROR: Configura las variables de entorno:")
        print("   export BINANCE_API_KEY='tu_api_key'")
        print("   export BINANCE_API_SECRET='tu_api_secret'")
        return
    
    # Ejecutar sincronización
    async with BinanceHistoricalSync(config) as sync:
        results = await sync.full_sync(trading_symbols, days_back)
        
        if results['success']:
            print(f"\n🎉 SINCRONIZACIÓN EXITOSA!")
            print(f"   ✅ Balance: {'Sincronizado' if results['balance_synced'] else 'Error'}")
            print(f"   📊 Trades: {results['total_trades_synced']} sincronizados")
            print(f"   💹 PnL: ${results['total_pnl']:.2f}")
        else:
            print(f"\n❌ SINCRONIZACIÓN CON ERRORES:")
            for error in results['errors']:
                print(f"   ❌ {error}")

if __name__ == "__main__":
    asyncio.run(main()) 