import ccxt.async_support as ccxt
import pandas as pd
import numpy as np
import ta
import talib
from datetime import datetime, timedelta
import time
import logging
import json
import os
import asyncio
from collections import defaultdict, deque
import traceback
from scipy import stats
import yfinance as yf
import aiohttp
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
import warnings
warnings.filterwarnings('ignore')

# Настройка логирования
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Форматировщик
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
formatter = logging.Formatter(log_format)

# Файловый обработчик
file_handler = logging.FileHandler('grid_advisor.log', encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

# Консольный обработчик
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)
logger.propagate = False

@dataclass
class GridRecommendation:
    """Класс для хранения рекомендаций по сетке"""
    symbol: str
    recommendation: str  # "STRONG_BUY", "BUY", "NEUTRAL", "SELL", "STRONG_SELL"
    direction: str  # "LONG", "SHORT", "NEUTRAL"
    confidence: float  # 0-100%
    timeframe: str  # Рекомендуемый таймфрейм для сетки
    entry_range: Tuple[float, float]  # Диапазон входа
    take_profit_levels: List[float]  # Уровни тейк-профита
    stop_loss: float  # Уровень стоп-лосса
    support_levels: List[float] = field(default_factory=list)  # Уровни поддержки
    resistance_levels: List[float] = field(default_factory=list)  # Уровни сопротивления
    volatility: float = 0.0  # Волатильность (ATR %)
    grid_spacing: float = 0.0  # Расстояние между ордерами в %
    grid_levels: int = 5  # Количество уровней в сетке
    position_size: float = 0.0  # Рекомендуемый размер позиции в %
    expected_duration: str = ""  # Ожидаемая длительность сетки
    risk_reward: float = 0.0  # Соотношение риск/вознаграждение
    market_regime: str = ""  # Рыночный режим
    trailing_up: float = 0.0  # Трейлинг ап в %
    trailing_down: float = 0.0  # Трейлинг даун в %
    notes: List[str] = field(default_factory=list)  # Дополнительные заметки
    
class MarketAnalyzer:
    """Анализатор рынка для торговых сеток"""
    
    def __init__(self, exchange, config=None):
        self.exchange = exchange
        self.config = config or {
            'min_volume_24h': 1000000,  # Минимальный объем 1M USDT
            'min_price': 0.01,
            'max_volatility': 0.15,  # Максимальная волатильность
            'grid_levels': 5,  # Количество уровней в сетке
            'timeframes': ['5m', '15m', '30m', '1h', '4h', '1d'],
            'support_resistance_lookback': 100
        }
        self.data_cache = {}
        
    async def get_trading_pairs(self):
        """Получение торговых пар с фильтрацией"""
        try:
            markets = await self.exchange.load_markets()
            pairs = []
            
            for symbol, market in markets.items():
                if (market.get('quote') == 'USDT' and 
                    market.get('spot', False) and 
                    market.get('active', False)):
                    
                    # Проверка объема
                    try:
                        ticker = await self.exchange.fetch_ticker(symbol)
                        volume_24h = ticker.get('quoteVolume', 0)
                        
                        if volume_24h > self.config['min_volume_24h']:
                            pairs.append(symbol)
                    except Exception as e:
                        logger.debug(f"Ошибка получения тикера для {symbol}: {e}")
                        continue
                        
            logger.info(f"Найдено {len(pairs)} ликвидных пар")
            return pairs
            
        except Exception as e:
            logger.error(f"Ошибка получения торговых пар: {e}")
            return []
    
    async def fetch_ohlcv(self, symbol, timeframe, limit=200):
        """Получение OHLCV данных"""
        cache_key = f"{symbol}_{timeframe}"
        
        if cache_key in self.data_cache:
            cached_time, data = self.data_cache[cache_key]
            if time.time() - cached_time < 300:  # Кэш на 5 минут
                return data
        
        try:
            ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            self.data_cache[cache_key] = (time.time(), df)
            return df
            
        except Exception as e:
            logger.error(f"Ошибка получения данных {symbol} {timeframe}: {e}")
            return None
    
    def calculate_indicators(self, df):
        """Расчет технических индикаторов"""
        if df is None or len(df) < 50:
            return df
        
        df = df.copy()
        
        # Трендовые индикаторы
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
        
        # Осцилляторы
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        df['macd'] = ta.trend.MACD(df['close']).macd()
        df['macd_signal'] = ta.trend.MACD(df['close']).macd_signal()
        
        # Волатильность
        df['atr'] = ta.volatility.AverageTrueRange(
            df['high'], df['low'], df['close'], window=14
        ).average_true_range()
        df['atr_pct'] = (df['atr'] / df['close'] * 100).fillna(0)
        
        # Боллинджер
        bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_middle'] = bb.bollinger_mavg()
        
        # Объем
        if 'volume' in df.columns:
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = (df['volume'] / df['volume_sma']).fillna(1)
        
        return df
    
    def find_support_resistance(self, df, lookback=100):
        """Поиск уровней поддержки и сопротивления"""
        if df is None or len(df) < 50:
            return [], []
        
        recent_data = df.tail(lookback)
        
        # Используем метод скользящих окон для поиска экстремумов
        support_levels = []
        resistance_levels = []
        
        # Находим локальные минимумы и максимумы
        window = 5
        for i in range(window, len(recent_data) - window):
            local_low = recent_data['low'].iloc[i-window:i+window+1].min()
            local_high = recent_data['high'].iloc[i-window:i+window+1].max()
            
            if recent_data['low'].iloc[i] == local_low:
                support_levels.append(recent_data['low'].iloc[i])
            
            if recent_data['high'].iloc[i] == local_high:
                resistance_levels.append(recent_data['high'].iloc[i])
        
        # Группировка близких уровней
        if support_levels:
            support_levels = self._cluster_levels(support_levels, threshold=0.005)
        if resistance_levels:
            resistance_levels = self._cluster_levels(resistance_levels, threshold=0.005)
        
        # Сортируем и берем ближайшие уровни
        current_price = df['close'].iloc[-1]
        
        if support_levels:
            support_levels = sorted([s for s in support_levels if s < current_price], reverse=True)[:3]
        if resistance_levels:
            resistance_levels = sorted([r for r in resistance_levels if r > current_price])[:3]
        
        return support_levels, resistance_levels
    
    def _cluster_levels(self, levels, threshold=0.005):
        """Группировка близких уровней"""
        if not levels:
            return []
        
        levels = sorted(levels)
        clusters = []
        current_cluster = [levels[0]]
        
        for level in levels[1:]:
            if abs(level - current_cluster[-1]) / current_cluster[-1] < threshold:
                current_cluster.append(level)
            else:
                clusters.append(np.mean(current_cluster))
                current_cluster = [level]
        
        if current_cluster:
            clusters.append(np.mean(current_cluster))
        
        return clusters
    
    def analyze_market_regime(self, df):
        """Анализ рыночного режима"""
        if df is None or len(df) < 50:
            return "UNKNOWN", 0.0
        
        try:
            current_price = df['close'].iloc[-1]
            sma_20 = df['sma_20'].iloc[-1] if 'sma_20' in df.columns else current_price
            sma_50 = df['sma_50'].iloc[-1] if 'sma_50' in df.columns else current_price
            rsi = df['rsi'].iloc[-1] if 'rsi' in df.columns else 50
            atr_pct = df['atr_pct'].iloc[-1] if 'atr_pct' in df.columns else 2.0
            
            # Определение тренда
            trend_strength = abs((sma_20 - sma_50) / sma_50) if sma_50 > 0 else 0
            
            if sma_20 > sma_50 and trend_strength > 0.02:
                regime = "STRONG_UPTREND"
            elif sma_20 > sma_50:
                regime = "UPTREND"
            elif sma_20 < sma_50 and trend_strength > 0.02:
                regime = "STRONG_DOWNTREND"
            elif sma_20 < sma_50:
                regime = "DOWNTREND"
            else:
                regime = "RANGING"
            
            # Корректировка на основе RSI и волатильности
            if regime == "RANGING":
                if atr_pct < 1.0:
                    regime = "LOW_VOLATILITY_RANGE"
                elif atr_pct > 3.0:
                    regime = "HIGH_VOLATILITY_RANGE"
            
            return regime, trend_strength
            
        except Exception as e:
            logger.error(f"Ошибка анализа рыночного режима: {e}")
            return "UNKNOWN", 0.0
    
    def calculate_grid_parameters(self, df, current_price, volatility, direction):
        """Расчет параметров сетки"""
        # Базовые настройки
        base_grid_levels = self.config['grid_levels']
        
        # Адаптация к волатильности
        if volatility < 1.0:
            grid_spacing_pct = 0.5
            grid_levels = base_grid_levels + 2  # Больше уровней при низкой волатильности
        elif volatility < 2.0:
            grid_spacing_pct = 1.0
            grid_levels = base_grid_levels
        elif volatility < 3.0:
            grid_spacing_pct = 1.5
            grid_levels = base_grid_levels - 1
        elif volatility < 5.0:
            grid_spacing_pct = 2.0
            grid_levels = base_grid_levels - 2
        else:
            grid_spacing_pct = 3.0
            grid_levels = max(3, base_grid_levels - 3)  # Минимум 3 уровня
        
        # Расчет TP уровней
        tp_levels = []
        tp_multiplier = 1.2  # Коэффициент для TP
        
        if direction == "LONG":
            for i in range(1, grid_levels + 1):
                tp_price = current_price * (1 + (grid_spacing_pct * i * tp_multiplier / 100))
                tp_levels.append(tp_price)
            # Стоп-лосс
            sl_pct = grid_spacing_pct * grid_levels * 0.8
            stop_loss = current_price * (1 - sl_pct / 100)
            
        elif direction == "SHORT":
            for i in range(1, grid_levels + 1):
                tp_price = current_price * (1 - (grid_spacing_pct * i * tp_multiplier / 100))
                tp_levels.append(tp_price)
            # Стоп-лосс
            sl_pct = grid_spacing_pct * grid_levels * 0.8
            stop_loss = current_price * (1 + sl_pct / 100)
            
        else:  # NEUTRAL
            # Для нейтральной сетки используем симметричные TP
            for i in range(1, grid_levels + 1):
                tp_up = current_price * (1 + (grid_spacing_pct * i * tp_multiplier / 100))
                tp_down = current_price * (1 - (grid_spacing_pct * i * tp_multiplier / 100))
                tp_levels.extend([tp_up, tp_down])
            # Стоп-лосс для нейтральной стратегии
            sl_pct = grid_spacing_pct * grid_levels * 1.0
            stop_loss_up = current_price * (1 + sl_pct / 100)
            stop_loss_down = current_price * (1 - sl_pct / 100)
            stop_loss = (stop_loss_up, stop_loss_down)
        
        # Расчет размера позиции
        position_size = 100 / grid_levels  # Равномерное распределение
        
        # Расчет трейлинга
        trailing_up, trailing_down = self._calculate_trailing_params(volatility, direction)
        
        return {
            'grid_spacing_pct': grid_spacing_pct,
            'grid_levels': grid_levels,
            'tp_levels': sorted(tp_levels),
            'stop_loss': stop_loss,
            'position_size': position_size,
            'expected_duration': self._estimate_duration(volatility),
            'trailing_up': trailing_up,
            'trailing_down': trailing_down
        }
    
    def _calculate_trailing_params(self, volatility, direction):
        """Расчет параметров трейлинга"""
        # Базовые значения трейлинга (в %)
        if volatility < 1.0:
            base_trailing = 0.3
        elif volatility < 2.0:
            base_trailing = 0.5
        elif volatility < 3.0:
            base_trailing = 0.8
        elif volatility < 5.0:
            base_trailing = 1.2
        else:
            base_trailing = 1.5
        
        # Активация трейлинга (отступ от цены в %)
        activation_pct = base_trailing * 1.5
        
        if direction == "LONG":
            trailing_up = base_trailing
            trailing_down = base_trailing * 0.5  # Меньший trailing_down для лонга
        elif direction == "SHORT":
            trailing_down = base_trailing
            trailing_up = base_trailing * 0.5  # Меньший trailing_up для шорта
        else:  # NEUTRAL
            trailing_up = base_trailing
            trailing_down = base_trailing
        
        return trailing_up, trailing_down
    
    def _estimate_duration(self, volatility):
        """Оценка длительности сетки"""
        if volatility < 1.0:
            return "1-3 дня"  # Низкая волатильность
        elif volatility < 2.0:
            return "12-24 часа"  # Средняя волатильность
        elif volatility < 3.0:
            return "4-8 часов"  # Высокая волатильность
        else:
            return "2-4 часа"  # Очень высокая волатильность

class GridAdvisor:
    """Советник по торговым сеткам"""
    
    def __init__(self, exchange_id='binance'):
        self.exchange = self._init_exchange(exchange_id)
        self.analyzer = MarketAnalyzer(self.exchange)
        self.recommendations = []
        
        # Настройки для рекомендаций
        self.settings = {
            'max_recommendations': 10,
            'min_confidence': 60,
            'preferred_timeframes': ['1h', '4h'],
            'risk_free_rate': 0.05  # 5% годовых
        }
        
        logger.info("Советник по торговым сеткам инициализирован")
    
    def _init_exchange(self, exchange_id):
        """Инициализация биржи"""
        exchange_class = getattr(ccxt, exchange_id)
        return exchange_class({
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'},
            'timeout': 30000
        })
    
    async def analyze_pair(self, symbol):
        """Анализ одной пары для сеточной торговли"""
        try:
            # Получаем данные для разных таймфреймов
            data = {}
            for tf in self.settings['preferred_timeframes']:
                df = await self.analyzer.fetch_ohlcv(symbol, tf)
                if df is not None and len(df) > 50:
                    df = self.analyzer.calculate_indicators(df)
                    data[tf] = df
            
            if not data:
                return None
            
            # Используем данные с наибольшим таймфреймом для анализа
            tf = self.settings['preferred_timeframes'][-1]
            df = data[tf]
            current_price = df['close'].iloc[-1]
            
            # Анализируем рыночный режим
            market_regime, trend_strength = self.analyzer.analyze_market_regime(df)
            
            # Находим уровни поддержки/сопротивления
            support_levels, resistance_levels = self.analyzer.find_support_resistance(df)
            
            # Анализируем волатильность
            volatility = df['atr_pct'].iloc[-1] if 'atr_pct' in df.columns else 2.0
            
            # Определяем направление
            direction, confidence = self._determine_direction(df, market_regime)
            
            # Рассчитываем параметры сетки
            grid_params = self.analyzer.calculate_grid_parameters(df, current_price, volatility, direction)
            
            # Определяем рекомендацию
            recommendation = self._generate_recommendation(
                direction, confidence, market_regime, volatility
            )
            
            # Определяем лучший таймфрейм для сетки
            best_timeframe = self._select_best_timeframe(data, volatility)
            
            # Рассчитываем диапазон входа
            entry_range = self._calculate_entry_range(
                current_price, support_levels, resistance_levels, direction, volatility
            )
            
            # Рассчитываем риск/вознаграждение
            if direction == "LONG" and grid_params['tp_levels']:
                risk_reward = self._calculate_risk_reward(
                    current_price, 
                    grid_params['stop_loss'], 
                    grid_params['tp_levels'][-1]
                )
            elif direction == "SHORT" and grid_params['tp_levels']:
                risk_reward = self._calculate_risk_reward(
                    current_price, 
                    grid_params['stop_loss'], 
                    grid_params['tp_levels'][0]
                )
            else:
                risk_reward = 1.5  # По умолчанию
            
            # Формируем рекомендацию
            grid_rec = GridRecommendation(
                symbol=symbol,
                recommendation=recommendation,
                direction=direction,
                confidence=confidence,
                timeframe=best_timeframe,
                entry_range=entry_range,
                take_profit_levels=grid_params['tp_levels'],
                stop_loss=grid_params['stop_loss'],
                support_levels=support_levels,
                resistance_levels=resistance_levels,
                volatility=volatility,
                grid_spacing=grid_params['grid_spacing_pct'],
                grid_levels=grid_params['grid_levels'],
                position_size=grid_params['position_size'],
                expected_duration=grid_params['expected_duration'],
                risk_reward=risk_reward,
                market_regime=market_regime,
                trailing_up=grid_params['trailing_up'],
                trailing_down=grid_params['trailing_down'],
                notes=self._generate_notes(df, market_regime, direction, volatility)
            )
            
            return grid_rec
            
        except Exception as e:
            logger.error(f"Ошибка анализа пары {symbol}: {e}")
            logger.error(traceback.format_exc())
            return None
    
    def _determine_direction(self, df, market_regime):
        """Определение направления для сетки"""
        current_price = df['close'].iloc[-1]
        rsi = df['rsi'].iloc[-1] if 'rsi' in df.columns else 50
        macd = df['macd'].iloc[-1] if 'macd' in df.columns else 0
        macd_signal = df['macd_signal'].iloc[-1] if 'macd_signal' in df.columns else 0
        
        # Собираем сигналы
        signals = []
        weights = []
        
        # RSI сигналы
        if rsi < 30:
            signals.append('LONG')
            weights.append(1.2)
        elif rsi > 70:
            signals.append('SHORT')
            weights.append(1.2)
        elif 40 < rsi < 60:
            signals.append('NEUTRAL')
            weights.append(0.8)
        
        # MACD сигналы
        if macd > macd_signal:
            signals.append('LONG')
            weights.append(0.9)
        elif macd < macd_signal:
            signals.append('SHORT')
            weights.append(0.9)
        
        # Режим рынка
        if 'UPTREND' in market_regime:
            signals.append('LONG')
            weights.append(1.1)
        elif 'DOWNTREND' in market_regime:
            signals.append('SHORT')
            weights.append(1.1)
        elif 'RANGING' in market_regime or 'LOW_VOLATILITY' in market_regime:
            signals.append('NEUTRAL')
            weights.append(1.0)
        
        # Подсчет баллов
        scores = {'LONG': 0, 'SHORT': 0, 'NEUTRAL': 0}
        
        for direction, weight in zip(signals, weights):
            scores[direction] += weight
        
        # Определяем направление с максимальным баллом
        max_score = max(scores.values())
        directions = [d for d, s in scores.items() if s == max_score]
        
        if len(directions) == 1:
            direction = directions[0]
        else:
            # При равенстве баллов выбираем NEUTRAL
            direction = 'NEUTRAL'
        
        # Рассчитываем уверенность
        total_score = sum(scores.values())
        confidence = (max_score / total_score * 100) if total_score > 0 else 50
        
        return direction, min(100, confidence)
    
    def _generate_recommendation(self, direction, confidence, market_regime, volatility):
        """Генерация текстовой рекомендации"""
        if confidence >= 80:
            strength = "STRONG_"
        elif confidence >= 60:
            strength = ""
        else:
            return "NEUTRAL"
        
        if direction == 'LONG':
            return f"{strength}BUY"
        elif direction == 'SHORT':
            return f"{strength}SELL"
        else:
            return "NEUTRAL"
    
    def _select_best_timeframe(self, data_dict, volatility):
        """Выбор лучшего таймфрейма для сетки"""
        # Предпочтение отдаем таймфреймам в зависимости от волатильности
        if volatility < 1.0:
            # Низкая волатильность - более длинные ТФ
            preferred = ['4h', '1h', '30m']
        elif volatility < 3.0:
            # Средняя волатильность
            preferred = ['1h', '30m', '15m']
        else:
            # Высокая волатильность - более короткие ТФ
            preferred = ['30m', '15m', '5m']
        
        for tf in preferred:
            if tf in data_dict:
                return tf
        
        # Возвращаем первый доступный таймфрейм
        return list(data_dict.keys())[0] if data_dict else '1h'
    
    def _calculate_entry_range(self, current_price, supports, resistances, direction, volatility):
        """Расчет диапазона входа для сетки"""
        # Базовый диапазон на основе волатильности
        base_range = volatility * 1.5  # % от цены
        
        if direction == 'LONG':
            # Для лонга: от текущей цены до ближайшей поддержки
            if supports:
                entry_low = min(current_price * (1 - base_range/100), supports[0] * 0.995)
                entry_high = current_price * (1 + base_range/200)  # Уже сверху
            else:
                entry_low = current_price * (1 - base_range/100)
                entry_high = current_price * (1 + base_range/200)
                
        elif direction == 'SHORT':
            # Для шорта: от текущей цены до ближайшего сопротивления
            if resistances:
                entry_low = current_price * (1 - base_range/200)  # Уже снизу
                entry_high = min(current_price * (1 + base_range/100), resistances[0] * 1.005)
            else:
                entry_low = current_price * (1 - base_range/200)
                entry_high = current_price * (1 + base_range/100)
                
        else:  # NEUTRAL
            # Для нейтральной сетки: симметричный диапазон
            entry_low = current_price * (1 - base_range/100)
            entry_high = current_price * (1 + base_range/100)
        
        return (round(entry_low, 8), round(entry_high, 8))
    
    def _calculate_risk_reward(self, entry, stop_loss, take_profit):
        """Расчет соотношения риск/вознаграждение"""
        risk = abs(entry - stop_loss)
        reward = abs(take_profit - entry)
        
        return round(reward / risk, 2) if risk > 0 else 0
    
    def _generate_notes(self, df, market_regime, direction, volatility):
        """Генерация заметок и предупреждений"""
        notes = []
        current_price = df['close'].iloc[-1]
        rsi = df['rsi'].iloc[-1] if 'rsi' in df.columns else 50
        
        # Заметки по RSI
        if rsi < 30:
            notes.append("RSI в зоне перепроданности - возможен отскок")
        elif rsi > 70:
            notes.append("RSI в зоне перекупленности - возможна коррекция")
        
        # Заметки по волатильности
        if volatility < 1.0:
            notes.append("Очень низкая волатильность - используйте узкую сетку")
        elif volatility > 3.0:
            notes.append("Высокая волатильность - будьте осторожны")
        
        # Заметки по режиму
        if 'STRONG' in market_regime:
            notes.append("Сильный тренд - используйте меньше уровней")
        elif 'RANGING' in market_regime:
            notes.append("Рынок в боковике - идеально для сетки")
        
        # Рекомендации по направлению
        if direction == 'NEUTRAL':
            notes.append("Рекомендуется симметричная сетка")
        elif direction == 'LONG':
            notes.append("Фокусируйтесь на покупках на откатах")
        elif direction == 'SHORT':
            notes.append("Фокусируйтесь на продажах на отскоках")
        
        return notes
    
    async def get_top_recommendations(self, limit=10):
        """Получение лучших рекомендаций"""
        # Получаем все пары
        pairs = await self.analyzer.get_trading_pairs()
        
        if not pairs:
            logger.warning("Не найдено торговых пар для анализа")
            return []
        
        # Анализируем каждую пару
        recommendations = []
        
        logger.info(f"Начинаем анализ {len(pairs)} пар...")
        
        for i, symbol in enumerate(pairs[:50]):  # Ограничиваем для скорости
            logger.info(f"Анализируем пару {i+1}/{min(50, len(pairs))}: {symbol}")
            
            recommendation = await self.analyze_pair(symbol)
            if recommendation and recommendation.confidence >= self.settings['min_confidence']:
                recommendations.append(recommendation)
            
            # Небольшая пауза для избежания лимитов
            await asyncio.sleep(0.2)
        
        # Сортируем по уверенности
        recommendations.sort(key=lambda x: x.confidence, reverse=True)
        
        # Сохраняем рекомендации
        self.recommendations = recommendations[:limit]
        
        logger.info(f"Получено {len(self.recommendations)} рекомендаций")
        
        return self.recommendations
    
    def print_recommendations(self):
        """Красивый вывод рекомендаций"""
        if not self.recommendations:
            print("\n❌ Нет рекомендаций для отображения")
            return
        
        print("\n" + "="*120)
        print("🎯 ЛУЧШИЕ ПАРЫ ДЛЯ ТОРГОВЫХ СЕТОК")
        print("="*120)
        
        for i, rec in enumerate(self.recommendations, 1):
            print(f"\n{i}. {rec.symbol}")
            print(f"   📊 Рекомендация: {self._get_recommendation_emoji(rec.recommendation)} {rec.recommendation}")
            print(f"   🧭 Направление: {self._get_direction_emoji(rec.direction)} {rec.direction}")
            print(f"   ✅ Уверенность: {rec.confidence:.1f}%")
            print(f"   ⏰ Таймфрейм: {rec.timeframe}")
            print(f"   📈 Волатильность: {rec.volatility:.2f}%")
            print(f"   ⚖️  Риск/Вознаграждение: 1:{rec.risk_reward:.1f}")
            print(f"   🕐 Ожидаемая длительность: {rec.expected_duration}")
            print(f"   🏢 Рыночный режим: {rec.market_regime}")
            
            # Уровни поддержки/сопротивления
            if rec.support_levels:
                print(f"   🛡️  Поддержка: {', '.join([f'{s:.8f}' for s in rec.support_levels])}")
            else:
                print(f"   🛡️  Поддержка: Не обнаружено")
            
            if rec.resistance_levels:
                print(f"   🚧 Сопротивление: {', '.join([f'{r:.8f}' for r in rec.resistance_levels])}")
            else:
                print(f"   🚧 Сопротивление: Не обнаружено")
            
            # Параметры сетки
            print(f"   📍 Диапазон входа: {rec.entry_range[0]:.8f} - {rec.entry_range[1]:.8f}")
            print(f"   🔢 Количество уровней: {rec.grid_levels}")
            print(f"   📏 Расстояние сетки: {rec.grid_spacing:.1f}%")
            print(f"   💰 Размер позиции: {rec.position_size:.1f}% на уровень")
            print(f"   🛑 Стоп-лосс: {rec.stop_loss:.8f}")
            
            # Трейлинг
            if rec.trailing_up > 0:
                print(f"   📈 Трейлинг ап: {rec.trailing_up:.1f}%")
            if rec.trailing_down > 0:
                print(f"   📉 Трейлинг даун: {rec.trailing_down:.1f}%")
            
            # Заметки
            if rec.notes:
                print(f"   📝 Заметки: {', '.join(rec.notes[:3])}")  # Показываем только первые 3 заметки
            
            print("-" * 80)
    
    def _get_recommendation_emoji(self, recommendation):
        """Получение эмодзи для рекомендации"""
        emoji_map = {
            'STRONG_BUY': '🚀',
            'BUY': '📈',
            'NEUTRAL': '↔️',
            'SELL': '📉',
            'STRONG_SELL': '🔻'
        }
        return emoji_map.get(recommendation, '📊')
    
    def _get_direction_emoji(self, direction):
        """Получение эмодзи для направления"""
        emoji_map = {
            'LONG': '🟢',
            'SHORT': '🔴',
            'NEUTRAL': '🟡'
        }
        return emoji_map.get(direction, '⚪')
    
    def save_recommendations_to_file(self, filename="grid_recommendations.json"):
        """Сохранение рекомендаций в файл"""
        try:
            recs_data = []
            for rec in self.recommendations:
                rec_dict = {
                    'symbol': rec.symbol,
                    'recommendation': rec.recommendation,
                    'direction': rec.direction,
                    'confidence': rec.confidence,
                    'timeframe': rec.timeframe,
                    'entry_range': rec.entry_range,
                    'take_profit_levels': rec.take_profit_levels,
                    'stop_loss': rec.stop_loss,
                    'support_levels': rec.support_levels,
                    'resistance_levels': rec.resistance_levels,
                    'volatility': rec.volatility,
                    'grid_spacing': rec.grid_spacing,
                    'grid_levels': rec.grid_levels,
                    'position_size': rec.position_size,
                    'expected_duration': rec.expected_duration,
                    'risk_reward': rec.risk_reward,
                    'market_regime': rec.market_regime,
                    'trailing_up': rec.trailing_up,
                    'trailing_down': rec.trailing_down,
                    'notes': rec.notes,
                    'timestamp': datetime.now().isoformat()
                }
                recs_data.append(rec_dict)
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(recs_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Рекомендации сохранены в {filename}")
            
        except Exception as e:
            logger.error(f"Ошибка сохранения рекомендаций: {e}")
    
    async def generate_summary_report(self):
        """Генерация сводного отчета"""
        if not self.recommendations:
            return "Нет данных для отчета"
        
        total_pairs = len(self.recommendations)
        long_count = sum(1 for r in self.recommendations if r.direction == 'LONG')
        short_count = sum(1 for r in self.recommendations if r.direction == 'SHORT')
        neutral_count = sum(1 for r in self.recommendations if r.direction == 'NEUTRAL')
        
        avg_confidence = np.mean([r.confidence for r in self.recommendations])
        avg_volatility = np.mean([r.volatility for r in self.recommendations])
        avg_rr = np.mean([r.risk_reward for r in self.recommendations])
        
        report = f"""
📊 СВОДНЫЙ ОТЧЕТ ПО СЕТОЧНОЙ ТОРГОВЛЕ
{'='*50}
Всего проанализировано пар: {total_pairs}
Рекомендации по направлению:
  Лонг (🟢): {long_count} пар
  Шорт (🔴): {short_count} пар
  Нейтрально (🟡): {neutral_count} пар

Средние показатели:
  Уверенность: {avg_confidence:.1f}%
  Волатильность: {avg_volatility:.2f}%
  Риск/Вознаграждение: 1:{avg_rr:.1f}

Лучшие пары для сетки:
"""
        
        for i, rec in enumerate(self.recommendations[:5], 1):
            report += f"\n{i}. {rec.symbol} ({rec.recommendation} - {rec.confidence:.1f}%)"
            report += f"\n   Направление: {rec.direction} | ТФ: {rec.timeframe}"
            report += f"\n   Диапазон: {rec.entry_range[0]:.8f}-{rec.entry_range[1]:.8f}"
            report += f"\n   Сетка: {rec.grid_levels} уровней, {rec.grid_spacing}% | Длительность: {rec.expected_duration}"
            report += f"\n   Трейлинг: ап {rec.trailing_up:.1f}%, даун {rec.trailing_down:.1f}%"
        
        return report
    
    async def get_detailed_analysis(self, symbol):
        """Получение детального анализа для конкретной пары"""
        recommendation = await self.analyze_pair(symbol)
        
        if not recommendation:
            return f"❌ Не удалось проанализировать пару {symbol}"
        
        analysis = f"""
📈 ДЕТАЛЬНЫЙ АНАЛИЗ: {symbol}
{'='*50}

📊 Основные показатели:
   Текущее направление: {recommendation.direction}
   Рекомендация: {recommendation.recommendation}
   Уверенность: {recommendation.confidence:.1f}%
   Рыночный режим: {recommendation.market_regime}
   Волатильность: {recommendation.volatility:.2f}%

🎯 Параметры сетки:
   Оптимальный таймфрейм: {recommendation.timeframe}
   Количество уровней: {recommendation.grid_levels}
   Расстояние между ордерами: {recommendation.grid_spacing:.1f}%
   Диапазон входа: {recommendation.entry_range[0]:.8f} - {recommendation.entry_range[1]:.8f}
   Размер позиции на уровень: {recommendation.position_size:.1f}%

⚡ Управление рисками:
   Стоп-лосс: {recommendation.stop_loss:.8f}
   Риск/Вознаграждение: 1:{recommendation.risk_reward:.1f}
   Трейлинг ап: {recommendation.trailing_up:.1f}%
   Трейлинг даун: {recommendation.trailing_down:.1f}%

📊 Ключевые уровни:
   Поддержка: {', '.join([f'{s:.8f}' for s in recommendation.support_levels]) if recommendation.support_levels else 'Не обнаружено'}
   Сопротивление: {', '.join([f'{r:.8f}' for r in recommendation.resistance_levels]) if recommendation.resistance_levels else 'Не обнаружено'}

⏰ Временные параметры:
   Ожидаемая длительность: {recommendation.expected_duration}
   Рекомендуемое время удержания: {self._get_holding_time(recommendation)}

📝 Заметки и рекомендации:
   {chr(10).join(['   • ' + note for note in recommendation.notes]) if recommendation.notes else '   Нет заметок'}
"""
        
        return analysis
    
    def _get_holding_time(self, recommendation):
        """Рекомендации по времени удержания позиции"""
        if recommendation.volatility < 1.0:
            return "1-3 дня (долгосрочная сетка)"
        elif recommendation.volatility < 2.0:
            return "12-24 часа (среднесрочная сетка)"
        elif recommendation.volatility < 3.0:
            return "4-8 часов (краткосрочная сетка)"
        else:
            return "2-4 часа (скальпинг сетка)"
    
    async def close(self):
        """Закрытие соединений"""
        await self.exchange.close()

async def main():
    """Основная функция"""
    print("🤖 ЗАПУСК СОВЕТНИКА ПО ТОРГОВЫМ СЕТКАМ")
    print("="*50)
    
    # Инициализация советника
    advisor = GridAdvisor('binance')
    
    try:
        # Получение рекомендаций
        print("\n🔍 Анализ рынка...")
        recommendations = await advisor.get_top_recommendations(limit=10)
        
        if not recommendations:
            print("❌ Не найдено подходящих пар для сеточной торговли")
            return
        
        # Вывод результатов
        advisor.print_recommendations()
        
        # Сохранение в файл
        advisor.save_recommendations_to_file()
        
        # Генерация отчета
        report = await advisor.generate_summary_report()
        print(report)
        
        # Дополнительные опции
        print("\n📊 ДОПОЛНИТЕЛЬНЫЕ ОПЦИИ:")
        print("1. Показать детальный анализ лучшей пары")
        print("2. Экспортировать в CSV")
        print("3. Показать рекомендации по трейлингу")
        print("4. Анализ конкретной пары")
        print("5. Выход")
        
        while True:
            choice = input("\nВыберите опцию (1-5): ").strip()
            
            if choice == "1" and recommendations:
                best = recommendations[0]
                analysis = await advisor.get_detailed_analysis(best.symbol)
                print(analysis)
                
            elif choice == "2":
                # Экспорт в CSV
                import csv
                with open('grid_recommendations.csv', 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Symbol', 'Recommendation', 'Direction', 'Confidence', 'Timeframe', 
                                   'Entry Low', 'Entry High', 'Stop Loss', 'Volatility', 'RR', 
                                   'Grid Levels', 'Grid Spacing', 'Trailing Up', 'Trailing Down', 'Duration'])
                    for rec in recommendations:
                        writer.writerow([rec.symbol, rec.recommendation, rec.direction, rec.confidence, 
                                       rec.timeframe, rec.entry_range[0], rec.entry_range[1], 
                                       rec.stop_loss, rec.volatility, rec.risk_reward,
                                       rec.grid_levels, rec.grid_spacing, 
                                       rec.trailing_up, rec.trailing_down, rec.expected_duration])
                print("✅ Данные экспортированы в grid_recommendations.csv")
                
            elif choice == "3":
                print("\n📈 РЕКОМЕНДАЦИИ ПО ТРЕЙЛИНГУ:")
                print("="*50)
                for rec in recommendations[:3]:
                    print(f"\n{rec.symbol}:")
                    print(f"  Трейлинг ап: {rec.trailing_up:.1f}% - активировать при движении на {rec.trailing_up*1.5:.1f}% от входа")
                    print(f"  Трейлинг даун: {rec.trailing_down:.1f}% - активировать при движении на {rec.trailing_down*1.5:.1f}% от входа")
                    print(f"  Стратегия: {'Следовать за трендом' if rec.direction != 'NEUTRAL' else 'Фиксировать прибыль на каждом уровне'}")
                
            elif choice == "4":
                symbol = input("Введите символ пары (например, BTC/USDT): ").strip()
                if symbol:
                    analysis = await advisor.get_detailed_analysis(symbol)
                    print(analysis)
                    
            elif choice == "5":
                break
                
            else:
                print("❌ Неверный выбор. Попробуйте снова.")
                
    except KeyboardInterrupt:
        print("\n\n👋 Прерывание пользователем")
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        logger.error(f"Ошибка в main: {e}", exc_info=True)
    finally:
        # Закрытие соединений
        await advisor.close()
        print("\n✅ Советник завершил работу")

if __name__ == "__main__":
    # Настройка асинхронного запуска
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 До свидания!")