# === IMPORT & SETUP ===
import os
import time
import threading
import requests
import ccxt
import json
import logging
import traceback
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mplfinance as mpf
from datetime import datetime, timezone, timedelta

# === LOGGING SETUP ===
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s][%(asctime)s] %(message)s',
    handlers=[
        logging.FileHandler("bot_gentrade.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# === KONFIGURASI BOT ===
TOKEN = os.getenv('GENTRADE_BOT_TOKEN') or '7613368831:AAEDKVY9bJimfFgjehAXYXXUB0z6riXtGbw'
BASE_URL = f'https://api.telegram.org/bot{TOKEN}'
CHAT_ID = os.getenv('GENTRADE_BOT_CHATID') or '6709995631'

SPOT_SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT']
FUTURES_SYMBOLS = ['FET/USDT', 'ARB/USDT', 'VIRTUAL/USDT', 'OP/USDT']
TIMEFRAME = '5m'
LIMIT = 100

# Inisialisasi Exchange
spot_exchange = ccxt.binance({'enableRateLimit': True})
futures_exchange = ccxt.binance({
    'enableRateLimit': True,
    'options': {'defaultType': 'future'}
})

# === RISK MANAGEMENT ===
MODAL_FILE = "user_modals.json"

def load_modals():
    if os.path.exists(MODAL_FILE):
        with open(MODAL_FILE, "r") as f:
            return json.load(f)
    return {}

def save_modals(data):
    with open(MODAL_FILE, "w") as f:
        json.dump(data, f)

def set_modal(chat_id, jumlah):
    data = load_modals()
    data[str(chat_id)] = float(jumlah)
    save_modals(data)
    return jumlah

def get_modal(chat_id):
    data = load_modals()
    return float(data.get(str(chat_id), 0))

def hitung_position_size(chat_id, entry, sl, risk_pct=0.8):
    modal = get_modal(chat_id)
    if modal <= 0:
        return None
    risk_amount = modal * (risk_pct / 100)
    jarak_risk = abs(entry - sl)
    if jarak_risk == 0:
        return None
    size = risk_amount / jarak_risk
    return round(size, 4)

# === SIGNAL MANAGEMENT ===
def get_signal_history(pair=None, limit=10):
    if not os.path.exists("sinyal_history.csv"):
        return []
    df = pd.read_csv("sinyal_history.csv")
    if pair:
        df = df[df["pair"].str.lower() == pair.lower()]
    df = df.sort_values(by="time", ascending=False).head(limit)
    return df.to_dict("records")

def log_signal(pair, timeframe, candle_time, signal):
    try:
        filename = "sinyal_history.csv"
        candle_str = candle_time.strftime('%Y-%m-%d %H:%M:%S')
        signal_id = f"{pair.replace('/','_')}_{int(candle_time.timestamp())}"
        
        if os.path.exists(filename):
            df = pd.read_csv(filename)
        else:
            df = pd.DataFrame(columns=["id", "pair", "timeframe", "time", "signal", "result"])
        
        # Cek duplikat
        is_duplicate = ((df["pair"] == pair) & (df["time"] == candle_str)).any()
        if not is_duplicate:
            new_row = {
                "id": signal_id,
                "pair": pair,
                "timeframe": timeframe,
                "time": candle_str,
                "signal": signal,
                "result": ""
            }
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            df.to_csv(filename, index=False)
            logging.info(f"[LOG_SIGNAL] {pair} @ {candle_str} → {signal.upper()}")
        return signal_id
    except Exception as e:
        logging.error(f"[LOG_SIGNAL ERROR] {e}")
        return None

def update_result_in_csv(pair, candle_time, result):
    try:
        df = pd.read_csv("sinyal_history.csv")
        candle_str = candle_time.strftime('%Y-%m-%d %H:%M:%S')
        mask = (df["pair"] == pair) & (df["time"] == candle_str)
        df.loc[mask, "result"] = result
        df.to_csv("sinyal_history.csv", index=False)
        logging.info(f"[AUTO-LOG] {pair} @ {candle_str} → {result.upper()}")
    except Exception as e:
        logging.error(f"[AUTO-LOG ERROR] {e}")

# === TELEGRAM UTILS ===
def send_message(text, buttons=None):
    try:
        data = {'chat_id': CHAT_ID, 'text': text, 'parse_mode': 'Markdown'}
        if buttons:
            data['reply_markup'] = json.dumps(buttons)
        resp = requests.post(f'{BASE_URL}/sendMessage', json=data, timeout=20)
        if resp.status_code != 200:
            logging.warning(f"[SEND_MESSAGE FAIL] {resp.text}")
    except Exception as e:
        logging.error(f"[SEND_MESSAGE ERROR] {e}")

def send_chart(image_path, caption, signal_id=None, pair=None):
    try:
        with open(image_path, 'rb') as photo:
            files = {'photo': photo}
            data = {'chat_id': CHAT_ID, 'caption': caption, 'parse_mode': 'Markdown'}
            if signal_id or pair:
                buttons = {
                    "inline_keyboard": [
                        [{"text": "📊 Detail", "callback_data": f"detail_{signal_id or 0}"}],
                        [{"text": "📜 History", "callback_data": f"history_{pair or ''}"}]
                    ]
                }
                data['reply_markup'] = json.dumps(buttons)
            resp = requests.post(f'{BASE_URL}/sendPhoto', files=files, data=data, timeout=30)
        if resp.status_code != 200:
            logging.warning(f"[SEND_CHART FAIL] {resp.text}")
        os.remove(image_path)
    except Exception as e:
        logging.error(f"[SEND_CHART ERROR] {e}")

# === MARKET DATA & INDICATORS ===
def get_ohlcv(symbol, mode='spot', limit=LIMIT):
    try:
        exchange = futures_exchange if mode == 'futures' else spot_exchange
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        logging.error(f"[OHLCV ERROR] {symbol}: {e}")
        return pd.DataFrame()

def apply_indicators(df):
    if len(df) < 30:
        return df
    
    # EMA & MACD
    df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema12'] - df['ema26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + (avg_gain / avg_loss)))
    
    # Bollinger Bands
    df['bb_mid'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
    
    # Volume
    df['vol_avg'] = df['volume'].rolling(20).mean()
    
    # ATR & ADX (Trend Strength)
    df['tr'] = np.maximum.reduce([
        df['high'] - df['low'],
        abs(df['high'] - df['close'].shift(1)),
        abs(df['low'] - df['close'].shift(1))
    ])
    df['atr'] = df['tr'].rolling(14).mean()
    
    # ADX Calculation
    up_move = df['high'].diff()
    down_move = -df['low'].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_di = 100 * (plus_dm.rolling(14).sum() / df['atr'])
    minus_di = 100 * (minus_dm.rolling(14).sum() / df['atr'])
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    df['adx'] = dx.rolling(14).mean()
    
    return df.dropna()

# === STRATEGY IMPROVED ===
def detect_signal(df):
    if len(df) < 2:
        return None, []
    
    last, prev = df.iloc[-1], df.iloc[-2]
    signal, reasons = None, []
    
    # Trend Filter
    trend_up = last['ema12'] > last['ema26']
    strong_trend = last['adx'] > 25 if 'adx' in last else True
    
    # Pattern Detection
    bullish_engulf = (prev['close'] < prev['open']) and (last['close'] > last['open']) and (last['close'] > prev['open'])
    bearish_engulf = (prev['close'] > prev['open']) and (last['close'] < last['open']) and (last['close'] < prev['open'])
    
    # MACD Crossover
    macd_bull = (prev['macd'] < prev['macd_signal']) and (last['macd'] > last['macd_signal'])
    macd_bear = (prev['macd'] > prev['macd_signal']) and (last['macd'] < last['macd_signal'])
    
    # RSI Conditions
    rsi_oversold = last['rsi'] < 35
    rsi_overbought = last['rsi'] > 65
    
    # Volume Spike
    volume_spike = last['volume'] > 2 * last['vol_avg']
    
    # Signal Logic
    if strong_trend:
        if bullish_engulf and trend_up:
            signal = 'bull'
            reasons.append("Bullish Engulfing + Uptrend")
        elif bearish_engulf and not trend_up:
            signal = 'bear'
            reasons.append("Bearish Engulfing + Downtrend")
            
        if macd_bull and trend_up:
            signal = 'bull'
            reasons.append("MACD Bullish Crossover")
        elif macd_bear and not trend_up:
            signal = 'bear'
            reasons.append("MACD Bearish Crossover")
    
    if rsi_oversold and last['close'] > last['ema26']:
        signal = 'bull'
        reasons.append("RSI Oversold + EMA Support")
    elif rsi_overbought and last['close'] < last['ema26']:
        signal = 'bear'
        reasons.append("RSI Overbought + EMA Resistance")
    
    if volume_spike:
        reasons.append("Volume Spike 2x")
    
    return signal, reasons

# === CHART GENERATION ===
def create_chart(df, symbol, signal):
    try:
        df_plot = df.set_index('timestamp').copy()
        entry = df_plot['close'].iloc[-1]
        atr = df_plot['atr'].iloc[-1]
        
        if signal == 'bull':
            sl = entry - atr * 1.5
            tp1 = entry + atr * 1.5
            tp2 = entry + atr * 3
        else:
            sl = entry + atr * 1.5
            tp1 = entry - atr * 1.5
            tp2 = entry - atr * 3
        
        apds = [
            mpf.make_addplot(df_plot['ema12'], color='cyan'),
            mpf.make_addplot(df_plot['ema26'], color='orange')
        ]
        
        fig, _ = mpf.plot(
            df_plot,
            type='candle',
            style='charles',
            title=f"{symbol} ({signal.upper()})",
            addplot=apds,
            hlines=dict(
                hlines=[entry, tp1, tp2, sl],
                colors=['cyan', 'lime', 'green', 'red'],
                linestyle='--'
            ),
            returnfig=True,
            figsize=(10, 6)
        )
        
        chart_path = f"chart_{symbol.replace('/', '')}.png"
        fig.savefig(chart_path)
        plt.close()
        
        return chart_path, entry, tp1, tp2, sl
    except Exception as e:
        logging.error(f"[CHART ERROR] {e}")
        return None, None, None, None, None

# === TRADE EXECUTION ===
def generate_signals(symbols, mode='spot'):
    for symbol in symbols:
        try:
            df = get_ohlcv(symbol, mode)
            if df.empty:
                continue
                
            df = apply_indicators(df)
            signal, reasons = detect_signal(df)
            
            if not signal:
                continue
                
            candle_time = df['timestamp'].iloc[-1].to_pydatetime()
            signal_id = log_signal(symbol, TIMEFRAME, candle_time, signal)
            
            chart_path, entry, tp1, tp2, sl = create_chart(df, symbol, signal)
            if not chart_path:
                continue
                
            # Prepare message
            winrate = get_winrate(symbol)
            size_hint = hitung_position_size(CHAT_ID, entry, sl)
            
            caption = f"""
📈 *SIGNAL {signal.upper()}*: `{symbol}`
⏰ Time: `{candle_time.strftime('%Y-%m-%d %H:%M')}`
💰 Entry: `{entry:.4f}` | TP1: `{tp1:.4f}` | TP2: `{tp2:.4f}` | SL: `{sl:.4f}`
📊 Winrate: `{winrate}`
📌 Reasons: {", ".join(reasons)}
            """.strip()
            
            if size_hint:
                caption += f"\n💎 Position Size: `{size_hint:.4f} {symbol.split('/')[0]}`"
            
            send_chart(chart_path, caption, signal_id, symbol)
            
        except Exception as e:
            logging.error(f"[SIGNAL ERROR] {symbol}: {e}")

def get_winrate(pair, limit=30):
    """
    Calculate winrate for a given pair from sinyal_history.csv.
    """
    history = get_signal_history(pair, limit)
    if not history:
        return "N/A"
    total = len(history)
    win = sum(1 for h in history if str(h.get("result", "")).lower() == "win")
    return f"{(win/total*100):.1f}% ({win}/{total})"

# === MAIN BOT LOOP ===
def run_bot():
    logging.info("🚀 Starting GENTRADE BOT...")
    last_rekap = datetime.now() - timedelta(days=1)
    
    while True:
        try:
            now = datetime.now()
            
            # Auto rekap setiap 24 jam
            if (now - last_rekap).total_seconds() >= 86400:
                send_message("📊 Auto-Rekap Harian...")
                generate_signals(SPOT_SYMBOLS + FUTURES_SYMBOLS, 'mixed')
                last_rekap = now
            
            time.sleep(60)
            
        except Exception as e:
            logging.error(f"[BOT ERROR] {e}")
            time.sleep(30)

if __name__ == '__main__':
    run_bot()