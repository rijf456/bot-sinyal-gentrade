# === IMPORT & SETUP ===
import os, time, threading, requests, ccxt, json, logging, traceback
import pandas as pd, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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

SPOT_SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
FUTURES_SYMBOLS = ['FET/USDT', 'OP/USDT', 'ARB/USDT', 'VIRTUAL/USDT']
TIMEFRAME = '5m'
LIMIT = 100

spot_exchange = ccxt.binance({'enableRateLimit': True})
futures_exchange = ccxt.binance({'enableRateLimit': True, 'options': {'defaultType': 'future'}})

# === MODAL (RISK MANAGEMENT) ===
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
    data[str(chat_id)] = jumlah
    save_modals(data)
    return jumlah

def get_modal(chat_id):
    data = load_modals()
    return float(data.get(str(chat_id), 0))

def hitung_position_size(chat_id, entry, sl, risk_pct=1):
    modal = get_modal(chat_id)
    if modal == 0:
        return None
    risk = modal * (risk_pct/100)
    jarak = abs(entry-sl)
    if jarak == 0:
        return None
    size = risk / jarak
    return round(size, 3)

# === WINRATE & HISTORY & SIGNAL MANAGEMENT ===
def get_signal_history(pair=None, limit=10):
    if not os.path.exists("sinyal_history.csv"):
        return []
    df = pd.read_csv("sinyal_history.csv")
    if pair:
        df = df[df["pair"].str.lower() == pair.lower()]
    df = df.sort_values(by="time", ascending=False).head(limit)
    return df.to_dict("records")

def get_signal_detail(signal_id):
    if not os.path.exists("sinyal_history.csv"):
        return None
    df = pd.read_csv("sinyal_history.csv")
    if signal_id.isdigit():
        idx = int(signal_id)
        if 0 <= idx < len(df):
            return df.iloc[idx].to_dict()
    if "id" in df.columns and signal_id in df["id"].astype(str).values:
        return df[df["id"].astype(str) == signal_id].iloc[0].to_dict()
    return None

def mark_signal(signal_id, result):
    if not os.path.exists("sinyal_history.csv"):
        return False
    df = pd.read_csv("sinyal_history.csv")
    if signal_id.isdigit():
        idx = int(signal_id)
        if 0 <= idx < len(df):
            df.at[idx, "result"] = result
            df.to_csv("sinyal_history.csv", index=False)
            return True
    if "id" in df.columns and signal_id in df["id"].astype(str).values:
        df.loc[df["id"].astype(str) == signal_id, "result"] = result
        df.to_csv("sinyal_history.csv", index=False)
        return True
    return False

def list_commands():
    return """
*Daftar Command GENTRADE BOT:*

/start - Mulai bot
/help - Lihat semua command
/sinyal - Generate sinyal trading
/rekap - Rekap mingguan
/spot - Sinyal Spot market
/futures - Sinyal Futures market
/history [pair] - Riwayat sinyal per pair
/detail [id] - Detail satu sinyal
/markwin [id] - Tandai sinyal jadi WIN
/markloss [id] - Tandai sinyal jadi LOSS
/setmodal [jumlah] - Atur modal (USDT)
"""

# === TELEGRAM UTILS & INLINE KEYBOARD ===
def send_message(text, buttons=None):
    try:
        data = {'chat_id': CHAT_ID, 'text': text, 'parse_mode': 'Markdown'}
        if buttons: data['reply_markup'] = json.dumps(buttons)
        headers = {'Content-Type': 'application/json'}
        resp = requests.post(f'{BASE_URL}/sendMessage', data=json.dumps(data, ensure_ascii=False), headers=headers, timeout=20)
        logging.info(f"[SEND_MESSAGE] {text}")
        if resp.status_code != 200:
            logging.warning(f"[SEND_MESSAGE FAIL] {resp.status_code}: {resp.text}")
    except Exception as e:
        logging.error(f"[SEND_MESSAGE ERROR] {e}")

def send_chart(image_path, caption, signal_id=None, pair=None):
    try:
        with open(image_path, 'rb') as photo:
            files = {'photo': photo}
            data = {'chat_id': CHAT_ID, 'caption': caption, 'parse_mode': 'Markdown'}
            buttons = {
                "inline_keyboard": [
                    [{"text":"Detail Sinyal","callback_data":f"detail_{signal_id or 0}"}],
                    [{"text":"History Pair","callback_data":f"history_{pair or ''}"}]
                ]
            }
            data['reply_markup'] = json.dumps(buttons)
            response = requests.post(f'{BASE_URL}/sendPhoto', files=files, data=data, timeout=30)
        logging.info(f"[SEND_CHART] {image_path}")
        if response.status_code != 200:
            logging.warning(f"[SEND_CHART FAIL] {response.status_code}: {response.text}")
    except Exception as e:
        logging.error(f"[SEND_CHART ERROR] {e}")

def get_updates(offset=None):
    try:
        params = {'timeout': 30}
        if offset is not None:
            params['offset'] = offset
        response = requests.get(f'{BASE_URL}/getUpdates', params=params, timeout=35)
        return response.json()
    except Exception as e:
        logging.error(f"[GET_UPDATES ERROR] {e}")
        return {}

# === AMBIL DATA OHLCV ===
def get_ohlcv(sym, mode='spot', limit=LIMIT):
    try:
        ex = futures_exchange if mode == 'futures' else spot_exchange
        ohlcv = ex.fetch_ohlcv(sym, timeframe=TIMEFRAME, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        logging.error(f"[GET_OHLCV ERROR] {sym} ({mode}): {e}")
        return pd.DataFrame()

# === INDIKATOR PREMIUM ===
def apply_indicators(df):
    if len(df) < 30:
        return df
    df['ema12'] = df['close'].ewm(span=12).mean()
    df['ema26'] = df['close'].ewm(span=26).mean()
    df['macd'] = df['ema12'] - df['ema26']
    df['macd_sig'] = df['macd'].ewm(span=9).mean()
    delta = df['close'].diff()
    gain = delta.clip(lower=0); loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean(); avg_loss = loss.rolling(14).mean()
    df['rsi'] = 100 - (100/(1+avg_gain/avg_loss))
    df['bb_mid'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_mid'] + 2*df['bb_std']
    df['bb_lower'] = df['bb_mid'] - 2*df['bb_std']
    low_min = df['low'].rolling(14).min(); high_max = df['high'].rolling(14).max()
    df['stoch_k'] = (df['close'] - low_min)/(high_max - low_min)*100
    df['stoch_d'] = df['stoch_k'].rolling(3).mean()
    df['vol_avg'] = df['volume'].rolling(20).mean()
    # PREMIUM: Tambah ATR & ADX
    df['tr'] = np.maximum.reduce([
        df['high'] - df['low'],
        abs(df['high'] - df['close'].shift(1)),
        abs(df['low'] - df['close'].shift(1))
    ])
    df['atr'] = df['tr'].rolling(14).mean()
    up_move = df['high'].diff()
    down_move = df['low'].diff() * -1
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_di = 100 * pd.Series(plus_dm).rolling(14).sum() / df['atr']
    minus_di = 100 * pd.Series(minus_dm).rolling(14).sum() / df['atr']
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    df['adx'] = pd.Series(dx).rolling(14).mean()
    return df.dropna()

# === DETEKSI SINYAL PREMIUM ===
def detect(df):
    last, prev = df.iloc[-1], df.iloc[-2]
    signal, reasons = None, []
    if prev['close'] < prev['open'] and last['close'] > last['open'] and last['close'] > prev['open'] and last['open'] < prev['close']:
        signal = 'bull'; reasons.append('Bull Engulf')
    elif prev['close'] > prev['open'] and last['close'] < last['open'] and last['open'] > prev['close'] and last['close'] < prev['open']:
        signal = 'bear'; reasons.append('Bear Engulf')
    if prev['macd'] < prev['macd_sig'] and last['macd'] > last['macd_sig']:
        signal = 'bull'; reasons.append('MACD Cross')
    elif prev['macd'] > prev['macd_sig'] and last['macd'] < last['macd_sig']:
        signal = 'bear'; reasons.append('MACD Cross')
    if last['rsi'] < 30:
        signal = 'bull'; reasons.append('RSI <30')
    elif last['rsi'] > 70:
        signal = 'bear'; reasons.append('RSI >70')
    if last['close'] > last['bb_upper']:
        signal = 'bear'; reasons.append('BB Upper Break')
    elif last['close'] < last['bb_lower']:
        signal = 'bull'; reasons.append('BB Lower Break')
    if prev['stoch_k'] < prev['stoch_d'] and last['stoch_k'] > last['stoch_d']:
        signal = 'bull'; reasons.append('Stoch Cross')
    elif prev['stoch_k'] > prev['stoch_d'] and last['stoch_k'] < last['stoch_d']:
        signal = 'bear'; reasons.append('Stoch Cross')
    if last['volume'] > 1.5 * last['vol_avg']:
        reasons.append('Vol Spike')
    if 'adx' in last and last['adx'] < 20:
        reasons.append('Weak Trend (ADX<20)')
        signal = None
    # MA breakout
if last['close'] > last['ema12'] and last['close'] > last['ema26']:
    sig = 'bull'; reasons.append('MA Breakout')
elif last['close'] < last['ema12'] and last['close'] < last['ema26']:
    sig = 'bear'; reasons.append('MA Breakdown')

# Volume spike
if last['volume'] > 2 * last['vol_avg']:
    reasons.append('Volume Spike')  
    return signal, reasons

# === CHART PREMIUM ===
import mplfinance as mpf

def create_chart(df, symbol, signal, signal_id=None):
    try:
        df_plot = df.copy()
        df_plot.set_index('timestamp', inplace=True)
        entry = df_plot['close'].iloc[-1]
        atr = df_plot['atr'].iloc[-1] if 'atr' in df_plot.columns else entry * 0.005
        if signal == 'bear':
            tp1 = entry - atr * 1.5; tp2 = entry - atr * 3; sl = entry + atr * 1.5
        else:
            tp1 = entry + atr * 1.5; tp2 = entry + atr * 3; sl = entry - atr * 1.5
        apds = [
            mpf.make_addplot(df_plot['ema12'], color='cyan'),
            mpf.make_addplot(df_plot['ema26'], color='orange')
        ]
        hlines = [entry, tp1, tp2, sl]
        colors = ['cyan' if signal != 'bear' else 'orange', 'lime', 'green', 'red']
        labels = ['Entry (Long)' if signal != 'bear' else 'Entry (Short)', 'TP1', 'TP2', 'SL']
        fig, axlist = mpf.plot(
            df_plot,
            type='candle',
            style='charles',
            title=f'{symbol} - {TIMEFRAME} - {"SHORT" if signal == "bear" else "LONG"}',
            ylabel='Harga',
            ylabel_lower='Volume',
            volume=True,
            addplot=apds,
            hlines=dict(hlines=hlines, colors=colors, linewidths=1, linestyle='--'),
            returnfig=True,
            figsize=(11, 6),
            tight_layout=True
        )
        ax = axlist[0]
        for y, label, color in zip(hlines, labels, colors):
            ax.text(df_plot.index[-1], y, f' {label}: {y:.4f}', color=color, fontsize=8,
                    va='center', ha='left', bbox=dict(facecolor='black', edgecolor='none', pad=1.0))
        filename = f"chart_{symbol.replace('/', '')}_{signal}.png"
        path = os.path.join(os.getcwd(), filename)
        fig.savefig(path)
        plt.close(fig)
        return path, entry, tp1, tp2, sl
    except Exception as e:
        logging.error(f"[CHART ERROR] {symbol}: {e}\n{traceback.format_exc()}")
        return None, None, None, None, None

# === WINRATE ===
def get_winrate(pair):
    try:
        df = pd.read_csv("sinyal_history.csv")
        pair_data = df[df["pair"] == pair]
        total = len(pair_data)
        wins = len(pair_data[pair_data["result"] == "win"])
        return round((wins / total) * 100, 2) if total > 0 else "N/A"
    except Exception as e:
        logging.error(f"[WINRATE ERROR]: {e}")
        return "N/A"

def log_signal(pair, timeframe, candle_time, signal):
    try:
        filename = "sinyal_history.csv"
        candle_str = candle_time.strftime('%Y-%m-%d %H:%M:%S')
        # Tambah id unik (timestamp + pair)
        id_sinyal = f"{pair.replace('/','_')}_{int(candle_time.timestamp())}"
        if os.path.exists(filename):
            df = pd.read_csv(filename)
        else:
            df = pd.DataFrame(columns=["id", "pair", "timeframe", "time", "signal", "result"])
        is_duplicate = ((df["pair"] == pair) & (df["time"] == candle_str)).any()
        if not is_duplicate:
            new_row = {
                "id": id_sinyal,
                "pair": pair,
                "timeframe": timeframe,
                "time": candle_str,
                "signal": signal,
                "result": ""
            }
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            df.to_csv(filename, index=False)
            logging.info(f"[LOG_SIGNAL] {pair} @ {candle_str} → {signal.upper()}")
        return id_sinyal
    except Exception as e:
        logging.error(f"[LOG_SIGNAL ERROR] {e}")
        return None

def update_result_in_csv(pair, candle_time, result):
    try:
        df = pd.read_csv("sinyal_history.csv", parse_dates=["time"])
        candle_str = candle_time.strftime('%Y-%m-%d %H:%M:%S')
        mask = (df["pair"] == pair) & (df["time"] == candle_str)
        df.loc[mask, "result"] = result
        df.to_csv("sinyal_history.csv", index=False)
        logging.info(f"[AUTO-LOG] {pair} @ {candle_str} → {result.upper()}")
    except Exception as e:
        logging.error(f"[AUTO-LOG ERROR] {e}")

def check_and_log_result(sym, candle_time, entry, tp, sl, mode):
    try:
        df_fut = get_ohlcv(sym, mode=mode, limit=LIMIT + 3).tail(3)
        high = df_fut["high"].max()
        low = df_fut["low"].min()
        result = "win" if (high >= tp if tp > entry else low <= tp) else "loss"
        update_result_in_csv(sym, candle_time, result)
    except Exception as e:
        logging.error(f"[CHECK LOG ERROR] {sym}: {e}")

# === GENERATE SIGNAL PREMIUM + ID ===
def generate_signals(symbols, mode):
    for sym in symbols:
        try:
            logging.info(f"[DATA] Mengambil {sym} ({mode})")
            exch_mode = 'futures' if sym in FUTURES_SYMBOLS else 'spot' if mode == 'mixed' else mode
            df = get_ohlcv(sym, mode=exch_mode)
            if df.empty or len(df) < 30:
                logging.warning(f"[SKIP] Data tidak cukup untuk {sym}")
                continue
            df = apply_indicators(df)
            if len(df) < 2:
                logging.warning(f"[SKIP] Data indikator tidak cukup untuk {sym}")
                continue
            signal, reasons = detect(df)
            if not signal:
                logging.info(f"[SKIP] Tidak ada sinyal untuk {sym}")
                continue
            candle_time = df['timestamp'].iloc[-1].to_pydatetime().replace(tzinfo=timezone.utc)
            signal_id = log_signal(sym, TIMEFRAME, candle_time, signal)
            now = datetime.now(timezone.utc)
            delay_minutes = round((now - candle_time).total_seconds() / 60)
            chart_path, entry, tp1, tp2, sl = create_chart(df, sym, signal, signal_id=signal_id)
            strategy_note = " | ".join(reasons) if reasons else "Manual Entry — tunggu konfirmasi candle"
            winrate = get_winrate(sym)
            posisi = "📈 *LONG*" if signal == 'bull' else "📉 *SHORT*"
            # Hitung size otomatis jika modal tersedia
            size_hint = hitung_position_size(CHAT_ID, entry, sl)
            size_str = f"\n💰 *Rek. Size*: `{size_hint} {sym.split('/')[0]}`" if size_hint else ""
            caption = f"""
📊 *Sinyal GENTRADE AI BOT* [PREMIUM]
Mode: `{exch_mode.capitalize()}`  |  Pair: `{sym}`
⌚ `{candle_time.strftime('%Y-%m-%d %H:%M')}`  |  ⏱ Delay: `{delay_minutes} menit`
ID: `{signal_id}`
{posisi}
• Entry : `{entry:.4f}`
• TP1   : `{tp1:.4f}`
• TP2   : `{tp2:.4f}`
• SL    : `{sl:.4f}`
🧠 *Strategi*: {strategy_note}
📈 *Winrate Historis*: `{winrate}%`
{size_str}
""".strip()
            send_chart(chart_path, caption, signal_id=signal_id, pair=sym)
            check_and_log_result(sym, candle_time, entry, tp1, sl, exch_mode)
            os.remove(chart_path)
            time.sleep(1)
        except Exception as e:
            logging.error(f"[ERROR] {sym}: {e}\n{traceback.format_exc()}")

# === REKAP MINGGUAN PREMIUM ===
def rekap_mingguan():
    try:
        df = pd.read_csv("sinyal_history.csv", parse_dates=["time"])
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        weekly = df[(df["time"] >= start_date) & (df["time"] <= end_date)]
        total = len(weekly)
        wins = len(weekly[weekly["result"] == "win"])
        losses = len(weekly[weekly["result"] == "loss"])
        winrate = round((wins / total) * 100, 2) if total > 0 else 0
        best_pair = "-"
        if wins > 0:
            try:
                best_pair = weekly[weekly["result"] == "win"]['pair'].value_counts().idxmax()
            except:
                pass
        pesan = f"""
📊 *Rekap Sinyal Mingguan* [PREMIUM]
🗓 Periode: {start_date.strftime('%d %b')} – {end_date.strftime('%d %b %Y')}
• Total Sinyal : {total}
• WIN          : {wins}
• LOSS         : {losses}
• Winrate      : {winrate}%
• Pair Terbaik : {best_pair}
""".strip()
        send_message(pesan)
    except Exception as e:
        send_message(f"[ERROR] Rekap gagal: {e}")
        
def buat_grafik_rekap():
    import pandas as pd
    import matplotlib.pyplot as plt
    from datetime import datetime, timedelta
    import os

    df = pd.read_csv("sinyal_history.csv", parse_dates=["time"])
    akhir = datetime.now()
    awal = akhir - timedelta(days=7)
    df = df[(df["time"] >= awal) & (df["time"] <= akhir)]

    hasil = df["result"].value_counts()

    fig, ax = plt.subplots(figsize=(6, 4))
    hasil.plot(kind='bar', color=['green', 'red'], ax=ax)
    ax.set_title("Rekap Win/Loss Mingguan")
    ax.set_ylabel("Jumlah Sinyal")
    ax.grid(alpha=0.3)

    filename = "rekap_weekly_chart.png"
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    return filename

# === HANDLER COMMAND PREMIUM ===
def handle_command(text, chat_id=CHAT_ID):
    try:
        text = text.strip().lower()
        if text == '/start':
            welcome_msg = """
🤖 *Selamat datang di GENTRADE AI BOT* [PREMIUM] 🤖

Fitur Premium:
/sinyal   ➜ Sinyal trading otomatis (semua market)
/rekap    ➜ Rekap mingguan hasil sinyal
/spot     ➜ Sinyal khusus Spot Market
/futures  ➜ Sinyal khusus Futures Market
/history [pair] - Riwayat sinyal per pair
/detail [id] - Detail satu sinyal
/markwin [id] - Tandai sinyal jadi WIN
/markloss [id] - Tandai sinyal jadi LOSS
/setmodal [jumlah] - Atur modal untuk rekomendasi size
/help - Daftar perintah
"""
            send_message(welcome_msg)
            return True
        elif text == '/help':
            send_message(list_commands())
            return True
        elif text == '/sinyal':
            send_message("🔎 Mencari sinyal untuk semua market (Spot & Futures)...")
            threading.Thread(target=generate_signals, args=(SPOT_SYMBOLS + FUTURES_SYMBOLS, 'mixed')).start()
            return True
        elif text == '/spot':
            send_message("🔎 Mencari sinyal Spot Market...")
            threading.Thread(target=generate_signals, args=(SPOT_SYMBOLS, 'spot')).start()
            return True
        elif text == '/futures':
            send_message("🔎 Mencari sinyal Futures Market...")
            threading.Thread(target=generate_signals, args=(FUTURES_SYMBOLS, 'futures')).start()
            return True
        elif text == '/rekap':
            send_message("📊 Menghitung rekap mingguan...")
            threading.Thread(target=rekap_mingguan).start()
            img_path = buat_grafik_rekap()
            send_chart(img_path, "📈 Grafik Rekap Mingguan")
            os.remove(img_path)
            return True
        elif text.startswith('/setmodal'):
            try:
                jumlah = float(text.split(maxsplit=1)[1])
                set_modal(chat_id, jumlah)
                send_message(f"✅ Modal kamu di-set ke: *{jumlah:,.2f} USDT*")
            except:
                send_message("Format: `/setmodal 1000` (tanpa tanda kutip)")
            return True
        elif text.startswith('/history'):
            pair = text.split(maxsplit=1)[1] if len(text.split()) > 1 else None
            history = get_signal_history(pair)
            if not history:
                send_message("❌ Tidak ada data riwayat sinyal.")
            else:
                msg = "\n".join([f"ID:{d['id']} {d['pair']} {d['time']} {d['signal']} {d['result']}" for d in history])
                send_message(f"*Riwayat Sinyal:*\n{msg}")
            return True
        elif text.startswith('/detail'):
            args = text.split(maxsplit=1)
            if len(args) < 2:
                send_message("Format: `/detail [id]`")
            else:
                detail = get_signal_detail(args[1])
                if not detail:
                    send_message("❌ Sinyal tidak ditemukan.")
                else:
                    msg = "\n".join([f"*{k}*: {v}" for k,v in detail.items()])
                    send_message(f"*Detail Sinyal:*\n{msg}")
            return True
        elif text.startswith('/markwin') or text.startswith('/markloss'):
            result = 'win' if text.startswith('/markwin') else 'loss'
            args = text.split(maxsplit=1)
            if len(args) < 2:
                send_message(f"Format: `/{'markwin' if result=='win' else 'markloss'} [id]`")
            else:
                success = mark_signal(args[1], result)
                if success:
                    send_message(f"✅ Sinyal ID {args[1]} ditandai sebagai *{result.upper()}*.")
                else:
                    send_message("❌ Gagal update status sinyal.")
            return True
        else:
            send_message("❌ Command tidak dikenali. Gunakan /help untuk melihat menu.")
            return False
    except Exception as e:
        logging.error(f"[HANDLE_COMMAND ERROR] {e}")
        send_message(f"⚠️ Terjadi error: {str(e)}")
        return False

# === CALLBACK HANDLER (untuk tombol inline) ===
def handle_callback_query(callback_data):
    if callback_data.startswith("detail_"):
        signal_id = callback_data.replace("detail_", "")
        detail = get_signal_detail(signal_id)
        if not detail:
            send_message("❌ Sinyal tidak ditemukan.")
        else:
            msg = "\n".join([f"*{k}*: {v}" for k,v in detail.items()])
            send_message(f"*Detail Sinyal:*\n{msg}")
    elif callback_data.startswith("history_"):
        pair = callback_data.replace("history_", "")
        history = get_signal_history(pair)
        if not history:
            send_message("❌ Tidak ada data history.")
        else:
            msg = "\n".join([f"ID:{d['id']} {d['pair']} {d['time']} {d['signal']} {d['result']}" for d in history])
            send_message(f"*Riwayat Sinyal:*\n{msg}")
    else:
        send_message("❌ Callback tidak dikenali.")

# === BOT LOOP PREMIUM ===
def run():
    offset = None
    logging.info("[BOT] Start listening...")
    try:
        test = requests.get(f'{BASE_URL}/getMe', timeout=10)
        if test.status_code != 200:
            logging.error(f"[ERROR] Koneksi gagal. Pastikan TOKEN valid. Status: {test.status_code}")
            return
    except Exception as e:
        logging.error(f"[CONNECTION ERROR] Gagal terhubung ke Telegram: {e}")
        return
    while True:
        try:
            updates = get_updates(offset)
            if not updates or 'result' not in updates:
                logging.debug("[DEBUG] Tidak ada update atau format response tidak valid")
                time.sleep(5)
                continue
            for update in updates.get('result', []):
                offset = update['update_id'] + 1
                if 'message' in update:
                    msg = update['message']
                    chat_id = str(msg.get('chat', {}).get('id', ''))
                    text = msg.get('text', '').strip()
                    logging.info(f"[DEBUG] Received: ChatID={chat_id}, Text='{text}'")
                    if chat_id != CHAT_ID:
                        logging.info(f"[DEBUG] Ignoring message from unauthorized chat: {chat_id}")
                        continue
                    if not text.startswith('/'):
                        logging.info(f"[DEBUG] Ignoring non-command text: {text}")
                        continue
                    handle_command(text, chat_id=chat_id)
                elif 'callback_query' in update:
                    callback_query = update['callback_query']
                    callback_data = callback_query['data']
                    handle_callback_query(callback_data)
        except Exception as e:
            logging.error(f"[MAIN LOOP ERROR] {type(e).__name__}: {e}\n{traceback.format_exc()}")
            time.sleep(10)
        time.sleep(1)

# === START ===
if __name__ == '__main__':
    run()
