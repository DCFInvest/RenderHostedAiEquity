from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Serve static files from frontend folder
app = Flask(__name__, static_folder='frontend', static_url_path='')
CORS(app)

STOCK_CACHE = None
MODEL = None
CACHE_TIME = None

# --- STOCK DATA FUNCTIONS ---
def get_tickers():
    """Pull all NYSE + NASDAQ tickers from yfinance"""
    try:
        tickers = pd.read_csv('https://pkgstore.datahub.io/core/nasdaq-listings/nasdaq-listed_csv/data/7b0b0a3e5a7e3bbf2fdbb9ee7b7d248f/nasdaq-listed_csv.csv')
        nyse = pd.read_csv('https://pkgstore.datahub.io/core/nyse-other-listings/nyse-listed_csv/data/9d8de3a5fdf16c6d78b6b0fbbd20a2ef/nyse-listed_csv.csv')
        tickers = pd.concat([tickers[['Symbol']], nyse[['ACT Symbol']].rename(columns={'ACT Symbol':'Symbol'})])
        tickers = tickers['Symbol'].unique().tolist()
        return tickers
    except Exception as e:
        logger.error(f"Error fetching tickers: {e}")
        return []

def calculate_dcf(ticker, discount_rate=0.10, growth_rate=0.05, years=5):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        cf = stock.cashflow

        if cf.empty or len(cf.columns) == 0:
            return None

        try:
            if 'Operating Cash Flow' in cf.index and 'Capital Expenditure' in cf.index:
                ocf = float(cf.loc['Operating Cash Flow'].iloc[0])
                capex = float(cf.loc['Capital Expenditure'].iloc[0])
                fcf = ocf + capex
            elif 'Free Cash Flow' in cf.index:
                fcf = float(cf.loc['Free Cash Flow'].iloc[0])
            else:
                return None
        except:
            return None

        if pd.isna(fcf) or fcf <= 0:
            return None

        if discount_rate <= growth_rate:
            discount_rate = growth_rate + 0.02

        projected_fcf = [fcf * ((1 + growth_rate)**i) for i in range(1, years+1)]
        discounted_fcf = [f / ((1 + discount_rate)**i) for i, f in enumerate(projected_fcf, 1)]

        terminal_value = projected_fcf[-1] * (1 + growth_rate) / (discount_rate - growth_rate)
        discounted_terminal = terminal_value / ((1 + discount_rate)**years)

        dcf_value = sum(discounted_fcf) + discounted_terminal
        shares = info.get('sharesOutstanding', 0)
        if not shares or shares <= 0:
            return None

        dcf_per_share = dcf_value / shares
        price = info.get('currentPrice', info.get('regularMarketPrice'))
        if not price or price <= 0:
            return None

        return {
            'ticker': ticker,
            'company': info.get('shortName', ticker),
            'sector': info.get('sector', 'Unknown'),
            'price': float(price),
            'dcf': float(dcf_per_share),
            'market_cap': info.get('marketCap', 0)
        }
    except Exception as e:
        logger.error(f"Error calculating DCF for {ticker}: {e}")
        return None

def process_stocks_parallel(tickers, max_workers=20):
    results = []
    total = len(tickers)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ticker = {executor.submit(calculate_dcf, t): t for t in tickers}
        completed = 0
        for future in as_completed(future_to_ticker):
            try:
                result = future.result(timeout=30)
                if result:
                    results.append(result)
                completed += 1
                if completed % 50 == 0:
                    logger.info(f"Processed {completed}/{total} stocks, valid: {len(results)}")
            except:
                completed += 1
    logger.info(f"Completed all stocks: {len(results)}/{total} valid")
    return pd.DataFrame(results)

def train_model(df):
    global MODEL
    sector_stats = df.groupby('sector').agg({'dcf':'mean', 'price':'mean'}).reset_index()
    sector_stats['sector_ratio'] = sector_stats['price']/sector_stats['dcf']
    df = df.merge(sector_stats[['sector','sector_ratio']], on='sector', how='left')
    market_avg = df['dcf'].mean()
    df['price_vs_dcf'] = df['price']/df['dcf']
    df['price_vs_market'] = df['price']/market_avg
    df['upside'] = (df['dcf']-df['price'])/df['price']
    df = df.dropna()
    features = ['price_vs_dcf','sector_ratio','price_vs_market']
    X = df[features]
    y = df['upside']
    MODEL = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
    MODEL.fit(X, y)
    df['ai_score'] = MODEL.predict(X)
    df['recommendation'] = df['ai_score'].apply(lambda s: 
        '🟢 Strong Buy' if s>0.3 else 
        '🟡 Buy' if s>0.15 else 
        '🟠 Hold' if s>-0.1 else '🔴 Sell'
    )
    return df

def initialize_stock_data():
    global STOCK_CACHE, CACHE_TIME
    logger.info("Initializing stock data...")
    tickers = get_tickers()[:200]  # reduce initial load, can increase later
    stock_data = process_stocks_parallel(tickers)
    if len(stock_data)<10:
        logger.error("Not enough valid stocks")
        return False
    stock_data = train_model(stock_data)
    STOCK_CACHE = stock_data
    CACHE_TIME = datetime.now()
    logger.info(f"Initialized {len(stock_data)} stocks")
    return True

# --- API ROUTES ---
@app.route('/api/status')
def status():
    if STOCK_CACHE is not None:
        return jsonify({
            'ready': True,
            'stock_count': len(STOCK_CACHE),
            'cache_age': (datetime.now() - CACHE_TIME).seconds if CACHE_TIME else 0
        })
    return jsonify({'ready': False})

@app.route('/api/query', methods=['POST'])
def query_stocks():
    if STOCK_CACHE is None:
        return jsonify({'error':'Stock data not ready','stocks':[]}), 503
    data = request.json
    query = data.get('query','').lower()
    df = STOCK_CACHE.copy()

    import re
    price_match = re.search(r'under \$?(\d+)', query)
    if price_match:
        df = df[df['price'] <= int(price_match.group(1))]

    # sector filters
    if 'tech' in query:
        df = df[df['sector'].str.contains('Tech', case=False, na=False)]
    if 'health' in query:
        df = df[df['sector'].str.contains('Health', case=False, na=False)]
    if 'financ' in query or 'bank' in query:
        df = df[df['sector'].str.contains('Financ', case=False, na=False)]
    if 'energy' in query:
        df = df[df['sector'].str.contains('Energy', case=False, na=False)]

    # market cap
    if 'large cap' in query:
        df = df[df['market_cap'] >= 200_000_000_000]
    if 'small cap' in query:
        df = df[df['market_cap'] < 10_000_000_000]

    # valuation filters
    if 'undervalued' in query or 'cheap' in query:
        df = df[df['price_vs_dcf'] < 0.85]
    if 'high upside' in query:
        df = df[df['upside'] > 0.2]

    df = df.nlargest(10, 'ai_score')
    return jsonify({'stocks': df.to_dict('records'), 'count': len(df)})

# --- SERVE FRONTEND ---
@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_any(path):
    return send_from_directory(app.static_folder, 'index.html')

# --- STARTUP ---
if __name__ == '__main__':
    from threading import Thread
    Thread(target=initialize_stock_data, daemon=True).start()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
