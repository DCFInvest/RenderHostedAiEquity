from flask import Flask, jsonify, request
from flask_cors import CORS
import yfinance as yf
import pandas as pd
from xgboost import XGBRegressor
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

STOCK_CACHE = None
MODEL = None
CACHE_TIME = None

# ----------------- Helper functions -----------------

def get_all_tickers():
    """
    Pull all tickers on NASDAQ and NYSE from yfinance.
    Returns a DataFrame with columns: ticker, company, sector.
    """
    try:
        nasdaq = pd.read_csv('https://raw.githubusercontent.com/datasets/nasdaq-listings/master/data/nasdaq-listed.csv')
        nyse = pd.read_csv('https://raw.githubusercontent.com/datasets/nyse-listings/master/data/nyse-listed.csv')

        tickers = pd.concat([nasdaq[['Symbol', 'Security Name']], nyse[['ACT Symbol', 'Company Name']]], ignore_index=True)
        tickers.columns = ['ticker', 'company']
        tickers['sector'] = 'Unknown'  # sector info not available here
        tickers = tickers.drop_duplicates(subset=['ticker'])
        return tickers
    except Exception as e:
        logger.error(f"Error fetching tickers: {e}")
        return pd.DataFrame(columns=['ticker', 'company', 'sector'])

def calculate_dcf(ticker, discount_rate=0.10, growth_rate=0.05, years=5):
    """
    Calculate DCF for a ticker. Returns dict or None if invalid.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        cf = stock.cashflow

        if cf.empty:
            return None

        try:
            if 'Operating Cash Flow' in cf.index and 'Capital Expenditure' in cf.index:
                fcf = float(cf.loc['Operating Cash Flow'].iloc[0] + cf.loc['Capital Expenditure'].iloc[0])
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

        shares = info.get('sharesOutstanding')
        if not shares or shares <= 0:
            return None

        dcf_per_share = dcf_value / shares
        price = info.get('currentPrice') or info.get('regularMarketPrice')
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
        logger.debug(f"Ticker {ticker} invalid: {e}")
        return None

def process_stocks_parallel(tickers, max_workers=20):
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(calculate_dcf, t): t for t in tickers}
        for future in as_completed(futures):
            try:
                res = future.result(timeout=30)
                if res:
                    results.append(res)
            except:
                continue
    logger.info(f"Processed {len(results)} valid tickers out of {len(tickers)}")
    return pd.DataFrame(results)

def train_model(df):
    global MODEL
    if df.empty:
        return df
    sector_stats = df.groupby('sector').agg({'dcf':'mean','price':'mean'}).reset_index()
    sector_stats['sector_ratio'] = sector_stats['price']/sector_stats['dcf']
    df = df.merge(sector_stats[['sector','sector_ratio']], on='sector', how='left')
    market_avg = df['dcf'].mean()
    df['price_vs_dcf'] = df['price']/df['dcf']
    df['price_vs_market'] = df['price']/market_avg
    df['upside'] = (df['dcf'] - df['price'])/df['price']

    features = ['price_vs_dcf','sector_ratio','price_vs_market']
    X = df[features]
    y = df['upside']

    MODEL = XGBRegressor(n_estimators=100,max_depth=4,learning_rate=0.1,random_state=42)
    MODEL.fit(X,y)

    df['ai_score'] = MODEL.predict(X)
    df['recommendation'] = df['ai_score'].apply(lambda s:'🟢 Strong Buy' if s>0.3 else '🟡 Buy' if s>0.15 else '🟠 Hold' if s>-0.1 else '🔴 Sell')
    return df

def initialize_stock_data():
    global STOCK_CACHE, CACHE_TIME
    logger.info("Fetching all NYSE and NASDAQ tickers...")
    tickers_df = get_all_tickers()
    tickers = tickers_df['ticker'].tolist()
    logger.info(f"Total tickers fetched: {len(tickers)}")

    df = process_stocks_parallel(tickers, max_workers=25)
    if df.empty:
        logger.error("No valid tickers found")
        return False

    df = train_model(df)
    STOCK_CACHE = df
    CACHE_TIME = datetime.now()
    logger.info(f"Stock data initialized: {len(df)} valid tickers")
    return True

# ----------------- API Endpoints -----------------

@app.route('/api/status')
def status():
    return jsonify({'ready': STOCK_CACHE is not None, 'stock_count': len(STOCK_CACHE) if STOCK_CACHE is not None else 0})

@app.route('/api/query', methods=['POST'])
def query_stocks():
    if STOCK_CACHE is None:
        return jsonify({'error':'Stock data not ready','stocks':[]}),503

    query = request.json.get('query','').lower()
    df = STOCK_CACHE.copy()

    import re
    price_match = re.search(r'under \$?(\d+)', query)
    if price_match:
        df = df[df['price'] <= int(price_match.group(1))]

    if 'tech' in query:
        df = df[df['sector'].str.contains('Tech', case=False, na=False)]
    if 'health' in query:
        df = df[df['sector'].str.contains('Health', case=False, na=False)]
    if 'financ' in query or 'bank' in query:
        df = df[df['sector'].str.contains('Financ', case=False, na=False)]
    if 'energy' in query:
        df = df[df['sector'].str.contains('Energy', case=False, na=False)]

    if 'large cap' in query:
        df = df[df['market_cap']>=200_000_000_000]
    if 'small cap' in query:
        df = df[df['market_cap']<10_000_000_000]

    if 'undervalued' in query or 'cheap' in query:
        df = df[df['price_vs_dcf']<0.85]
    if 'high upside' in query:
        df = df[df['upside']>0.2]

    df = df.nlargest(10,'ai_score')
    return jsonify({'stocks': df.to_dict('records'),'count':len(df)})

# ----------------- Initialize -----------------

if __name__ == '__main__':
    from threading import Thread
    Thread(target=initialize_stock_data, daemon=True).start()
    port = int(os.environ.get('PORT',5000))
    app.run(host='0.0.0.0', port=port, debug=False)
