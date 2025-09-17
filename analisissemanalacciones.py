import streamlit as st
import yfinance as yf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import requests
import math
import re

st.set_page_config(layout="wide")
st.title("Stock and Ratio Weekly/Monthly Variation Heatmap")

# Data source functions
def descargar_datos_yfinance(ticker, start, end):
    try:
        stock_data = yf.download(ticker, start=start, end=end)
        return stock_data
    except Exception as e:
        st.error(f"Error downloading data from yfinance for {ticker}: {e}")
        return pd.DataFrame()

def descargar_datos_analisistecnico(ticker, start_date, end_date):
    try:
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
        elif isinstance(start_date, datetime):
            start_date = start_date.date()

        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
        elif isinstance(end_date, datetime):
            end_date = end_date.date()

        from_timestamp = int(datetime.combine(start_date, datetime.min.time()).timestamp())
        to_timestamp = int(datetime.combine(end_date, datetime.max.time()).timestamp())

        cookies = {
            'ChyrpSession': '0e2b2109d60de6da45154b542afb5768',
            'i18next': 'es',
            'PHPSESSID': '5b8da4e0d96ab5149f4973232931f033',
        }

        headers = {
            'accept': '*/*',
            'content-type': 'text/plain',
            'dnt': '1',
            'referer': 'https://analisistecnico.com.ar/',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
        }

        symbol = ticker.replace('.BA', '')

        params = {
            'symbol': symbol,
            'resolution': 'D',
            'from': str(from_timestamp),
            'to': str(to_timestamp),
        }

        response = requests.get(
            'https://analisistecnico.com.ar/services/datafeed/history',
            params=params,
            cookies=cookies,
            headers=headers,
        )

        if response.status_code == 200:
            data = response.json()
            if not all(key in data for key in ['t', 'c', 'o', 'h', 'l', 'v']):
                st.error(f"Incomplete data received for {ticker}")
                return pd.DataFrame()

            df = pd.DataFrame({
                'Date': pd.to_datetime(data['t'], unit='s'),
                'Close': data['c'],
                'Open': data['o'],
                'High': data['h'],
                'Low': data['l'],
                'Volume': data['v']
            })
            df = df.sort_values('Date').drop_duplicates(subset=['Date'])
            df.set_index('Date', inplace=True)
            return df[['Close']]
        else:
            st.error(f"Error fetching data for {ticker}: Status code {response.status_code}")
            return pd.DataFrame()

    except Exception as e:
        st.error(f"Error downloading data from analisistecnico for {ticker}: {e}")
        return pd.DataFrame()

def descargar_datos_iol(ticker, start_date, end_date):
    try:
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
        elif isinstance(start_date, datetime):
            start_date = start_date.date()

        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
        elif isinstance(end_date, datetime):
            end_date = end_date.date()

        from_timestamp = int(datetime.combine(start_date, datetime.min.time()).timestamp())
        to_timestamp = int(datetime.combine(end_date, datetime.max.time()).timestamp())

        cookies = {
            'intencionApertura': '0',
            '__RequestVerificationToken': 'DTGdEz0miQYq1kY8y4XItWgHI9HrWQwXms6xnwndhugh0_zJxYQvnLiJxNk4b14NmVEmYGhdfSCCh8wuR0ZhVQ-oJzo1',
            'isLogged': '1',
            'uid': '1107644',
        }

        headers = {
            'accept': '*/*',
            'content-type': 'text/plain',
            'referer': 'https://iol.invertironline.com',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
        }

        symbol = ticker.replace('.BA', '')

        params = {
            'symbolName': symbol,
            'exchange': 'BCBA',
            'from': str(from_timestamp),
            'to': str(to_timestamp),
            'resolution': 'D',
        }

        response = requests.get(
            'https://iol.invertironline.com/api/cotizaciones/history',
            params=params,
            cookies=cookies,
            headers=headers,
        )

        if response.status_code == 200:
            data = response.json()
            if data.get('status') != 'ok' or 'bars' not in data:
                st.error(f"Error in API response for {ticker}")
                return pd.DataFrame()

            df = pd.DataFrame(data['bars'])
            df['Date'] = pd.to_datetime(df['time'], unit='s')
            df['Close'] = df['close']
            df = df[['Date', 'Close']]
            df.set_index('Date', inplace=True)
            df = df.sort_index().drop_duplicates()
            return df
        else:
            st.error(f"Error fetching data for {ticker}: Status code {response.status_code}")
            return pd.DataFrame()

    except Exception as e:
        st.error(f"Error downloading data from IOL for {ticker}: {e}")
        return pd.DataFrame()

def descargar_datos_byma(ticker, start_date, end_date):
    try:
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
        elif isinstance(start_date, datetime):
            start_date = start_date.date()

        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
        elif isinstance(end_date, datetime):
            end_date = end_date.date()

        from_timestamp = int(datetime.combine(start_date, datetime.min.time()).timestamp())
        to_timestamp = int(datetime.combine(end_date, datetime.max.time()).timestamp())

        cookies = {
            'JSESSIONID': '5080400C87813D22F6CAF0D3F2D70338',
            '_fbp': 'fb.2.1728347943669.954945632708052302',
        }

        headers = {
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'de-DE,de;q=0.9,es-AR;q=0.8,es;q=0.7,en-DE;q=0.6,en;q=0.5,en-US;q=0.4',
            'Connection': 'keep-alive',
            'DNT': '1',
            'Referer': 'https://open.bymadata.com.ar/',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
        }

        symbol = ticker.replace('.BA', '')
        if not symbol.endswith(' 24HS'):
            symbol = f"{symbol} 24HS"

        params = {
            'symbol': symbol,
            'resolution': 'D',
            'from': str(from_timestamp),
            'to': str(to_timestamp),
        }

        response = requests.get(
            'https://open.bymadata.com.ar/vanoms-be-core/rest/api/bymadata/free/chart/historical-series/history',
            params=params,
            cookies=cookies,
            headers=headers,
            verify=False
        )

        if response.status_code == 200:
            data = response.json()
            if not all(key in data for key in ['t', 'c']):
                st.error(f"Incomplete data received for {ticker}")
                return pd.DataFrame()

            df = pd.DataFrame({
                'Date': pd.to_datetime(data['t'], unit='s'),
                'Close': data['c']
            })
            df = df.sort_values('Date').drop_duplicates(subset=['Date'])
            df.set_index('Date', inplace=True)
            return df
        else:
            st.error(f"Error fetching data for {ticker}: Status code {response.status_code}")
            return pd.DataFrame()

    except Exception as e:
        st.error(f"Error downloading data from ByMA Data for {ticker}: {e}")
        return pd.DataFrame()

def extract_close_prices(data):
    if data.empty:
        return pd.Series(dtype=float)
    
    if isinstance(data.columns, pd.MultiIndex):
        if ('Adj Close', 'Close') in data.columns:
            close_series = data['Adj Close']['Close']
        elif ('Close', 'Close') in data.columns:
            close_series = data['Close']['Close']
        else:
            close_series = data.iloc[:, 0]
    else:
        if 'Adj Close' in data.columns:
            close_series = data['Adj Close']
        elif 'Close' in data.columns:
            close_series = data['Close']
        else:
            close_series = data.iloc[:, 0]
    
    if isinstance(close_series, pd.DataFrame):
        close_series = close_series.squeeze()
    
    return close_series


@st.cache_data(ttl=86400)
def load_bond_data():
    url = "https://raw.githubusercontent.com/mau1878/Analisis-acciones-semanal/main/bond_data.csv"
    try:
        bond_data = pd.read_csv(url)
        bond_data['Fecha'] = pd.to_datetime(bond_data['Fecha'], format='%d/%m/%Y')
        return bond_data
    except Exception as e:
        st.error(f"Error loading bond data: {e}")
        return pd.DataFrame()

bond_data = load_bond_data()
BONDS = bond_data['Ticker'].unique().tolist() if not bond_data.empty else []

def adjust_for_coupons(ticker, historical_data, bond_payments):
    if historical_data.empty or bond_payments.empty:
        return historical_data
    
    ticker_payments = bond_payments[bond_payments['Ticker'] == ticker].sort_values('Fecha')
    
    adjusted_prices = historical_data['Close'].copy()

    for _, payment in ticker_payments.iterrows():
        payment_date = payment['Fecha']
        coupon_amount = payment['Total']
        
        # Apply coupon adjustment on and after the payment date
        adjusted_prices[adjusted_prices.index >= payment_date] += coupon_amount
        
    historical_data['Close'] = adjusted_prices
    return historical_data

@st.cache_data(ttl=86400)
def fetch_stock_data(ticker, start_date, end_date, source='YFinance'):
    try:
        if source == 'Bonds':
            raw_data = descargar_datos_yfinance(ticker, start_date, end_date)
            if raw_data.empty:
                return pd.DataFrame()
            
            adjusted_data = adjust_for_coupons(ticker, raw_data, bond_data)
            close_prices = extract_close_prices(adjusted_data)
            
            if close_prices.empty:
                return pd.DataFrame()
            
            df = pd.DataFrame({'Close': close_prices})
            return df

        elif source == 'YFinance':
            raw_data = descargar_datos_yfinance(ticker, start_date, end_date)
            close_prices = extract_close_prices(raw_data)
            if close_prices.empty:
                return pd.DataFrame()
            df = pd.DataFrame({'Close': close_prices})
            return df
        elif source == 'AnálisisTécnico.com.ar':
            return descargar_datos_analisistecnico(ticker, start_date, end_date)
        elif source == 'IOL (Invertir Online)':
            return descargar_datos_iol(ticker, start_date, end_date)
        elif source == 'ByMA Data':
            return descargar_datos_byma(ticker, start_date, end_date)
        else:
            st.error(f"Unknown data source: {source}")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error downloading data for {ticker} from {source}: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=86400)
def fetch_ratio_data(ratio_expr, start_date, end_date, source='YFinance', _debug=False):
    try:
        if _debug:
            st.info(f"Debug: Processing ratio '{ratio_expr}' from {source}")

        def parse_ratio(expr):
            expr = expr.strip()
            if not '/' in expr:
                return expr, None
            
            if expr.startswith('(') and expr.endswith(')'):
                expr = expr[1:-1]
            
            depth = 0
            split_idx = -1
            for i, char in enumerate(expr):
                if char == '(':
                    depth += 1
                elif char == ')':
                    depth -= 1
                elif char == '/' and depth == 0:
                    split_idx = i
                    break
            
            if split_idx == -1:
                st.error(f"Invalid ratio expression: {ratio_expr} (no valid '/' found)")
                return None, None
            
            numerator = expr[:split_idx].strip()
            denominator = expr[split_idx + 1:].strip()
            return numerator, denominator

        def compute_ratio(num_expr, denom_expr, start_date, end_date, source):
            if denom_expr is None:
                if _debug:
                    st.info(f"Debug: Fetching single ticker '{num_expr}'")
                num_data = fetch_stock_data(num_expr, start_date, end_date, source)
            else:
                if _debug:
                    st.info(f"Debug: Recursing for numerator '{num_expr}'")
                num_data = fetch_ratio_data(num_expr, start_date, end_date, source, _debug)

            if denom_expr is None:
                return num_data
            if '/' in denom_expr:
                if _debug:
                    st.info(f"Debug: Recursing for denominator '{denom_expr}'")
                denom_data = fetch_ratio_data(denom_expr, start_date, end_date, source, _debug)
            else:
                denom_data = fetch_stock_data(denom_expr, start_date, end_date, source)

            if num_data.empty or denom_data.empty:
                st.error(f"Cannot compute ratio {ratio_expr}: Data missing for num='{num_expr}' or denom='{denom_expr}'")
                return pd.DataFrame()

            num_close = extract_close_prices(num_data)
            denom_close = extract_close_prices(denom_data)

            if num_close.empty or denom_close.empty:
                st.error(f"Cannot compute ratio {ratio_expr}: Close prices missing after extraction")
                return pd.DataFrame()

            aligned_data = pd.concat([num_close, denom_close], axis=1, keys=['num', 'denom']).dropna()
            ratio_data = pd.DataFrame({
                'Close': aligned_data['num'] / aligned_data['denom']
            }, index=aligned_data.index)

            if _debug:
                st.info(f"Debug: Ratio computed successfully for {ratio_expr}, shape: {ratio_data.shape}")
            return ratio_data

        numerator, denominator = parse_ratio(ratio_expr)
        if numerator is None:
            return pd.DataFrame()

        return compute_ratio(numerator, denominator, start_date, end_date, source)

    except Exception as e:
        st.error(f"Error computing ratio {ratio_expr}: {e}")
        return pd.DataFrame()

def calculate_weekly_variation(data):
    if data.empty:
        raise ValueError("No data available for the specified ticker and time range")

    close_prices = extract_close_prices(data)

    weekly_data = close_prices.resample('W').last()
    try:
        previous_year_last_day = close_prices.loc[:weekly_data.index[0] - pd.offsets.Week(1)].iloc[-1]
    except IndexError:
        previous_year_last_day = None

    weekly_variation = weekly_data.pct_change()
    if previous_year_last_day is not None:
        weekly_variation.iloc[0] = (weekly_data.iloc[0] - previous_year_last_day) / previous_year_last_day
    else:
        weekly_variation.iloc[0] = 0

    return weekly_variation

def prepare_comparison_data(ticker_source_pairs, year):
    comparison_data = pd.DataFrame()

    for ticker_input, source in ticker_source_pairs:
        ticker_input = ticker_input.strip()
        start_date = f"{year - 1}-12-25"
        end_date = f"{year}-12-31"
        if '/' not in ticker_input:
            stock_data = fetch_stock_data(ticker_input, start_date, end_date, source)
        else:
            stock_data = fetch_ratio_data(ticker_input, start_date, end_date, source)
        display_name = ticker_input

        weekly_variation = calculate_weekly_variation(stock_data)
        comparison_data[display_name] = weekly_variation.loc[f"{year}-01-01":f"{year}-12-31"]

    week_ranges = []
    for date in comparison_data.index:
        week_start = date - timedelta(days=date.weekday())
        week_end = week_start + timedelta(days=6)
        week_start_str = week_start.strftime('%d/%m')
        week_end_str = week_end.strftime('%d/%m')
        week_ranges.append(f"{week_start_str}-{week_end_str}")
    
    comparison_data.index = week_ranges
    return comparison_data

def plot_comparison_heatmap(data, title, year):
    plt.clf()
    fig = plt.figure(figsize=(10, 20), dpi=300)
    ax = plt.gca()
    custom_cmap = sns.diverging_palette(h_neg=10, h_pos=130, s=99, l=55, sep=3, as_cmap=True)
    max_abs_val = max(abs(data.min().min()), abs(data.max().max()))

    base_size = 8
    reference_cells = 50 * 5
    num_cells = data.shape[0] * data.shape[1]
    font_size = base_size * math.sqrt(reference_cells / max(num_cells, 1))
    font_size = max(6, min(12, font_size))

    sns.heatmap(data,
                cmap=custom_cmap,
                center=0,
                vmin=-max_abs_val,
                vmax=max_abs_val,
                annot=True,
                fmt='.1%',
                annot_kws={'size': font_size, 'weight': 'bold', 'family': 'Arial'},
                cbar_kws={'label': 'Weekly Variation', 'shrink': 0.8},
                square=False,
                ax=ax)

    plt.title(title, pad=20, fontsize=16, weight='bold', family='Arial')
    ax.set_xlabel('Ticker/Ratio', fontsize=12, family='Arial', weight='bold')
    ax.set_ylabel('Week Range (DD/MM-DD/MM)', fontsize=12, family='Arial', weight='bold')

    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(ax.get_xticks())
    ax2.set_xticklabels(data.columns, rotation=45, ha='left')
    ax.set_xticklabels(data.columns, rotation=45, ha='right')
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax2.tick_params(axis='x', which='major', labelsize=10)

    week_ranges = data.index
    week_dates = [datetime.strptime(r.split('-')[0] + f'/{year}', '%d/%m/%Y') for r in week_ranges]
    week_dates = [d.replace(year=year) if d.month != 12 else d.replace(year=year-1) for d in week_dates]

    q1_start = datetime(year, 1, 1)
    q2_start = datetime(year, 4, 1)
    q3_start = datetime(year, 7, 1)
    q4_start = datetime(year, 10, 1)

    quarter_starts = {
        'Q1': q1_start,
        'Q2': q2_start,
        'Q3': q3_start,
        'Q4': q4_start
    }
    quarter_positions = []
    quarter_labels = []
    for qtr, q_start in quarter_starts.items():
        min_diff = float('inf')
        closest_idx = 0
        for idx, week_date in enumerate(week_dates):
            diff = abs((week_date - q_start).days)
            if diff < min_diff:
                min_diff = diff
                closest_idx = idx
        quarter_positions.append(closest_idx)
        quarter_labels.append(qtr)

    ax3 = ax.twinx()
    ax3.set_ylim(ax.get_ylim())
    ax3.set_yticks(quarter_positions)
    ax3.set_yticklabels(quarter_labels, fontsize=12, weight='bold', family='Arial')
    ax3.tick_params(length=0)

    quarter_boundaries = []
    for q_start in [q2_start, q3_start, q4_start]:
        min_diff = float('inf')
        closest_idx = None
        for idx, week_date in enumerate(week_dates):
            diff = abs((week_date - q_start).days)
            if diff < min_diff:
                min_diff = diff
                closest_idx = idx
        if closest_idx is not None:
            quarter_boundaries.append(closest_idx)

    for boundary in quarter_boundaries:
        if boundary is not None and boundary >= 0:
            ax.hlines(y=boundary, xmin=0, xmax=data.shape[1],
                      colors='black', linestyles='solid', linewidth=2)

    fig.text(0.5, 0.5, "MTaurus - X: @MTaurus_ok", fontsize=12, color='gray',
             ha='center', va='center', alpha=0.5, weight='bold', family='Arial')

    plt.tight_layout()
    return fig

def calculate_monthly_variation(data):
    close_prices = extract_close_prices(data)

    monthly_data = close_prices.resample('M').last()
    try:
        previous_december = close_prices.loc[:monthly_data.index[0] - pd.offsets.MonthBegin(1)].iloc[-1]
    except IndexError:
        previous_december = None

    monthly_variation = monthly_data.pct_change()
    if previous_december is not None:
        monthly_variation.iloc[0] = (monthly_data.iloc[0] - previous_december) / previous_december
    else:
        monthly_variation.iloc[0] = 0

    return monthly_variation

def prepare_monthly_comparison_data(ticker_source_pairs, year):
    comparison_data = pd.DataFrame()

    for ticker_input, source in ticker_source_pairs:
        ticker_input = ticker_input.strip()
        start_date = f"{year - 1}-12-01"
        end_date = f"{year}-12-31"
        if '/' not in ticker_input:
            stock_data = fetch_stock_data(ticker_input, start_date, end_date, source)
        else:
            stock_data = fetch_ratio_data(ticker_input, start_date, end_date, source)
        display_name = ticker_input

        monthly_variation = calculate_monthly_variation(stock_data)
        comparison_data[display_name] = monthly_variation.loc[f"{year}-01-01":f"{year}-12-31"]

    comparison_data.index = comparison_data.index.strftime('%b')
    return comparison_data

def plot_monthly_comparison_heatmap(data, title):
    plt.clf()
    fig = plt.figure(figsize=(10, 8), dpi=300)
    ax = plt.gca()
    custom_cmap = sns.diverging_palette(h_neg=10, h_pos=130, s=99, l=55, sep=3, as_cmap=True)
    max_abs_val = max(abs(data.min().min()), abs(data.max().max()))

    base_size = 8
    reference_cells = 12 * 5
    num_cells = data.shape[0] * data.shape[1]
    font_size = base_size * math.sqrt(reference_cells / max(num_cells, 1))
    font_size = max(6, min(12, font_size))

    sns.heatmap(data,
                cmap=custom_cmap,
                center=0,
                vmin=-max_abs_val,
                vmax=max_abs_val,
                annot=True,
                fmt='.1%',
                annot_kws={'size': font_size, 'weight': 'bold', 'family': 'Arial'},
                cbar_kws={'label': 'Variación Mensual', 'shrink': 0.8},
                square=False,
                ax=ax)

    plt.title(title, pad=20, fontsize=16, weight='bold', family='Arial')
    ax.set_xlabel('Ticker/Ratio', fontsize=12, family='Arial', weight='bold')
    ax.set_ylabel('Mes', fontsize=12, family='Arial', weight='bold')

    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(ax.get_xticks())
    ax2.set_xticklabels(data.columns, rotation=45, ha='left')
    ax.set_xticklabels(data.columns, rotation=45, ha='right')
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax2.tick_params(axis='x', which='major', labelsize=10)

    fig.text(0.5, 0.5, "MTaurus - X: @MTaurus_ok", fontsize=12, color='gray',
             ha='center', va='center', alpha=0.5, weight='bold', family='Arial')

    plt.tight_layout()
    return fig

def main():
    data_sources = ['YFinance', 'AnálisisTécnico.com.ar', 'IOL (Invertir Online)', 'ByMA Data', 'Bonds']

    mode = st.radio("Selecciona el modo",
                    ["Un Ticker, Múltiples Años",
                     "Múltiples Tickers o Ratios, Un Año (Cambios Semanales)",
                     "Múltiples Tickers o Ratios, Un Año (Cambios Mensuales)"])

    if mode == "Un Ticker, Múltiples Años":
        with st.sidebar:
            selected_source = st.selectbox('Seleccionar Fuente de Datos', data_sources)
            ticker = st.text_input("Introduce el Ticker de la Acción (no se admiten ratios en este modo)", value="AAPL")
            start_date = st.date_input("Fecha de Inicio", value=pd.to_datetime("2017-01-01"))
            end_date = st.date_input("Fecha de Fin", value=pd.to_datetime("2019-12-31"))
            confirm_data = st.button("Confirmar Datos")

        if confirm_data:
            if '/' in ticker:
                st.error("Ratios no están soportados en el modo 'Un Ticker, Múltiples Años'. Introduce un solo ticker.")
            else:
                try:
                    with st.spinner('Obteniendo y procesando datos...'):
                        stock_data = fetch_stock_data(ticker, start_date, end_date, selected_source)
                        weekly_df = calculate_weekly_variation(stock_data).to_frame(name='Variación')
                        weekly_df['Año'] = weekly_df.index.year
                        weekly_df['Semana'] = weekly_df.index.isocalendar().week
                        heatmap_data = weekly_df.pivot(index='Semana', columns='Año', values='Variación')

                        fig = plot_comparison_heatmap(heatmap_data, f'Heatmap de Variación Semanal para {ticker}', start_date.year)
                        st.pyplot(fig)
                except Exception as e:
                    st.error(f"Ocurrió un error: {str(e)}")
                    st.info("Por favor, verifica si el símbolo del ticker es válido y si el rango de fechas es apropiado.")

    else:
        with st.sidebar:
            selected_sources = st.multiselect('Seleccionar Fuentes de Datos', data_sources, default=['YFinance'])
            debug_mode = st.checkbox("Modo Debug (muestra logs detallados)")
            
            ticker_inputs = {}
            for source in selected_sources:
                default_ticker = "^MERV/(YPFD.BA/YPF)" if source != 'YFinance' else "AAPL, MSFT"
                if source == 'YFinance' and '^MERV' in default_ticker:
                    st.warning("Para ^MERV, usa 'ByMA Data' o 'AnálisisTécnico.com.ar'. YFinance no soporta este índice.")
                ticker_input = st.text_input(
                    f"Tickers o Ratios para {source} (separados por comas, ej: AAPL, ^MERV/(YPFD.BA/YPF))",
                    value=default_ticker,
                    key=f"ticker_{source}"
                )
                ticker_inputs[source] = ticker_input

            year = st.number_input("Selecciona el Año", min_value=2000, max_value=2025, value=2020, step=1)
            confirm_data = st.button("Confirmar Datos")

        if confirm_data:
            if not selected_sources:
                st.error("Por favor, selecciona al menos una fuente de datos.")
                return

            ticker_source_pairs = []
            for source in selected_sources:
                if ticker_inputs[source].strip():
                    tickers = [t.strip() for t in ticker_inputs[source].split(",")]
                    for ticker in tickers:
                        if ticker:
                            ticker_source_pairs.append((ticker, source))

            if not ticker_source_pairs:
                st.error("Por favor, introduce al menos un ticker o ratio.")
                return

            try:
                with st.spinner('Obteniendo y procesando datos...'):
                    if mode == "Múltiples Tickers o Ratios, Un Año (Cambios Semanales)":
                        comparison_data = prepare_comparison_data(ticker_source_pairs, year)
                        if comparison_data.empty:
                            st.error("No se pudo generar datos para el heatmap.")
                            return
                        fig = plot_comparison_heatmap(comparison_data, f'Comparación de Variación Semanal para {year}', year)
                        st.pyplot(fig)
                    else:  # Cambios Mensuales
                        monthly_comparison_data = prepare_monthly_comparison_data(ticker_source_pairs, year)
                        if monthly_comparison_data.empty:
                            st.error("No se pudo generar datos para el heatmap.")
                            return
                        fig = plot_monthly_comparison_heatmap(monthly_comparison_data, f'Comparación de Variación Mensual para {year}')
                        st.pyplot(fig)
            except Exception as e:
                st.error(f"Ocurrió un error: {str(e)}")
                st.info("Por favor, verifica si los tickers o ratios son válidos y si el año es apropiado. Prueba con 'ByMA Data' para ^MERV.")

if __name__ == "__main__":
    main()
