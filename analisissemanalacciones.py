import streamlit as st
import yfinance as yf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import requests
import math
import re  # NEW: For parsing complex ratio expressions

st.set_page_config(layout="wide")
st.title("Stock and Ratio Weekly/Monthly Variation Heatmap")

# Existing data source functions (descargar_datos_yfinance, etc.) remain unchanged
# ... [Include all the existing data source functions here for completeness] ...

@st.cache_data(ttl=86400)
def fetch_stock_data(ticker, start_date, end_date, source='YFinance'):
    try:
        if source == 'YFinance':
            return descargar_datos_yfinance(ticker, start_date, end_date)
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
def fetch_ratio_data(ratio_expr, start_date, end_date, source='YFinance'):
    try:
        # Parse the ratio expression
        def parse_ratio(expr):
            # Remove extra spaces and handle nested ratios
            expr = expr.strip()
            # Find the main division
            # Handle nested ratios by finding the outermost division
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
                # No division found, treat as a single ticker
                return expr.strip(), None
            
            numerator = expr[:split_idx].strip()
            denominator = expr[split_idx + 1:].strip()
            return numerator, denominator

        def compute_ratio(num_expr, denom_expr, start_date, end_date, source):
            # Compute data for numerator
            if '/' in num_expr and '(' in num_expr:
                num_data = fetch_ratio_data(num_expr, start_date, end_date, source)
            else:
                num_data = fetch_stock_data(num_expr, start_date, end_date, source)
            
            # Compute data for denominator
            if denom_expr is None:
                # Single ticker case
                return num_data
            elif '/' in denom_expr and '(' in denom_expr:
                denom_data = fetch_ratio_data(denom_expr, start_date, end_date, source)
            else:
                denom_data = fetch_stock_data(denom_expr, start_date, end_date, source)

            if num_data.empty or denom_data.empty:
                st.error(f"Cannot compute ratio {ratio_expr}: Data missing for one or both components")
                return pd.DataFrame()

            num_close = num_data['Close'] if 'Close' in num_data.columns else num_data.iloc[:, 0]
            denom_close = denom_data['Close'] if 'Close' in denom_data.columns else denom_data.iloc[:, 0]

            aligned_data = pd.concat([num_close, denom_close], axis=1, keys=['num', 'denom']).dropna()
            ratio_data = pd.DataFrame({
                'Close': aligned_data['num'] / aligned_data['denom']
            }, index=aligned_data.index)

            return ratio_data

        # Handle nested ratios by checking for parentheses
        if '(' in ratio_expr and ')' in ratio_expr:
            # Remove outer parentheses if present
            if ratio_expr.startswith('(') and ratio_expr.endswith(')'):
                ratio_expr = ratio_expr[1:-1]
            numerator, denominator = parse_ratio(ratio_expr)
        else:
            numerator, denominator = parse_ratio(ratio_expr)

        return compute_ratio(numerator, denominator, start_date, end_date, source)

    except Exception as e:
        st.error(f"Error computing ratio {ratio_expr}: {e}")
        return pd.DataFrame()

def calculate_weekly_variation(data):
    if data.empty:
        raise ValueError("No data available for the specified ticker and time range")

    if 'Close' not in data.columns and not isinstance(data.columns, pd.MultiIndex):
        raise ValueError("Data does not contain required 'Close' column")

    if isinstance(data.columns, pd.MultiIndex):
        close_prices = data['Close'].iloc[:, 0]
    else:
        close_prices = data['Close']

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
        stock_data = fetch_ratio_data(ticker_input, start_date, end_date, source)
        display_name = ticker_input

        weekly_variation = calculate_weekly_variation(stock_data)
        comparison_data[display_name] = weekly_variation.loc[f"{year}-01-01":f"{year}-12-31"]

    comparison_data.index = comparison_data.index.strftime('Semana %U')
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
    ax.set_ylabel('Week Number', fontsize=12, family='Arial', weight='bold')

    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(ax.get_xticks())
    ax2.set_xticklabels(data.columns, rotation=45, ha='left')
    ax.set_xticklabels(data.columns, rotation=45, ha='right')
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax2.tick_params(axis='x', which='major', labelsize=10)

    week_numbers = [int(idx.split()[-1]) for idx in data.index]
    min_week = min(week_numbers)
    max_week = max(week_numbers)

    q1_start = datetime(year, 1, 1).isocalendar()[1]
    q2_start = datetime(year, 4, 1).isocalendar()[1]
    q3_start = datetime(year, 7, 1).isocalendar()[1]
    q4_start = datetime(year, 10, 1).isocalendar()[1]

    quarter_starts = {
        'Q1': q1_start,
        'Q2': q2_start,
        'Q3': q3_start,
        'Q4': q4_start
    }
    quarter_positions = []
    quarter_labels = []
    for qtr, start_week in quarter_starts.items():
        if min_week <= start_week <= max_week:
            position = start_week - min_week
            quarter_positions.append(position)
            quarter_labels.append(qtr)

    ax3 = ax.twinx()
    ax3.set_ylim(ax.get_ylim())
    ax3.set_yticks(quarter_positions)
    ax3.set_yticklabels(quarter_labels, fontsize=12, weight='bold', family='Arial')
    ax3.tick_params(length=0)

    quarter_boundaries = [
        (q2_start - min_week - 1) if min_week <= q2_start - 1 <= max_week else None,
        (q3_start - min_week - 1) if min_week <= q3_start - 1 <= max_week else None,
        (q4_start - min_week - 1) if min_week <= q4_start - 1 <= max_week else None
    ]
    for boundary in quarter_boundaries:
        if boundary is not None and boundary >= 0:
            ax.hlines(y=boundary, xmin=0, xmax=data.shape[1],
                      colors='black', linestyles='solid', linewidth=2)

    fig.text(0.5, 0.5, "MTaurus - X: @MTaurus_ok", fontsize=12, color='gray',
             ha='center', va='center', alpha=0.5, weight='bold', family='Arial')

    plt.tight_layout()
    return fig

def calculate_monthly_variation(data):
    if isinstance(data.columns, pd.MultiIndex):
        close_prices = data['Close'].iloc[:, 0]
    else:
        close_prices = data['Close']

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
    data_sources = ['YFinance', 'AnálisisTécnico.com.ar', 'IOL (Invertir Online)', 'ByMA Data']

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
                        st.pyplot(fig, dpi=300)
                except Exception as e:
                    st.error(f"Ocurrió un error: {str(e)}")
                    st.info("Por favor, verifica si el símbolo del ticker es válido y si el rango de fechas es apropiado.")

    else:
        with st.sidebar:
            selected_sources = st.multiselect('Seleccionar Fuentes de Datos', data_sources, default=['YFinance'])
            ticker_inputs = {}
            for source in selected_sources:
                ticker_input = st.text_input(
                    f"Tickers o Ratios para {source} (separados por comas, ej: AAPL, ^MERV/(YPFD.BA/YPF))",
                    value="AAPL" if source == 'YFinance' else "^MERV/(YPFD.BA/YPF)",
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
                        fig = plot_comparison_heatmap(comparison_data, f'Comparación de Variación Semanal para {year}', year)
                        st.pyplot(fig, dpi=300)
                    else:  # Cambios Mensuales
                        monthly_comparison_data = prepare_monthly_comparison_data(ticker_source_pairs, year)
                        fig = plot_monthly_comparison_heatmap(monthly_comparison_data, f'Comparación de Variación Mensual para {year}')
                        st.pyplot(fig, dpi=300)
            except Exception as e:
                st.error(f"Ocurrió un error: {str(e)}")
                st.info("Por favor, verifica si los tickers o ratios son válidos y si el año es apropiado.")

if __name__ == "__main__":
    main()
