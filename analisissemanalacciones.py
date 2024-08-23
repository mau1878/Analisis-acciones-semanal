import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib.colors import LinearSegmentedColormap

# Define custom colormap
def get_custom_cmap():
    colors = ['red', 'white', 'green']
    return LinearSegmentedColormap.from_list('custom_diverging', colors)

# Function to fetch data
def fetch_data(tickers, start_date, end_date):
    data = {}
    for ticker in tickers:
        ticker = ticker.upper()
        st.write(f"Intentando descargar datos para el ticker {ticker}...")
        try:
            df = yf.download(ticker, start=start_date, end=end_date, interval='1wk')  # Fetch weekly data
            if df.empty:
                st.warning(f"No hay datos disponibles para el ticker {ticker} en el rango de fechas seleccionado.")
            else:
                data[ticker] = df
        except Exception as e:
            st.error(f"Error al descargar datos para el ticker {ticker}: {e}")
    return data

# Function to align dates and fill missing values
def align_dates(data):
    if not data:
        return {}
    
    first_ticker_dates = data[list(data.keys())[0]].index
    
    for ticker in data:
        data[ticker] = data[ticker].reindex(first_ticker_dates)
        data[ticker] = data[ticker].ffill()  # Forward fill missing values
    
    return data

# Function to evaluate the ratio
def evaluate_ratio(main_ticker, second_ticker, third_ticker, data, apply_ypfd_ratio=False):
    if not main_ticker:
        st.error("El ticker principal no puede estar vacío.")
        return None
    
    main_ticker = main_ticker.upper()

    # Apply YPFD.BA/YPF ratio if the option is activated
    if apply_ypfd_ratio:
        st.write(f"Aplicando la razón YPFD.BA/YPF al ticker {main_ticker}...")
        if 'YPFD.BA' in data and 'YPF' in data:
            result = data[main_ticker]['Adj Close'] / (data['YPFD.BA']['Adj Close'] / data['YPF']['Adj Close'])
        else:
            st.error("No hay datos disponibles para YPFD.BA o YPF.")
            return None
    else:
        result = data[main_ticker]['Adj Close']

    # Process the additional ratio with optional divisors
    if second_ticker and third_ticker:
        second_ticker = second_ticker.upper()
        third_ticker = third_ticker.upper()

        if second_ticker in data:
            if third_ticker in data:
                result /= (data[second_ticker]['Adj Close'] / data[third_ticker]['Adj Close'])
            else:
                st.error(f"El tercer divisor no está disponible en los datos: {third_ticker}")
                return None
        else:
            st.error(f"El segundo divisor no está disponible en los datos: {second_ticker}")
            return None
    elif second_ticker:
        second_ticker = second_ticker.upper()

        if second_ticker in data:
            result /= data[second_ticker]['Adj Close']
        else:
            st.error(f"El segundo divisor no está disponible en los datos: {second_ticker}")
            return None

    return result

# Streamlit app
st.title("Análisis de Variación Semanal de Precios de Acciones, ETFs e Índices - MTaurus - X: https://x.com/MTaurus_ok")

# User option to apply the YPFD.BA/YPF ratio
apply_ypfd_ratio = st.checkbox("Dividir el ticker principal por dólar CCL de YPF", value=False)

# User inputs
main_ticker = st.text_input("Ingrese el ticker principal (por ejemplo GGAL.BA o METR.BA o AAPL o BMA:")
second_ticker = st.text_input("Ingrese el segundo ticker o ratio divisor (opcional):")
third_ticker = st.text_input("Ingrese el tercer ticker o ratio divisor (opcional):")

start_date = st.date_input("Seleccione la fecha de inicio:", value=pd.to_datetime('2010-01-01'), min_value=pd.to_datetime('2000-01-01'))
end_date = st.date_input("Seleccione la fecha de fin:", value=pd.to_datetime('today'))

# Option to choose between average and median for weekly and yearly graphs
metric_option = st.radio("Seleccione la métrica para los gráficos semanales y anuales:", ("Promedio", "Mediana"))

# Extract tickers from the inputs
tickers = {main_ticker, second_ticker, third_ticker}
tickers = {ticker.upper() for ticker in tickers if ticker}
if apply_ypfd_ratio:
    tickers.update({'YPFD.BA', 'YPF'})

data = fetch_data(tickers, start_date, end_date)

if data:
    data = align_dates(data)
    
    # Evaluate ratio
    ratio_data = evaluate_ratio(main_ticker, second_ticker, third_ticker, data, apply_ypfd_ratio)
    
    if ratio_data is not None:
        # Calculate weekly price variations
       # Ensure the index is in datetime format
        ratio_data.index = pd.to_datetime(ratio_data.index)
        
        # Calculate weekly price variations
        ratio_data['Week_Start'] = ratio_data.index.to_period('W').start_time
        ratio_data['Year'] = ratio_data.index.year
        ratio_data['Week_Number'] = ratio_data.index.isocalendar().week
        
        # Resample weekly data
        weekly_data = ratio_data.resample('W').ffill()
        
        # Calculate percentage change week-over-week
        weekly_data['Cambio Semanal (%)'] = weekly_data['Adjusted Close'].pct_change() * 100
        
        # Plot weekly price variations
        st.write("### Variaciones Semanales de Precios")
        fig = px.line(weekly_data, x=weekly_data.index, y='Cambio Semanal (%)',
                      title=f"Variaciones Semanales de {main_ticker}" + (f" / {second_ticker}" if second_ticker else "") + (f" / {third_ticker}" if third_ticker else ""),
                      labels={'Cambio Semanal (%)': 'Cambio Semanal (%)'})
        fig.update_traces(mode='lines+markers')
        st.plotly_chart(fig)
        
        # Histogram with Gaussian and percentiles
        st.write("### Histograma de Variaciones Semanales con Ajuste de Gauss")
        weekly_changes = weekly_data['Cambio Semanal (%)'].dropna()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(weekly_changes, kde=False, stat="density", color="skyblue", ax=ax, binwidth=1)
        
        # Fit Gaussian distribution
        mu, std = norm.fit(weekly_changes)
        xmin, xmax = ax.get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)
        ax.plot(x, p, 'k', linewidth=2)
        
        # Percentiles with different colors and vertical labels
        percentiles = [5, 25, 50, 75, 95]
        colors = ['red', 'orange', 'green', 'blue', 'purple']
        for i, percentile in enumerate(percentiles):
            perc_value = np.percentile(weekly_changes, percentile)
            ax.axvline(perc_value, color=colors[i], linestyle='--', label=f'{percentile}º Percentil')
            ax.text(perc_value, ax.get_ylim()[1]*0.9, f'{perc_value:.2f}', color=colors[i],
                    rotation=90, verticalalignment='center', horizontalalignment='right')
        
        ax.set_title(f"Histograma de Cambios Semanales con Ajuste de Gauss para {main_ticker}" + (f" / {second_ticker}" if second_ticker else "") + (f" / {third_ticker}" if third_ticker else ""))
        ax.set_xlabel("Cambio Semanal (%)")
        ax.set_ylabel("Densidad")
        ax.legend()
        # Add watermark   
        plt.text(0.5, 0.01, "MTaurus - X: MTaurus_ok", fontsize=12, color='grey', ha='center', va='center', alpha=0.5, transform=ax.transAxes)
        st.pyplot(fig)
        
        # Heatmap of weekly variations
        st.write("### Mapa de Calor de Variaciones Semanales")
        weekly_pivot = weekly_data.pivot_table(values='Cambio Semanal (%)', index='Year', columns='Week_Number', aggfunc='mean')
        
        # Define a custom colormap with greens for positive values and reds for negative values
        cmap = get_custom_cmap()
        
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(weekly_pivot, cmap=cmap, annot=True, fmt=".2f", linewidths=0.5, center=0, ax=ax)
        plt.title(f"Mapa de Calor de Variaciones Semanales para {main_ticker}" + (f" / {second_ticker}" if second_ticker else "") + (f" / {third_ticker}" if third_ticker else ""))
        st.pyplot(fig)
