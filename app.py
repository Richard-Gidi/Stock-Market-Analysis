import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import xgboost as xgb
from textblob import TextBlob
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta

# Download required NLTK data
nltk.download('vader_lexicon')

# Configure Streamlit app
st.set_page_config(page_title="Stock Market Analysis Dashboard", layout="wide")

# Title and description
st.title("Stock Market Analysis Dashboard")
st.write("Analyze stock performance and market sentiment")

# Function to load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("SPY_PG_JNJ STOCKS DATA.csv")
        df['Date'] = pd.to_datetime(df['Date'])
        # Extract stock name from StockNumber (e.g., 'SPY3522' -> 'SPY')
        df['Stock'] = df['StockNumber'].str.extract(r'([A-Za-z]+)')
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Function to load model
@st.cache_data
def load_model():
    try:
        model = joblib.load("Models/xgboost.pkl")
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Function to prepare prediction data
def prepare_prediction_data(df, stock_name):
    stock_data = df[df['Stock'] == stock_name].copy()
    
    # Calculate returns and volatility
    stock_data['Daily Return'] = stock_data['Close'].pct_change()
    stock_data['Volatility'] = stock_data['Daily Return'].rolling(window=20).std()
    
    return stock_data

# Function to get market sentiment
def get_market_sentiment(stock_name):
    try:
        # Get news articles
        url = f"https://news.google.com/rss/search?q={stock_name}+stock&hl=en-US&gl=US&ceid=US:en"
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'xml')
        items = soup.find_all('item')
        
        # Analyze sentiment
        sia = SentimentIntensityAnalyzer()
        sentiments = []
        
        for item in items[:5]:  # Analyze top 5 articles
            title = item.title.text
            sentiment = sia.polarity_scores(title)
            sentiments.append(sentiment['compound'])
        
        avg_sentiment = np.mean(sentiments)
        return avg_sentiment
    except Exception as e:
        st.warning(f"Could not fetch market sentiment: {str(e)}")
        return 0

# Function to calculate Value at Risk
def calculate_var(returns, confidence_level=0.95):
    return np.percentile(returns, (1 - confidence_level) * 100)

# Load data
df = load_data()
if df is None:
    st.stop()

# Sidebar
st.sidebar.header("Settings")
stock_names = df['Stock'].unique()
selected_stock = st.sidebar.selectbox("Select Stock", stock_names)

# Date range selection
min_date = df['Date'].min()
max_date = df['Date'].max()
start_date = st.sidebar.date_input("Start Date", min_date)
end_date = st.sidebar.date_input("End Date", max_date)

# Filter data based on selection
filtered_data = df[
    (df['Stock'] == selected_stock) &
    (df['Date'] >= pd.to_datetime(start_date)) &
    (df['Date'] <= pd.to_datetime(end_date))
]

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["Historical Data", "Market Sentiment", "Risk Analysis", "Price Prediction"])

# Tab 1: Historical Data
with tab1:
    st.subheader("Historical Price Data")
    st.dataframe(filtered_data)
    
    # Create candlestick chart
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                       vertical_spacing=0.03, 
                       row_heights=[0.7, 0.3])
    
    # Candlestick
    fig.add_trace(go.Candlestick(x=filtered_data['Date'],
                                open=filtered_data['Open'],
                                high=filtered_data['High'],
                                low=filtered_data['Low'],
                                close=filtered_data['Close'],
                                name='OHLC'),
                 row=1, col=1)
    
    # Volume
    fig.add_trace(go.Bar(x=filtered_data['Date'],
                        y=filtered_data['Volume'],
                        name='Volume'),
                 row=2, col=1)
    
    fig.update_layout(height=800, title_text=f"{selected_stock} Price and Volume")
    st.plotly_chart(fig, use_container_width=True)

# Tab 2: Market Sentiment
with tab2:
    st.subheader("Market Sentiment Analysis")
    
    # Get market sentiment
    sentiment = get_market_sentiment(selected_stock)
    
    # Display sentiment gauge
    fig_sentiment = go.Figure(go.Indicator(
        mode="gauge+number",
        value=sentiment,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Market Sentiment"},
        gauge={'axis': {'range': [-1, 1]},
               'steps': [
                   {'range': [-1, -0.5], 'color': "red"},
                   {'range': [-0.5, 0], 'color': "lightgray"},
                   {'range': [0, 0.5], 'color': "lightgray"},
                   {'range': [0.5, 1], 'color': "green"}],
               'threshold': {'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': sentiment}}))
    
    st.plotly_chart(fig_sentiment, use_container_width=True)
    
    # Display sentiment interpretation
    if sentiment > 0.5:
        st.success("Strong Positive Market Sentiment")
    elif sentiment > 0:
        st.info("Moderate Positive Market Sentiment")
    elif sentiment > -0.5:
        st.warning("Moderate Negative Market Sentiment")
    else:
        st.error("Strong Negative Market Sentiment")

# Tab 3: Risk Analysis
with tab3:
    st.subheader("Risk Analysis")
    
    # Calculate returns
    returns = filtered_data['Close'].pct_change().dropna()
    
    # Calculate metrics
    annualized_return = returns.mean() * 252
    annualized_volatility = returns.std() * np.sqrt(252)
    sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility != 0 else 0
    var_95 = calculate_var(returns, 0.95)
    var_99 = calculate_var(returns, 0.99)
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Annualized Return", f"{annualized_return:.2%}")
        st.metric("Annualized Volatility", f"{annualized_volatility:.2%}")
    with col2:
        st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
        st.metric("VaR (95%)", f"{var_95:.2%}")
    with col3:
        st.metric("VaR (99%)", f"{var_99:.2%}")
    
    # Returns distribution
    fig_returns = go.Figure()
    fig_returns.add_trace(go.Histogram(x=returns, nbinsx=50, name='Returns Distribution'))
    fig_returns.update_layout(title='Returns Distribution')
    st.plotly_chart(fig_returns, use_container_width=True)

# Tab 4: Price Prediction
with tab4:
    st.subheader("Price Trend Prediction")
    
    # Prediction settings
    col1, col2 = st.columns(2)
    with col1:
        days_to_predict = st.slider("Days to Predict", 1, 30, 7)
    with col2:
        prediction_date = st.date_input(
            "Select Date for Prediction",
            value=df['Date'].max().date(),
            min_value=df['Date'].min().date(),
            max_value=df['Date'].max().date()
        )
    
    if st.button("Generate Trend Prediction"):
        model = load_model()
        if model is not None:
            try:
                # Prepare data for prediction
                prediction_data = prepare_prediction_data(df, selected_stock)
                
                # Find the closest date in the data to the selected prediction date
                date_diff = abs(prediction_data['Date'] - pd.to_datetime(prediction_date))
                closest_date_idx = date_diff.idxmin()
                prediction_data = prediction_data.loc[:closest_date_idx]
                
                if len(prediction_data) > 0:
                    latest_data = prediction_data.iloc[-1]
                    
                    # Create features for prediction
                    features = pd.DataFrame({
                        'Close': latest_data['Close'],
                        'Daily Return': latest_data['Daily Return'],
                        'Volatility': latest_data['Volatility'],
                        'Open': latest_data['Open'],
                        'High': latest_data['High'],
                        'Low': latest_data['Low'],
                        'Volume': latest_data['Volume']
                    }, index=[0])
                    
                    # Make prediction
                    prediction = model.predict(features)
                    
                    # Calculate trend strength
                    trend_strength = abs(prediction[0])
                    
                    # Determine trend category
                    if prediction[0] > 0.02:  # More than 2% predicted increase
                        trend = "Up"
                        trend_color = "green"
                    elif prediction[0] < -0.02:  # More than 2% predicted decrease
                        trend = "Down"
                        trend_color = "red"
                    else:  # Between -2% and 2%
                        trend = "Neutral"
                        trend_color = "gray"
                    
                    # Create daily predictions
                    daily_predictions = []
                    current_price = latest_data['Close']
                    
                    for day in range(1, days_to_predict + 1):
                        # Simulate daily prediction (in a real scenario, this would use a time series model)
                        daily_change = prediction[0] / days_to_predict
                        predicted_price = current_price * (1 + daily_change)
                        
                        # Determine daily trend
                        if daily_change > 0.02:
                            daily_trend = "Up"
                            daily_color = "green"
                        elif daily_change < -0.02:
                            daily_trend = "Down"
                            daily_color = "red"
                        else:
                            daily_trend = "Neutral"
                            daily_color = "gray"
                        
                        daily_predictions.append({
                            'Day': day,
                            'Date': (pd.to_datetime(prediction_date) + pd.Timedelta(days=day)).strftime('%Y-%m-%d'),
                            'Trend': daily_trend,
                            'Color': daily_color,
                            'Predicted Change': daily_change,
                            'Predicted Price': predicted_price
                        })
                        
                        current_price = predicted_price
                    
                    # Display daily predictions in a table
                    st.subheader("Daily Trend Predictions")
                    for pred in daily_predictions:
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.write(f"**Day {pred['Day']}** ({pred['Date']})")
                        with col2:
                            st.markdown(f"<span style='color:{pred['Color']}; font-weight:bold'>{pred['Trend']}</span>", unsafe_allow_html=True)
                        with col3:
                            st.write(f"{pred['Predicted Change']:.2%}")
                        with col4:
                            st.write(f"${pred['Predicted Price']:.2f}")
                    
                    # Create trend visualization
                    fig_trend = go.Figure()
                    
                    # Add historical price
                    fig_trend.add_trace(go.Scatter(
                        x=prediction_data['Date'].tail(30),
                        y=prediction_data['Close'].tail(30),
                        name='Historical Price',
                        line=dict(color='blue')
                    ))
                    
                    # Add predicted prices
                    fig_trend.add_trace(go.Scatter(
                        x=[pd.to_datetime(pred['Date']) for pred in daily_predictions],
                        y=[pred['Predicted Price'] for pred in daily_predictions],
                        name='Predicted Prices',
                        mode='lines+markers',
                        line=dict(color='purple', dash='dot'),
                        marker=dict(
                            color=[pred['Color'] for pred in daily_predictions],
                            size=10,
                            symbol=['arrow-up' if pred['Trend'] == "Up" else 'arrow-down' if pred['Trend'] == "Down" else 'circle' for pred in daily_predictions]
                        )
                    ))
                    
                    fig_trend.update_layout(
                        title=f"{selected_stock} Price Trend Prediction",
                        xaxis_title="Date",
                        yaxis_title="Price",
                        showlegend=True
                    )
                    
                    # Show trend visualization
                    st.plotly_chart(fig_trend, use_container_width=True)
                    
                    # Add detailed historical context
                    st.info("""
                        ### Historical Context
                        - **30-day Average Return:** {:.2%}
                        - **30-day Volatility:** {:.2%}
                        - **Current Daily Return:** {:.2%}
                        - **Current Volume:** {:,}
                    """.format(
                        prediction_data['Daily Return'].tail(30).mean(),
                        prediction_data['Volatility'].tail(30).mean(),
                        latest_data['Daily Return'],
                        latest_data['Volume']
                    ))
                    
                    # Add trading recommendations based on trend
                    st.warning("""
                        ### Trading Recommendations
                        {}
                    """.format(
                        "Consider buying" if trend == "Up"
                        else "Consider selling" if trend == "Down"
                        else "Hold position or wait for clearer trend"
                    ))
                else:
                    st.error("No data available for the selected date")
            except Exception as e:
                st.error(f"Error generating prediction: {str(e)}")
        else:
            st.error("Model not available for prediction") 