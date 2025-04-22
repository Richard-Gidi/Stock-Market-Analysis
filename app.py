"""
Stock Market Analysis Application
This application provides comprehensive stock market analysis with the following features:
1. Technical Analysis with multiple indicators
2. Portfolio Analysis with performance metrics
3. Advanced Risk Analysis
4. Market Sentiment Analysis
5. Price Prediction with multiple models
6. Correlation Analysis
7. Real-time Market Data
8. AI-Powered Chat Assistant
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import ta
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
from scipy.stats import skew, kurtosis
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import joblib
import time

# Download required NLTK data
nltk.download('vader_lexicon')
nltk.download('punkt')

# Set page configuration
st.set_page_config(
    page_title="Advanced Stock Market Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    /* Main background and general text */
    .main {
        background-color: #0e0e11;  /* dark theme */
        color: white;
    }

    /* Metrics container card (light) */
    div[data-testid="metric-container"] {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        color: black !important;
    }

    /* Metric label (e.g., "Average Sentiment") */
    div[data-testid="metric-container"] > label {
        color: black !important;
        font-weight: bold;
    }

    /* Metric value (e.g., "0.33") */
    div[data-testid="metric-container"] > div {
        color: black !important;
        font-size: 1.5rem;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)


# Initialize session state for chat
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Title and description
st.title("Advanced Stock Market Analysis Dashboard")
st.write("Comprehensive stock analysis with technical indicators, portfolio optimization, and advanced risk metrics")

# Function to load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("SPY_PG_JNJ STOCKS DATA.csv")
        df['Date'] = pd.to_datetime(df['Date'])
        df['Stock'] = df['StockNumber'].str.extract(r'([A-Za-z]+)')
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Function to load XGBoost model
@st.cache_data
def load_models():
    try:
        model = joblib.load("Models/xgboost_model.pkl")
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Function to calculate Value at Risk
def calculate_var(returns, confidence_level=0.95):
    """
    Calculate Value at Risk (VaR) for a given confidence level
    Args:
        returns (pd.Series): Series of returns
        confidence_level (float): Confidence level for VaR calculation (default: 0.95)
    Returns:
        float: Value at Risk
    """
    return np.percentile(returns, (1 - confidence_level) * 100)

# Function to calculate advanced risk metrics
def calculate_advanced_risk_metrics(returns):
    """
    Calculate advanced risk metrics for a portfolio
    Args:
        returns (pd.Series): Series of returns
    Returns:
        dict: Dictionary containing risk metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['Annualized Return'] = returns.mean() * 252
    metrics['Annualized Volatility'] = returns.std() * np.sqrt(252)
    metrics['Sharpe Ratio'] = metrics['Annualized Return'] / metrics['Annualized Volatility'] if metrics['Annualized Volatility'] != 0 else 0
    
    # Advanced metrics
    metrics['Sortino Ratio'] = metrics['Annualized Return'] / (returns[returns < 0].std() * np.sqrt(252)) if returns[returns < 0].std() != 0 else 0
    metrics['Maximum Drawdown'] = (returns.cumsum().expanding().max() - returns.cumsum()).max()
    metrics['Skewness'] = skew(returns)
    metrics['Kurtosis'] = kurtosis(returns)
    
    # Value at Risk
    metrics['VaR_95'] = np.percentile(returns, 5)
    metrics['VaR_99'] = np.percentile(returns, 1)
    metrics['CVaR_95'] = returns[returns <= metrics['VaR_95']].mean()
    
    # Stationarity test
    adf_result = adfuller(returns.dropna())
    metrics['ADF Statistic'] = adf_result[0]
    metrics['ADF p-value'] = adf_result[1]
    
    return metrics

# Function to analyze portfolio
def analyze_portfolio(stocks_data, weights):
    portfolio_returns = pd.DataFrame()
    for stock, weight in weights.items():
        stock_data = stocks_data[stocks_data['Stock'] == stock]
        returns = stock_data['Close'].pct_change()
        portfolio_returns[stock] = returns * weight
    
    portfolio_returns['Total'] = portfolio_returns.sum(axis=1)
    return portfolio_returns

# Make sure to install lxml first: pip install lxml

def advanced_sentiment_analysis(stock_name):
    try:
        # Get news articles
        url = f"https://news.google.com/rss/search?q={stock_name}+stock&hl=en-US&gl=US&ceid=US:en"
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'xml')  # Requires lxml or xml parser

        items = soup.find_all('item')
        
        # Initialize sentiment analyzers
        sia = SentimentIntensityAnalyzer()
        sentiments = []
        subjectivity_scores = []
        
        for item in items[:10]:  # Analyze top 10 articles
            title = item.title.text
            description = item.description.text if item.description else ""
            
            # VADER sentiment
            vader_sentiment = sia.polarity_scores(title + " " + description)
            sentiments.append(vader_sentiment['compound'])
            
            # TextBlob sentiment and subjectivity
            blob = TextBlob(title + " " + description)
            subjectivity_scores.append(blob.sentiment.subjectivity)
        
        if not sentiments:
            st.warning("No news articles found for sentiment analysis.")
            return None

        return {
            'avg_sentiment': np.mean(sentiments),
            'sentiment_std': np.std(sentiments),
            'subjectivity': np.mean(subjectivity_scores),
            'article_count': len(sentiments)
        }
    except Exception as e:
        st.warning(f"Could not perform sentiment analysis: {str(e)}")
        return None


# Function to prepare prediction data
def prepare_prediction_data(df, stock_name):
    stock_data = df[df['Stock'] == stock_name].copy()
    
    # Calculate returns and volatility
    stock_data['Daily Return'] = stock_data['Close'].pct_change()
    stock_data['Volatility'] = stock_data['Daily Return'].rolling(window=20).std()
    
    return stock_data

def get_realtime_data(symbol, retries=3, delay=2):
    for attempt in range(retries):
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            return {
                'current_price': info.get('currentPrice', 0),
                'day_high': info.get('dayHigh', 0),
                'day_low': info.get('dayLow', 0),
                'volume': info.get('volume', 0),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'dividend_yield': info.get('dividendYield', 0)
            }
        except Exception as e:
            st.warning(f"Attempt {attempt+1} failed: {str(e)}")
            time.sleep(delay)
    st.error("Could not fetch real-time data after multiple attempts.")
    return None

# Function to generate AI response
def generate_ai_response(user_input, context):
    """
    Generate AI response using OpenAI's GPT model
    Args:
        user_input (str): User's message
        context (list): Chat history for context
    Returns:
        str: AI's response
    """
    try:
        # Prepare the conversation history
        messages = [
            {"role": "system", "content": "You are a knowledgeable stock market analyst assistant. Provide accurate and helpful information about stocks, market trends, and investment strategies."}
        ]
        
        # Add context from chat history
        for msg in context[-5:]:  # Use last 5 messages for context
            messages.append({"role": "user", "content": msg['user']})
            messages.append({"role": "assistant", "content": msg['assistant']})
        
        # Add current user input
        messages.append({"role": "user", "content": user_input})
        
        # Generate response
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=500,
            temperature=0.7
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"I apologize, but I encountered an error: {str(e)}"

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
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Portfolio Analysis", 
    "Risk Analysis", 
    "Market Sentiment", 
    "Price Prediction", 
    "Correlation Analysis",
    "Real-time Data"
])

# Tab 1: Portfolio Analysis
with tab1:
    st.subheader("Portfolio Analysis")
    
    # Add explanation for portfolio analysis
    st.markdown("""
    ### What is Portfolio Analysis?
    Portfolio analysis helps you understand how your investments work together. A well-diversified portfolio can reduce risk while maintaining returns. 
    
    **Why it matters**: Proper allocation can help protect your investments during market downturns while capturing growth during upswings.
    """)
    
    # Portfolio weights input
    st.write("Enter portfolio weights (must sum to 1):")
    weights = {}
    col1, col2, col3 = st.columns(3)
    for i, stock in enumerate(stock_names):
        with col1 if i % 3 == 0 else col2 if i % 3 == 1 else col3:
            weights[stock] = st.number_input(f"{stock} Weight", min_value=0.0, max_value=1.0, value=1.0/len(stock_names), step=0.01)
    
    if abs(sum(weights.values()) - 1.0) > 0.01:
        st.error("Portfolio weights must sum to 1!")
    else:
        # Calculate portfolio returns
        portfolio_returns = analyze_portfolio(df, weights)
        
        # Plot portfolio performance
        fig_portfolio = go.Figure()
        for stock in stock_names:
            fig_portfolio.add_trace(go.Scatter(x=portfolio_returns.index, y=portfolio_returns[stock],
                                            name=stock, mode='lines'))
        fig_portfolio.add_trace(go.Scatter(x=portfolio_returns.index, y=portfolio_returns['Total'],
                                        name='Total Portfolio', line=dict(color='black', width=2)))
        
        fig_portfolio.update_layout(title='Portfolio Performance', height=500)
        st.plotly_chart(fig_portfolio, use_container_width=True)
        
        # Portfolio metrics with explanations
        portfolio_metrics = calculate_advanced_risk_metrics(portfolio_returns['Total'])
        st.write("Portfolio Metrics:")
        
        # Add metrics explanations
        st.markdown("""
        ### Understanding Portfolio Metrics
        """)
        
        col1, col2, col3 = st.columns(3)
        
        # Show metrics with explanations
        with col1:
            st.metric("Annualized Return", f"{portfolio_metrics['Annualized Return']:.4f}")
            st.markdown("**What it means**: Your portfolio's yearly return rate. Higher is better. This number shows how much your investments grow annually.")
            
            st.metric("Sharpe Ratio", f"{portfolio_metrics['Sharpe Ratio']:.4f}")
            st.markdown("**What it means**: Return per unit of risk. Above 1.0 is good, above 2.0 is excellent. Higher values indicate better risk-adjusted returns.")
            
            st.metric("Maximum Drawdown", f"{portfolio_metrics['Maximum Drawdown']:.4f}")
            st.markdown("**What it means**: The largest drop from peak to trough. Lower values are better. This shows your worst-case scenario loss.")
        
        with col2:
            st.metric("Annualized Volatility", f"{portfolio_metrics['Annualized Volatility']:.4f}")
            st.markdown("**What it means**: How much your portfolio's returns fluctuate annually. Lower volatility generally means more stable returns.")
            
            st.metric("Sortino Ratio", f"{portfolio_metrics['Sortino Ratio']:.4f}")
            st.markdown("**What it means**: Similar to Sharpe Ratio but only considers downside risk. Higher values indicate better returns for downside risk taken.")
            
            st.metric("Skewness", f"{portfolio_metrics['Skewness']:.4f}")
            st.markdown("**What it means**: Positive values indicate more upside returns, negative values indicate more downside returns.")
        
        with col3:
            st.metric("VaR_95", f"{portfolio_metrics['VaR_95']:.4f}")
            st.markdown("**What it means**: With 95% confidence, your portfolio won't lose more than this percentage in a single day. Lower absolute values are better.")
            
            st.metric("CVaR_95", f"{portfolio_metrics['CVaR_95']:.4f}")
            st.markdown("**What it means**: The expected loss when losses exceed the VaR threshold. This shows how bad your worst days might be.")
            
            st.metric("Kurtosis", f"{portfolio_metrics['Kurtosis']:.4f}")
            st.markdown("**What it means**: Higher values indicate more extreme returns (positive or negative). A normal distribution has kurtosis = 0.")
        
        # Add action recommendations based on metrics
        st.markdown("""
        ### What Actions Can You Take?
        
        - **If Sharpe Ratio is low** (<1.0): Consider rebalancing your portfolio to reduce risk or find assets with better risk-return profiles.
        - **If Maximum Drawdown is high** (>20%): Your portfolio may be too volatile for your risk tolerance. Consider adding more defensive assets.
        - **If Skewness is negative**: Your portfolio has more frequent small gains but occasional large losses. Consider adding assets with positive skew.
        - **If Annualized Return is lower than expected**: Review your asset allocation and consider increasing allocation to higher growth assets.
        
        Remember that higher returns typically come with higher risk. The ideal portfolio balances your return goals with your risk tolerance.
        """)

# Tab 2: Risk Analysis
with tab2:
    st.subheader("Advanced Risk Analysis")
    
    # Add explanation for risk analysis
    st.markdown("""
    ### Understanding Risk Analysis
    
    Risk analysis helps you understand potential downsides of your investments. This analysis goes beyond simple volatility measures to give you a comprehensive view of your investment risks.
    
    **Why it matters**: Understanding risk helps you make better investment decisions and avoid unexpected losses.
    """)
    
    # Calculate advanced risk metrics
    returns = filtered_data['Close'].pct_change().dropna()
    risk_metrics = calculate_advanced_risk_metrics(returns)
    
    # Display risk metrics with explanations
    st.write("Risk Metrics:")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Annualized Return", f"{risk_metrics['Annualized Return']:.4f}")
        st.markdown("**What it means**: The yearly return rate for this stock. Higher is better.")
        
        st.metric("Sharpe Ratio", f"{risk_metrics['Sharpe Ratio']:.4f}")
        st.markdown("**What it means**: Return per unit of risk. Above 1.0 is good. This helps you compare investments with different risk levels.")
        
        st.metric("VaR_95", f"{risk_metrics['VaR_95']:.4f}")
        st.markdown("**What it means**: Value at Risk - With 95% confidence, your daily loss won't exceed this percentage. This is your likely worst-case daily scenario.")
    
    with col2:
        st.metric("Annualized Volatility", f"{risk_metrics['Annualized Volatility']:.4f}")
        st.markdown("**What it means**: How much returns fluctuate annually. Higher values indicate a more volatile investment.")
        
        st.metric("Maximum Drawdown", f"{risk_metrics['Maximum Drawdown']:.4f}")
        st.markdown("**What it means**: The largest drop from peak to trough. This represents the worst historical loss you would have experienced.")
        
        st.metric("VaR_99", f"{risk_metrics['VaR_99']:.4f}")
        st.markdown("**What it means**: With 99% confidence, your daily loss won't exceed this percentage. This is your extreme worst-case daily scenario.")
        
    with col3:
        st.metric("Sortino Ratio", f"{risk_metrics['Sortino Ratio']:.4f}")
        st.markdown("**What it means**: Similar to Sharpe Ratio but only penalizes downside volatility. Better for comparing investments with different downside risks.")
        
        st.metric("Skewness", f"{risk_metrics['Skewness']:.4f}")
        st.markdown("**What it means**: Distribution symmetry of returns. Positive values indicate more frequent small losses and occasional large gains.")
        
        st.metric("CVaR_95", f"{risk_metrics['CVaR_95']:.4f}")
        st.markdown("**What it means**: Conditional Value at Risk - The average loss when losses exceed VaR. This shows how bad your worst days might be.")
    
    # Risk visualization
    fig_risk = make_subplots(rows=2, cols=2, subplot_titles=("Returns Distribution", "Cumulative Returns",
                                                           "Rolling Volatility", "Drawdown"))
    
    # Returns distribution
    fig_risk.add_trace(go.Histogram(x=returns, nbinsx=50, name='Returns Distribution'),
                      row=1, col=1)
    
    # Cumulative returns
    cum_returns = (1 + returns).cumprod()
    fig_risk.add_trace(go.Scatter(x=returns.index, y=cum_returns, name='Cumulative Returns'),
                      row=1, col=2)
    
    # Rolling volatility
    rolling_vol = returns.rolling(window=20).std() * np.sqrt(252)
    fig_risk.add_trace(go.Scatter(x=returns.index, y=rolling_vol, name='Rolling Volatility'),
                      row=2, col=1)
    
    # Drawdown
    drawdown = (cum_returns / cum_returns.cummax() - 1) * 100
    fig_risk.add_trace(go.Scatter(x=returns.index, y=drawdown, name='Drawdown (%)'),
                      row=2, col=2)
    
    fig_risk.update_layout(height=800, showlegend=False)
    st.plotly_chart(fig_risk, use_container_width=True)
    
    # Add explanations for charts
    st.markdown("""
    ### Understanding the Risk Charts
    
    **Returns Distribution**: Shows how daily returns are distributed. A wider distribution indicates higher volatility. Skewed distributions (not symmetrical) can indicate higher risk of extreme outcomes.
    
    **Cumulative Returns**: Shows how your investment would have grown over time. Steeper upward slopes indicate stronger performance periods.
    
    **Rolling Volatility**: Shows how volatility changes over time. Spikes indicate periods of market stress or uncertainty.
    
    **Drawdown**: Shows periods of decline from previous peaks. Deeper drawdowns indicate more severe market corrections or crashes.
    
    ### What Actions Can You Take?
    
    - **If Volatility is too high for your comfort**: Consider adding more stable assets to your portfolio or using hedging strategies.
    - **If Maximum Drawdown is larger than you can tolerate**: Reassess your risk tolerance or add uncorrelated assets to reduce overall portfolio drawdowns.
    - **If VaR values are concerning**: Consider position sizing to limit potential losses to acceptable amounts.
    - **If Sharpe/Sortino ratios are low**: Look for alternative investments with better risk-adjusted returns.
    
    Remember that different market environments favor different investment styles. What worked in the past may not work in the future.
    """)

# Tab 3: Market Sentiment
with tab3:
    st.subheader("Advanced Market Sentiment Analysis")
    
    # Add explanation for sentiment analysis
    st.markdown("""
    ### Understanding Market Sentiment
    
    Market sentiment analysis evaluates the mood and emotions of investors toward a stock by analyzing news articles and social media. Sentiment can often drive short-term price movements.
    
    **Why it matters**: Sentiment can predict price movements before they're reflected in technical indicators, giving you an edge in timing decisions.
    """)
    
    # Get advanced sentiment analysis
    sentiment_data = advanced_sentiment_analysis(selected_stock)
    
    if sentiment_data:
        # Display sentiment metrics with explanations
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average Sentiment", f"{sentiment_data['avg_sentiment']:.4f}")
            st.markdown("""
            **What it means**: Ranges from -1 (extremely negative) to +1 (extremely positive). 
            - Above 0.3: Strongly positive news coverage
            - 0 to 0.3: Mildly positive coverage
            - -0.3 to 0: Mildly negative coverage
            - Below -0.3: Strongly negative coverage
            """)
        with col2:
            st.metric("Sentiment Volatility", f"{sentiment_data['sentiment_std']:.4f}")
            st.markdown("""
            **What it means**: How much sentiment varies across news articles. 
            - High values: Mixed or contradicting news coverage
            - Low values: Consistent sentiment across news sources
            """)
        with col3:
            st.metric("Subjectivity", f"{sentiment_data['subjectivity']:.4f}")
            st.markdown("""
            **What it means**: How opinionated the news coverage is (0 = objective facts, 1 = highly subjective opinions).
            - High values: More opinion-based coverage, potentially less reliable
            - Low values: More fact-based coverage, potentially more reliable
            """)
        
        # Sentiment distribution
        fig_sentiment = go.Figure()
        fig_sentiment.add_trace(go.Indicator(
            mode="gauge+number",
            value=sentiment_data['avg_sentiment'],
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
                                'value': sentiment_data['avg_sentiment']}}))
        
        st.plotly_chart(fig_sentiment, use_container_width=True)
        
        # Add action recommendations
        st.markdown("""
        ### What Actions Can You Take?
        
        **When Sentiment is Positive (>0.3):**
        - Short-term investors might consider taking advantage of positive momentum
        - Consider whether positive news is already priced in
        - Watch for overly optimistic sentiment which might indicate overvaluation
        
        **When Sentiment is Negative (<-0.3):**
        - Assess if negative news presents a buying opportunity for fundamentally sound companies
        - Consider whether sentiment might worsen before improving
        - Check if the negative sentiment is justified by fundamentals
        
        **When Sentiment is Mixed (High Volatility):**
        - Research further to understand conflicting viewpoints
        - Consider waiting for clarity before making significant moves
        - Look for other indicators to confirm or reject sentiment signals
        
        **When Subjectivity is High (>0.6):**
        - Give more weight to hard data and less to news sentiment
        - Look for more objective sources of information
        
        Remember that sentiment often drives short-term price movements, while fundamentals matter more in the long run.
        """)

# Tab 4: Price Prediction
with tab4:
    st.subheader("Advanced Price Prediction")
    
    # Add explanation for price prediction
    st.markdown("""
    ### Understanding Price Prediction
    
    This analysis uses machine learning models to forecast potential future stock prices based on historical patterns, technical indicators, and market conditions.
    
    **Why it matters**: While no prediction is guaranteed, these forecasts can help you understand potential price movements and plan your investment strategy accordingly.
    
    **Note**: All predictions are estimates and should be used as one of many tools in your investment decision-making process.
    """)
    
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
    
    if st.button("Generate Prediction"):
        model = load_models()
        if model:
            try:
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
                    
                    # Make predictions for each day
                    predictions = []
                    current_price = latest_data['Close']
                    dates = []
                    
                    for day in range(1, days_to_predict + 1):
                        # Make prediction
                        prediction = model.predict(features)
                        predicted_return = prediction[0]
                        
                        # Calculate predicted price
                        predicted_price = current_price * (1 + predicted_return)
                        predictions.append(predicted_price)
                        
                        # Update features for next day
                        current_price = predicted_price
                        features['Close'] = predicted_price
                        features['Daily Return'] = predicted_return
                        features['Volatility'] = prediction_data['Volatility'].mean()  # Use average volatility
                        
                        # Add date
                        dates.append(prediction_data['Date'].iloc[-1] + pd.Timedelta(days=day))
                    
                    # Create DataFrame for predictions
                    predictions_df = pd.DataFrame({
                        'Date': dates,
                        'Predicted Price': predictions,
                        'Change': [p - latest_data['Close'] for p in predictions],
                        'Direction': ['↑' if p > latest_data['Close'] else '↓' for p in predictions]
                    })
                    
                    # Display predictions table with up/down indicators
                    st.write("Daily Price Predictions:")
                    st.dataframe(predictions_df.style.format({
                        'Predicted Price': '${:.2f}',
                        'Change': '${:.2f}'
                    }).applymap(lambda x: 'color: green' if x == '↑' else 'color: red' if x == '↓' else '', subset=['Direction']))
                    
                    # Plot historical and predicted prices
                    fig_pred = go.Figure()
                    
                    # Historical data
                    fig_pred.add_trace(go.Scatter(
                        x=prediction_data['Date'],
                        y=prediction_data['Close'],
                        name='Historical Price',
                        line=dict(color='blue')
                    ))
                    
                    # Predicted data
                    fig_pred.add_trace(go.Scatter(
                        x=predictions_df['Date'],
                        y=predictions_df['Predicted Price'],
                        name='Predicted Price',
                        line=dict(color='red', dash='dash')
                    ))
                    
                    # Add markers for each prediction point with up/down indicators
                    for i, row in predictions_df.iterrows():
                        fig_pred.add_trace(go.Scatter(
                            x=[row['Date']],
                            y=[row['Predicted Price']],
                            mode='markers+text',
                            name='Daily Predictions',
                            marker=dict(
                                size=8,
                                color='green' if row['Direction'] == '↑' else 'red'
                            ),
                            text=row['Direction'],
                            textposition='top center',
                            showlegend=False
                        ))
                    
                    # Add confidence intervals (
                    
                    # Add confidence intervals (assuming 95% confidence)
                    confidence_interval = 0.05  # 5% confidence interval
                    fig_pred.add_trace(go.Scatter(
                        x=predictions_df['Date'],
                        y=predictions_df['Predicted Price'] * (1 + confidence_interval),
                        fill=None,
                        mode='lines',
                        line_color='rgba(255,0,0,0.2)',
                        name='Upper Bound',
                        showlegend=False
                    ))
                    
                    fig_pred.add_trace(go.Scatter(
                        x=predictions_df['Date'],
                        y=predictions_df['Predicted Price'] * (1 - confidence_interval),
                        fill='tonexty',
                        mode='lines',
                        line_color='rgba(255,0,0,0.2)',
                        name='Lower Bound',
                        showlegend=False
                    ))
                    
                    fig_pred.update_layout(
                        title='Price Prediction with Daily Forecasts',
                        xaxis_title='Date',
                        yaxis_title='Price',
                        height=600,
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig_pred, use_container_width=True)
                    
                    # Display summary statistics with up/down indicators
                    st.write("Prediction Summary:")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Starting Price", f"${latest_data['Close']:.2f}")
                        final_direction = '↑' if predictions[-1] > latest_data['Close'] else '↓'
                        st.metric("Final Predicted Price", f"${predictions[-1]:.2f}", 
                                delta=f"{final_direction} ${abs(predictions[-1] - latest_data['Close']):.2f}")
                    with col2:
                        total_return = (predictions[-1] - latest_data['Close']) / latest_data['Close']
                        st.metric("Total Return", f"{total_return:.2%}")
                        daily_avg_return = total_return / days_to_predict
                        st.metric("Average Daily Return", f"{daily_avg_return:.2%}")
                    with col3:
                        price_change = predictions[-1] - latest_data['Close']
                        st.metric("Price Change", f"${price_change:.2f}")
                        volatility = np.std([(p2 - p1) / p1 for p1, p2 in zip(predictions[:-1], predictions[1:])])
                        st.metric("Predicted Volatility", f"{volatility:.2%}")
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")

# Tab 5: Correlation Analysis
with tab5:
    st.subheader("Correlation Analysis")
    
    # Calculate correlation matrix
    correlation_data = pd.DataFrame()
    for stock in stock_names:
        stock_data = df[df['Stock'] == stock]
        correlation_data[stock] = stock_data['Close'].pct_change()
    
    correlation_matrix = correlation_data.corr()
    
    # Plot correlation heatmap
    fig_corr = go.Figure(data=go.Heatmap(
        z=correlation_matrix,
        x=correlation_matrix.columns,
        y=correlation_matrix.columns,
        colorscale='RdBu',
        zmid=0
    ))
    
    fig_corr.update_layout(title='Stock Correlation Matrix', height=600)
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Display correlation metrics
    st.write("Correlation Metrics:")
    st.dataframe(correlation_matrix)

# Tab 6: Real-time Data
with tab6:
    st.subheader("Real-time Market Data")
    
    # Get real-time data
    realtime_data = get_realtime_data(selected_stock)
    
    if realtime_data:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Price", f"${realtime_data['current_price']:.2f}")
            st.metric("Day High", f"${realtime_data['day_high']:.2f}")
            st.metric("Day Low", f"${realtime_data['day_low']:.2f}")
        with col2:
            st.metric("Volume", f"{realtime_data['volume']:,}")
            st.metric("Market Cap", f"${realtime_data['market_cap']/1e9:.2f}B")
        with col3:
            st.metric("P/E Ratio", f"{realtime_data['pe_ratio']:.2f}")
            st.metric("Dividend Yield", f"{realtime_data['dividend_yield']*100:.2f}%") 