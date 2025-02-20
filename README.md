# Stock Price Prediction with Sentiment Analysis

## Overview
This project predicts Apple Inc.'s (AAPL) stock price using a hybrid deep learning model combining CNN, LSTM, and Transformer architectures. It integrates sentiment analysis from financial news to improve prediction accuracy.

## Features
- Fetches real-time stock market data using `yfinance`
- Collects financial news sentiment using `Alpha Vantage API`
- Applies sentiment analysis using `FinBERT`
- Preprocesses stock and sentiment data for training
- Constructs sequences for time-series forecasting
- Builds and trains a deep learning model combining CNN, LSTM, and Transformers
- Predicts the next day's stock closing price

## Installation
Ensure you have Python installed, then install the required dependencies:
```bash
pip install yfinance numpy pandas requests matplotlib transformers scikit-learn tensorflow
```

## Usage
Run the script to fetch stock data, analyze sentiment, train the model, and predict stock prices:
```bash
python main.py
```

## Modules
### 1. Fetch Stock Data
Retrieves historical stock data for a specified period and interval.
```python
def fetch_stock_data(ticker, period="7d", interval="1h"):
    stock = yf.download(ticker, period=period, interval=interval)
    return stock[['Open', 'High', 'Low', 'Close', 'Volume']]
```

## 2. Fetch Financial Statements

yf.Ticker(ticker): Initializes a Yahoo Finance object for the given stock ticker.

.financials.T: Retrieves the income statement, transposed for easier row-wise access.

.balance_sheet.T: Retrieves the balance sheet, also transposed.

.cashflow.T: Retrieves the cash flow statement, transposed.

```python
stock = yf.Ticker(ticker)
income_stmt = stock.financials.T  # Income Statement
balance_sheet = stock.balance_sheet.T  # Balance Sheet
cash_flow = stock.cashflow.T  # Cash Flow Statement
```
The function tries to extract specific key financial metrics:
Income Statement: 'Total Revenue', 'Net Income'
Balance Sheet: 'Total Assets', 'Total Liabilities'
Cash Flow Statement: 'Operating Cash Flow'
Concatenates the extracted data into a single DataFrame.

```python
financials = pd.concat([
    income_stmt[['Total Revenue', 'Net Income']],  # From Income Statement
    balance_sheet[['Total Assets', 'Total Liabilities']],  # From Balance Sheet
    cash_flow[['Operating Cash Flow']]  # From Cash Flow Statement
], axis=1)
```
Replaces NaN values with 0 to avoid issues during processing.
```python
financials.fillna(0, inplace=True)  # Handle missing values
```
Returns the final compiled financial statement DataFrame.
```python
return financials
```

### 2. Fetch Financial News
Fetches the latest financial news headlines using the Alpha Vantage API.
```python
def fetch_financial_news():
    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers=AAPL&apikey={ALPHA_VANTAGE_API_KEY}"
    response = requests.get(url).json()
```

### 3. Sentiment Analysis
Analyzes sentiment of financial news headlines using FinBERT.
```python
def analyze_sentiment(texts):
    results = sentiment_model(texts, truncation=True, max_length=512)
    return [1 if res['label'] == 'positive' else (-1 if res['label'] == 'negative' else 0) for res in results]
```

### 4. Data Preprocessing
Merges sentiment data into stock dataset and normalizes features.
```python
def preprocess_data(stock_df, sentiment_scores):
    stock_df["Sentiment"] = np.random.choice(sentiment_scores, size=len(stock_df))
    stock_df.dropna(inplace=True)
    return stock_df
```

### 5. Model Architecture
A deep learning model combining CNN, LSTM, and Transformer layers for stock price prediction.
```python
def create_hybrid_model(seq_length, feature_dim):
    inputs = Input(shape=(seq_length, feature_dim))
    x = Conv1D(filters=128, kernel_size=5, activation="relu", padding="same")(inputs)
    x = Conv1D(filters=64, kernel_size=3, activation="relu", padding="same")(x)
    attn_output = MultiHeadAttention(num_heads=8, key_dim=feature_dim)(x, x)
    x = LayerNormalization()(attn_output + x)
    x = LSTM(64, return_sequences=True)(x)
    x = LSTM(32)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1, activation="linear")(x)
    model = Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model
```

## Results
- The model predicts the next closing stock price for AAPL.
- Compares the predicted price with actual market data.

## License
This project is open-source and available under the MIT License.

## Contact
For further inquiries, contact:
- Email: fabrisio.ponte@gmail.com
- LinkedIn: [Your Profile](https://linkedin.com/in/fabrisio-ponte)

