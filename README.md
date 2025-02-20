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

### 3. Fetch Financial News
Fetches the latest financial news headlines using the Alpha Vantage API.
```python
def fetch_financial_news():
    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers=AAPL&apikey={ALPHA_VANTAGE_API_KEY}"
    response = requests.get(url).json()
```
Checking for 'feed': The response from Alpha Vantage should contain a key called 'feed', which contains the list of news articles.

Extracting Headlines:
The list comprehension loops through the first 10 articles from the feed.
article.get("headline", "No headline available") retrieves the headline for each article. If no headline is found, it returns "No headline available".
This will return a list of the top 10 headlines.

```python
if 'feed' in response:
    headlines = [article.get("headline", "No headline available") for article in response["feed"][:10]]
    return headlines
```

### 4. Sentiment Analysis
Analyzes sentiment of financial news headlines using FinBERT.

pipeline("text-classification", model="ProsusAI/finbert"): This line initializes a transformer-based sentiment analysis model called FinBERT, specifically trained for financial data (i.e., it’s sensitive to the language used in financial reports or news).

FinBERT is a model developed by ProsusAI, fine-tuned on financial text for sentiment classification.

```python
sentiment_model = pipeline("text-classification", model="ProsusAI/finbert")
```
sentiment_model(texts, truncation=True, max_length=512): This sends the list of texts (like news headlines) to the sentiment model and:

truncation=True: Ensures that if a text exceeds the model's input size (512 tokens for BERT-based models), it will be truncated.

max_length=512: The model has a maximum input length of 512 tokens, which is typical for BERT models.
results: The model returns a list of predictions where each entry corresponds to the sentiment classification of a given text. Each prediction will contain a label (e.g., 'positive', 'negative', 'neutral') and a score.

```python
def analyze_sentiment(texts):
    results = sentiment_model(texts, truncation=True, max_length=512)
    return [1 if res['label'] == 'positive' else (-1 if res['label'] == 'negative' else 0) for res in results]
```
Sentiment Mapping:

The function processes the model’s results:
1 for positive sentiment,
-1 for negative sentiment,
0 for neutral sentiment.
This list comprehension loops through the results and maps each label to its corresponding numerical value.

```python
return [1 if res['label'] == 'positive' else (-1 if res['label'] == 'negative' else 0) for res in results]
```

### 4. Data Preprocessing
Merges sentiment data into stock dataset and normalizes features.

This function preprocess_data is designed to prepare the stock data by adding sentiment scores and merging relevant financial data for further analysis. Here's a step-by-step explanation:

Add Sentiment Scores:

np.random.choice(sentiment_scores, size=len(stock_df)) randomly selects sentiment scores from the provided sentiment_scores list and assigns them to the new column. This is done randomly, which may not be ideal if sentiment data is meant to be linked with the stock data based on news articles (e.g., each news headline should correspond to specific stock data).

Potential Issue: The sentiment scores are being assigned randomly, which might not reflect actual correlations between the stock price movements and the sentiment of news articles.

```python
stock_df["Sentiment"] = np.random.choice(sentiment_scores, size=len(stock_df))
```

Align Financial Data to Stock Data Dates:

reindex(): This re-aligns the financial data to the dates of the stock data (stock_df).
method='ffill': The forward fill method is used, meaning that if a financial value for a specific date is missing, the function fills it with the most recent available value.
This ensures that financial data is properly aligned with the dates in the stock price data.

```python
financial_data = financial_data.reindex(stock_df.index, method='ffill')
```

Merge Stock Data with Financial Data:

pd.concat() merges the stock data (stock_df) with the financial data along columns (axis=1). After this step, the stock_df will include additional financial features like revenue, net income, etc., in addition to the stock price and sentiment.

```python
stock_df = pd.concat([stock_df, financial_data], axis=1)
```

## Sequence Creation:

Function Arguments:

data: This is the dataset containing stock features and additional data (e.g., stock price, sentiment, financial indicators). It’s expected to be a 2D array or DataFrame with shape (num_samples, num_features).
seq_length=30: This is the length of the sequence. The default is 30, meaning the model will use the previous 30 time steps to predict the next stock price.

```python
def create_sequences(data, seq_length=30):
```

 Loop to Create Sequences:
 
 Looping through the data: The loop iterates through the dataset, creating sequences of the previous seq_length time steps (30 by default).
 
data[i:i+seq_length]: This extracts a sequence of seq_length time steps from the data, starting at index i and ending at index i+seq_length. This forms the input sequence to predict the future price.

data[i+seq_length, 3]: This appends the label (target value) to the labels list. The label is the closing price of the next time step (i.e., i+seq_length). The 3 indicates that you’re selecting the 4th column (index 3) in the data, which corresponds to the Closing price.

 ```python
for i in range(len(data) - seq_length):
        sequences.append(data[i:i+seq_length])
        labels.append(data[i+seq_length, 3])
```

Return sequences of labels:

sequences: A list of sequences of length seq_length, where each sequence is a slice of the historical data.
labels: A list of corresponding labels (closing prices) for each sequence, which the model will predict.

The sequences and labels are converted into NumPy arrays before being returned. This is because NumPy arrays are typically used in machine learning models for efficient computation.

```python
return np.array(sequences), np.array(labels)
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

The create_hybrid_model function defines a hybrid model combining both CNN (Convolutional Neural Network) layers and Transformer (Multi-Head Attention) layers to process sequential data like stock prices, which can benefit from capturing both local and global dependencies.

Here's a breakdown of the layers and architecture:

 Input Layer:

 seq_length: Length of the input sequence (e.g., 30 time steps).
feature_dim: Number of features (columns) in the dataset. This might include stock prices, sentiment scores, technical indicators, etc.

The input shape is (seq_length, feature_dim), indicating sequences of seq_length time steps, with feature_dim features at each time step.

```python
 inputs = Input(shape=(seq_length, feature_dim))
```

## Convolutional Layers:

Conv1D stands for 1-dimensional convolution.
Convolution layers are used to detect patterns (features) in the data by applying filters (also called kernels) over the input.

In your case, you are working with time-series data (e.g., stock prices over time), and Conv1D helps identify patterns within a window of consecutive time steps.

## Parameters in Detail:

Inputs:

inputs represents the input layer of the model, where the data is fed into the network.
Input(shape=(seq_length, feature_dim)) specifies the shape of the input data.
seq_length: The number of time steps in each sequence (e.g., 30 time steps).
feature_dim: The number of features (e.g., stock prices, technical indicators, sentiment scores) at each time step in the sequence.
For example, if you have 30 time steps and 5 features (e.g., Open, Close, Volume, Sentiment, RSI), the input shape will be (30, 5).

So, inputs is a tensor (data structure) representing the input data that will flow through the model.

```python
inputs = Input(shape=(seq_length, feature_dim))
```

Convolutional Layers CONV1D

filters=128 and filters=64:

This defines the number of filters (kernels) used in the convolution process.
Filter: It's a small matrix that "slides" over the input data (e.g., the stock price time series) to extract features.
128 filters in the first convolution layer means the model will learn 128 different features from the input data.
64 filters in the second layer means the model will learn 64 more abstract features from the output of the first layer.

kernel_size=5 and kernel_size=3:

Kernel size defines the size of the sliding window the convolution filter will use to scan over the input data.
For example, in the first Conv1D, kernel_size=5 means the filter will cover 5 consecutive time steps (e.g., 5 hours or 5 days, depending on your data).
This means the model looks at the previous 5 steps (or time periods) to identify a feature.
In the second Conv1D, kernel_size=3 means the filter will cover 3 consecutive time steps.
Smaller windows help capture finer-grained patterns after the broader ones are captured in the previous layer.

activation="relu":

ReLU (Rectified Linear Unit) is the activation function used after each convolution operation.
The activation function introduces non-linearity to the model, enabling it to learn more complex patterns, in each steps the price could bend more or less and capture movements that are not proportinal (not-linear) and learn this.

ReLU is defined as f(x) = max(0, x), which means it outputs the input directly if it's positive, and zero otherwise. This helps the model capture non-linear relationships in the data (like stock prices).

padding="same":

Padding ensures that the output shape of the convolution is the same as the input shape.
"same" padding means the model adds zero-padding (extra values) around the edges of the input data when necessary.
This ensures the output length is the same as the input length (e.g., you don't lose time steps at the beginning or end of the series).
Without padding, the output would shrink with each convolution step.

```python
    x = Conv1D(filters=128, kernel_size=5, activation="relu", padding="same")(inputs)
    x = Conv1D(filters=64, kernel_size=3, activation="relu", padding="same")(x)
```

Final output first two CONV Layers:

The final output is a transformed version of your stock price data, with local patterns (like price rises or falls) detected by the filters and non-linearities introduced by ReLU.
The model can then use these transformed features to predict whether the stock price will rise or fall in the future.

## Results
- The model predicts the next closing stock price for AAPL.
- Compares the predicted price with actual market data.

## License
This project is open-source and available under the MIT License.

## Contact
For further inquiries, contact:
- Email: fabrisio.ponte@gmail.com
- LinkedIn: [Your Profile](https://linkedin.com/in/fabrisio-ponte)

