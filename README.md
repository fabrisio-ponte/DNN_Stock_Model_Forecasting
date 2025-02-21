# Stock Modeling with sentiment analysis (FinBERT) and deep learning hybrid model (CCN, LSTM, Transformer)

## Overview
This project predicts a single stock's price using a hybrid deep learning model that combines CNN, LSTM, and Transformer architectures. It integrates sentiment analysis from financial news to enhance prediction accuracy.

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

### 2. Fetch Financial Statements

- yf.Ticker(ticker): Initializes a Yahoo Finance object for the given stock ticker.
- .financials.T: Retrieves the income statement, transposed for easier row-wise access.
- .balance_sheet.T: Retrieves the balance sheet, also transposed.
- .cashflow.T: Retrieves the cash flow statement, transposed.

```python
stock = yf.Ticker(ticker)
income_stmt = stock.financials.T  # Income Statement
balance_sheet = stock.balance_sheet.T  # Balance Sheet
cash_flow = stock.cashflow.T  # Cash Flow Statement
```
The function tries to extract specific key financial metrics:

- Income Statement: 'Total Revenue', 'Net Income'
- Balance Sheet: 'Total Assets', 'Total Liabilities'
- Cash Flow Statement: 'Operating Cash Flow'
- Concatenates the extracted data into a single DataFrame.

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
#### Sentiment Mapping:

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

#### Add Sentiment Scores:

np.random.choice(sentiment_scores, size=len(stock_df)) randomly selects sentiment scores from the provided sentiment_scores list and assigns them to the new column. This is done randomly, which may not be ideal if sentiment data is meant to be linked with the stock data based on news articles (e.g., each news headline should correspond to specific stock data).

Potential Issue: The sentiment scores are being assigned randomly, which might not reflect actual correlations between the stock price movements and the sentiment of news articles.

```python
stock_df["Sentiment"] = np.random.choice(sentiment_scores, size=len(stock_df))
```

#### Align Financial Data to Stock Data Dates:

reindex(): This re-aligns the financial data to the dates of the stock data (stock_df).
method='ffill': The forward fill method is used, meaning that if a financial value for a specific date is missing, the function fills it with the most recent available value.
This ensures that financial data is properly aligned with the dates in the stock price data.

```python
financial_data = financial_data.reindex(stock_df.index, method='ffill')
```

#### Merge Stock Data with Financial Data:

pd.concat() merges the stock data (stock_df) with the financial data along columns (axis=1). After this step, the stock_df will include additional financial features like revenue, net income, etc., in addition to the stock price and sentiment.

```python
stock_df = pd.concat([stock_df, financial_data], axis=1)
```

### 4.1 Sequence Creation:

#### Function Arguments:

data: This is the dataset containing stock features and additional data (e.g., stock price, sentiment, financial indicators). It’s expected to be a 2D array or DataFrame with shape (num_samples, num_features).
seq_length=30: This is the length of the sequence. The default is 30, meaning the model will use the previous 30 time steps to predict the next stock price.

```python
def create_sequences(data, seq_length=30):
```

 #### Loop to Create Sequences:
 
 Looping through the data: The loop iterates through the dataset, creating sequences of the previous seq_length time steps (30 by default).
 
data[i:i+seq_length]: This extracts a sequence of seq_length time steps from the data, starting at index i and ending at index i+seq_length. This forms the input sequence to predict the future price.

data[i+seq_length, 3]: This appends the label (target value) to the labels list. The label is the closing price of the next time step (i.e., i+seq_length). The 3 indicates that you’re selecting the 4th column (index 3) in the data, which corresponds to the Closing price.

 ```python
for i in range(len(data) - seq_length):
        sequences.append(data[i:i+seq_length])
        labels.append(data[i+seq_length, 3])
```

#### Return sequences of labels:

sequences: A list of sequences of length seq_length, where each sequence is a slice of the historical data.
labels: A list of corresponding labels (closing prices) for each sequence, which the model will predict.

The sequences and labels are converted into NumPy arrays before being returned. This is because NumPy arrays are typically used in machine learning models for efficient computation.

```python
return np.array(sequences), np.array(labels)
```
## 5. Model Architecture
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

 #### Input Layer:

 seq_length: Length of the input sequence (e.g., 30 time steps).
feature_dim: Number of features (columns) in the dataset. This might include stock prices, sentiment scores, technical indicators, etc.

The input shape is (seq_length, feature_dim), indicating sequences of seq_length time steps, with feature_dim features at each time step.

```python
 inputs = Input(shape=(seq_length, feature_dim))
```

## 5.1 Convolutional Neural Network Layers:

Conv1D stands for 1-dimensional convolution.
Convolution layers are used to detect patterns (features) in the data by applying filters (also called kernels) over the input.

In your case, you are working with time-series data (e.g., stock prices over time), and Conv1D helps identify patterns within a window of consecutive time steps.

## 5.1.1 Parameters in Detail:

#### Inputs:

inputs represents the input layer of the model, where the data is fed into the network.
Input(shape=(seq_length, feature_dim)) specifies the shape of the input data.
seq_length: The number of time steps in each sequence (e.g., 30 time steps).
feature_dim: The number of features (e.g., stock prices, technical indicators, sentiment scores) at each time step in the sequence.
For example, if you have 30 time steps and 5 features (e.g., Open, Close, Volume, Sentiment, RSI), the input shape will be (30, 5).

So, inputs is a tensor (data structure) representing the input data that will flow through the model.

```python
inputs = Input(shape=(seq_length, feature_dim))
```

#### Convolutional Layers CONV1D

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
#### Final output first two CONV Layers:

The final output is a transformed version of your stock price data, with local patterns (like price rises or falls) detected by the filters and non-linearities introduced by ReLU.
The model can then use these transformed features to predict whether the stock price will rise or fall in the future.

## What is MultiHeadAttention?

MultiHeadAttention is a mechanism used in transformers (like the one we're using in here) to allow the model to focus on different parts of the input sequence. It does this through multiple "attention heads," each of which learns to focus on different aspects or relationships in the data.

## Components of MultiHeadAttention:

num_heads=8:

This means that the model is using 8 separate attention heads.
Each attention head will focus on a different aspect of the input. For example, one head might focus on short-term trends in the stock prices, another might focus on long-term trends, another might look at relationships between price and volume, etc.
These heads operate in parallel, learning different representations of the input sequence.

key_dim=feature_dim:

key_dim refers to the dimension of the query, key, and value vectors.

feature_dim here represents the number of features in your input data (e.g., number of features in the stock price data such as Open, High, Low, Close, Volume).
Essentially, this defines the size of the vectors that the attention mechanism operates on.

(x, x):

This is the key part where we’re using self-attention.
Both the query (Q) and key (K) are set to x (the input data). This means that the attention mechanism will look at the relationships within the input data itself. The value (V) is also taken from x.
In other words, you're saying: "I want to compute how each part of the sequence (x) relates to every other part of the sequence (x) in terms of their features."

## How Does it Work?

#### Input Data (x):

Your input x is stock data over time, where each time step has several features (e.g., Open, High, Low, Close, Volume).
Suppose x is a sequence of 5 time steps, and each time step has feature_dim features (e.g., Open, High, Low, Close, Volume).

#### Query, Key, and Value:

For each attention head, the input x is used to create three representations: queries, keys, and values. These are done using learned linear transformations (weights) specific to each attention head.

The MultiHeadAttention layer will split the queries, keys, and values into 8 different "heads" (since num_heads=8), and each head will learn a different transformation of x in its own way.

## Attention Mechanism:

Each attention head computes the attention scores for the query with respect to the key (from the same input x).
The attention score is essentially a measure of how much focus should be placed on a particular time step (in the input x) when making predictions.

#### Dot Product

For each time step in x, the query vector for that time step is compared (via a dot product) with the key vectors of all the time steps.
The result of this comparison is then used to compute the attention weights—i.e., how much attention each time step should pay to all other time steps in the sequence.

#### Combining the Outputs

After computing the attention weights for each head, the weighted sum of the values is calculated.
Each attention head outputs its own weighted sum of the values.

#### Concatenation

Once all the heads have computed their weighted sums, these outputs are concatenated together.
After concatenation, a final linear transformation is applied to combine all the heads' outputs into a single representation, which will be used for further processing.


## Summary of What's Happening:

MultiHeadAttention(num_heads=8, key_dim=feature_dim): This defines a multi-head attention layer with 8 attention heads. Each attention head learns a different "view" or projection of the data.

(x, x): This means we are using the same input x as both the queries (Q) and the keys (K). This is a self-attention mechanism, where the model learns which parts of the input are important relative to each other.
The attention heads each learn to focus on different parts of the input sequence, and their outputs are combined into a richer representation that the model uses for prediction.


### In a Real-World Scenario (Example):
Let's say you're analyzing stock prices. The input x could be the stock price data over time, and each feature could represent the price at different points (e.g., Open, Close, High, Low, Volume).

The attention heads might learn to focus on:

Head 1: Short-term trends (e.g., looking at recent fluctuations in price).

Head 2: Long-term trends (e.g., looking at broader price movements over several days).

Head 3: Relationships between price and volume (e.g., how volume correlates with price movements).

```python
attn_output = MultiHeadAttention(num_heads=8, key_dim=feature_dim)(x, x)
```
## 5.2 Layer Normalization

attn_output + x:

This is called a residual connection or skip connection. The idea is to add the output of the multi-head attention (attn_output) with the input (x), which helps the model preserve information from earlier layers.
Why add them? This helps the network avoid vanishing gradients during training, as the gradients can flow more easily through the network without being diminished.
LayerNormalization():

After the residual connection, the combined output (attn_output + x) is passed through LayerNormalization.

Layer normalization normalizes the output across the features of each time step. This means that for each time step, the mean and variance of the features are adjusted to have a zero mean and unit variance. It ensures that the output of the residual connection is on a similar scale, preventing issues like exploding or vanishing gradients and improving the stability of training.

Why normalize here? The attention mechanism can introduce very large or very small values due to the self-attention computation. Layer normalization ensures that these values are standardized, which helps the model to train more efficiently and to converge faster.

```python
    x = LayerNormalization()(attn_output + x)
```
### 5.3 Long-Short Term Memory (LSTM) Layer Core Idea:

An LSTM contains special components that control the flow of information over time. The key components are:

Forget Gate: Decides which information to discard from the cell state.
Input Gate: Determines which new information to store in the cell state.
Cell State: Carries the long-term memory of the network across time steps.
Output Gate: Decides which part of the cell state should be output as the current hidden state.

## Forget Gate:
The forget gate controls how much of the previous memory should be forgotten or kept. It looks at the current input and the previous hidden state to decide this.

Mathematical function: The forget gate computes a value between 0 and 1 (using a sigmoid function) for each piece of the cell state. A value close to 0 means "forget," while a value close to 1 means "keep."

Intuition: It decides whether to forget or remember certain information in the cell state based on the current input and past hidden state. For example, in stock prices, the forget gate might decide to discard information from a few days ago if it's no longer relevant for predicting future prices.

## Input Gate:
The input gate decides what new information will be stored in the cell state. This new information is based on the current input and the previous hidden state.

Mathematical function: The input gate uses a sigmoid activation to decide which parts of the new information should be updated, and a tanh function to generate new candidate values.

Intuition: The input gate evaluates the relevance of the new input (e.g., today's stock price, volume) and adjusts the cell state accordingly, storing the most relevant information.

## Cell State:
The cell state acts as the "memory" of the LSTM. It carries relevant information through the sequence and is updated by the forget and input gates.

Mathematical function: The cell state is updated by combining the information from the forget gate and the input gate:

Intuition: The cell state is continuously updated over time. The forget gate decides how much of the past memory to keep, and the input gate decides how much new information to store. This allows the network to "remember" useful information over long periods, which is crucial for sequences where past events (e.g., stock prices) have an influence on future predictions.

## Output Gate:
The output gate decides which part of the cell state to output as the current hidden state, which is passed to the next time step and used for predictions.

Mathematical function: The output gate looks at the current input and the previous hidden state to determine which part of the cell state should be output. The final output (hidden state) is a combination of the current cell state and the output gate:

Intuition: This gate ensures that only the most relevant information from the cell state is passed on to the next time step. This is the part of the memory that is used for making predictions or decisions. For example, the model might output information about whether the stock price will increase or decrease based on what it "remembers" from the past.

## How LSTMs Help in Stock Prediction (Example)

In stock price prediction, the LSTM learns patterns in the stock data (price movements, volume, etc.) over time. For example:

Forget Gate: It might decide to forget past price data that’s no longer useful for predicting future prices.
Input Gate: It might focus on relevant patterns from recent stock prices or volume changes.
Cell State: The LSTM’s memory will store long-term trends, like how the stock generally increases or decreases based on specific patterns.
Output Gate: The final output will be based on the learned information from all the gates and used to predict the stock’s future behavior (e.g., the closing price for the next day).

```python
x = LSTM(64, return_sequences=True)(x)
    x = LSTM(32)(x)
```

## 5.4 Fully connected layers

Dense(64): This defines a fully connected layer (also called a Dense layer) with 64 units (neurons).

A fully connected layer means that each neuron in this layer is connected to every neuron in the previous layer. It takes all the features from the previous layer and learns a weighted combination of them.
The 64 units specify that the layer will output a vector with 64 values. The more units you have, the more complex patterns the layer can potentially learn. However, too many units can lead to overfitting (where the model learns the training data too well and doesn't generalize to new data).
activation="relu": The ReLU (Rectified Linear Unit) activation function is used here. It introduces non-linearity into the model, allowing it to learn complex relationships.

ReLU takes any negative value and converts it to 0, while positive values remain unchanged.
The activation function helps the model learn patterns that are not simply linear, which is important for tasks like stock price prediction, where the relationships between input features (e.g., historical prices) and the target (e.g., future prices) are often complex.

```python
x = Dense(64, activation="relu")(x)
```

## 5.4.1 Regularization Factor

Dropout(0.3): This is a dropout layer, and it's used to regularize the model. During training, it randomly "drops" (sets to zero) 30% of the neurons in this layer on each forward pass.
This means that during each iteration, the model will ignore 30% of the neurons in this layer, forcing the network to learn more robust features and not rely too heavily on specific neurons.
The number 0.3 means that 30% of the neurons are dropped, and the remaining 70% are used in training.
Why Use Dropout?
Prevent overfitting: Dropout helps the model generalize better to unseen data by preventing it from memorizing specific patterns in the training data.
Without dropout, the model might rely too heavily on certain features, leading to overfitting. Dropout makes the model more robust, so it learns patterns that are more general and applicable to new data.
In the case of stock prediction, using dropout ensures that the model doesn't memorize only the training data and can generalize to new data points, such as future stock prices.

```python
x = Dropout(0.3)(x)
```

## 5.5 Output Layer Final Prediction

Dense(1): This is another fully connected (dense) layer, but with only 1 unit.

The 1 unit means this layer will output a single value (which is typical in regression problems where you want to predict a continuous value like stock price).
This output represents the final prediction made by the model.
activation="linear": The activation function is set to linear, meaning there is no non-linearity applied here.

Linear activation means that the output is directly proportional to the input (i.e., there’s no squashing or transformation). This is appropriate for regression tasks like stock price prediction, where you want the model to output a continuous value without any bounds (e.g., no upper or lower limit).
For example, if you're predicting a stock price, this layer will output the actual predicted price (e.g., $150).

```python
    outputs = Dense(1, activation="linear")(x)
```

### Why Are These Layers Important in Stock Price Prediction?

Dense layers allow the model to integrate the complex features learned by earlier layers (like LSTMs or attention mechanisms) and make predictions based on those features.
Dropout helps prevent the model from overfitting, ensuring it generalizes well to new data (which is crucial in stock market prediction because the data is highly variable).
The output layer provides the final prediction of the stock price (or any other continuous value you are predicting).

## Summary of LSTM

LSTM neurons "remember" by using the cell state to store relevant information, which is updated by the forget and input gates.
The output gate controls what part of the memory is passed on to the next time step or used for predictions.
The LSTM learns by adjusting the weights of these gates through backpropagation, ensuring that it can better capture temporal dependencies and improve prediction accuracy over time.


## 5. 6 Model Optimization: Adaptive Moment Estimation (Adam)

## Model Definition

Model(inputs, outputs): This defines the model architecture by specifying the input layer and output layer.

inputs: The input to the model, which is defined at the beginning of your code (in this case, inputs = Input(shape=(seq_length, feature_dim))). This specifies the shape of the data that the model will receive. For example, if you're feeding in stock data, it might have dimensions like (30 days, 5 features), where 30 is the number of time steps (days), and 5 is the number of features (e.g., Open, High, Low, Close, Volume).
outputs: The final output of the model, which is defined in the last part of the model (in this case, outputs = Dense(1, activation="linear")(x)). This specifies the prediction the model will make—whether that’s a stock price, classification label, or other value.
Essentially, this line connects all the layers and defines how the model will flow from the input to the output.

```python
model = Model(inputs, outputs)
```
## Model Compilation

This line compiles the model, preparing it for training. Let's go over the individual components:

optimizer="adam": This specifies the optimizer to use for updating the weights during training.

Adam (Adaptive Moment Estimation) is a popular optimizer that combines the benefits of AdaGrad and RMSProp. It adjusts the learning rate based on the momentums of the gradient, which helps the model converge faster and can avoid oscillations or overshooting.
In the case of stock price prediction, Adam is often used because it adapts the learning rate and works well with noisy, high-dimensional data.
loss="mse": This specifies the loss function, which measures how far the model's predictions are from the true values.

MSE (Mean Squared Error) is a commonly used loss function for regression tasks (like predicting stock prices). It calculates the average of the squared differences between the predicted values and the actual values.
For example, if the model predicts a stock price of $100, but the actual price is $105, the MSE would penalize this prediction with a larger error compared to smaller errors, which helps the model learn to minimize prediction errors.
metrics=["mae"]: This specifies the metrics that you want to track during training.

MAE (Mean Absolute Error) is another metric that measures the average absolute difference between the predicted and actual values. Unlike MSE, MAE is less sensitive to outliers.
Both MSE and MAE are useful for regression tasks, and MAE gives you a more interpretable measure of prediction error (in the same units as the stock price, for example).

```python
model.compile(optimizer="adam", loss="mse", metrics=["mae"])
```

## 5.7 Returning Model for Train or Test

This returns the compiled model from the function so that you can use it for training, evaluation, or prediction outside the function.

Why return the model? Returning the model allows you to use it in other parts of your code for training (model.fit()), evaluation (model.evaluate()), or making predictions (model.predict()).

```python
return model
```


## Results
- The model predicts the next closing stock price for AAPL.
- Compares the predicted price with actual market data.

## License
This project is open-source and available under the MIT license.

## Contact
For further inquiries, contact:
- Email: fabrisio.ponte@gmail.com
- LinkedIn: https://linkedin.com/in/fabrisio-ponte)](https://www.linkedin.com/in/fabrisio-ponte-vela/

