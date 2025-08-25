#!/usr/bin/env python
# coding: utf-8

# In[48]:


import math
import pandas as pd
import numpy as np
import requests
# import tensorflow as tf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import yfinance as yf

# Analisis de sentimientos
import requests
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Ejecucion con grafica
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Normalizar datos
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
#from keras.models import Sequential
#from keras.layers import Dense, LSTM, Dropout, GRU
#from keras import Input

#get_ipython().run_line_magic('matplotlib', 'inline')

#from tensorflow.keras.optimizers import Adam
#from tensorflow.keras.callbacks import EarlyStopping


# #### Probar con la grafica RTX 5090

# In[4]:


SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ## Declarar variables

# In[5]:


# News API
url = "https://newsapi.org/v2/everything?" 
API_KEY = "88282befa9004fc388fb57f794bdaefb"
today = datetime.now()
start_date = (today - timedelta(days=31)).strftime("%Y-%m-%d")
end_date = today.strftime("%Y-%m-%d")
# Yfinance
df_netflix_yfinance = yf.download(tickers="NFLX", start=today - timedelta(days=365*3), end=end_date, auto_adjust=True)


# In[6]:


df_netflix_yfinance.columns = [col[0] for col in df_netflix_yfinance.columns]
df_netflix_yfinance.columns


# In[7]:


# Configurar modelo FinBERT
tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
finbert = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone", torch_dtype="auto").to(device)
finbert.eval()


# In[8]:


def get_news():
    params = {
        "q": "(Netflix OR NFLX) AND financial",
        "apiKey": API_KEY,
        "language": "en",
        "from": start_date,
        "to": end_date,
        "sortBy": "popularity"
    }

    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        articles = data.get("articles", [])
        df_news = pd.DataFrame(articles)
        return df_news
    else:
        raise Exception(f"Error al obtener datos de NewsAPI: {response.status_code} - {response.text}")


# In[9]:


df_news = get_news()


# In[11]:


def create_combined_text(row):
    title = row.get('title') or ""
    description = row.get('description') or ""
    content = row.get('content') or ""
    return f"{title} {description} {content}".strip()
'''
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="tf", truncation=True, padding=True, max_length=512)
    outputs = model(inputs)
    predictions = tf.nn.softmax(outputs.logits, axis=-1).numpy()[0]
    sentiment = ['positive', 'negative', 'neutral'][predictions.argmax()]
    return sentiment'''


# In[12]:


df_news['combined_text'] = df_news.apply(create_combined_text, axis=1)
df_news


# ## NOTA: El metodo FinBERT devuelve labels en formato: 'positive', 'negative', 'neutral'; sin embargo en el repo de
# ## 'yiyanghkust/finbert-tone' devuelve '#LABEL_0: neutral; LABEL_1: positive; LABEL_2: negative'

# In[9]:


'''
#Predecir las noticias por bloques
def predict_sentiment_batch(texts, batch_size=16):
    sentiments = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        # Tokenizar lote
        encoded_inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="tf"
        )

        # Predicción
        outputs = model(encoded_inputs)
        probs = tf.nn.softmax(outputs.logits, axis=-1).numpy()

        label_list = ['positive', 'negative', 'neutral']
        batch_sentiments = [ label_list[p.argmax()] for p in probs ]
        sentiments.extend(batch_sentiments)

    return sentiments

def predict_sentiment_from_news(df_news, batch_size=16):
    # Preparar texto combinado
    df_news['combined_text'] = df_news.apply(create_combined_text, axis=1)
    df_news['publishedAt'] = pd.to_datetime(df_news['publishedAt']).dt.date

    # Predecir sentimientos en lotes (batches)
    texts = df_news['combined_text'].tolist()
    sentiments = predict_sentiment_batch(texts, batch_size=batch_size)
    df_news['sentiment'] = sentiments
    #df_news['sentiment'] = df_news['combined_text'].apply(predict_sentiment)

    # Agrupar y contar por fecha
    sentiment_counts = df_news.groupby('publishedAt')['sentiment'].value_counts().unstack().fillna(0)
    sentiment_counts.index = pd.to_datetime(sentiment_counts.index)

    # Convertir a lista de diccionarios para cada día
    sentiment_data = []
    for date, row in sentiment_counts.iterrows():
        sentiment_data.append({
            'Date': date,
            'Positive': row.get('positive', 0),
            'Negative': row.get('negative', 0),
            'Neutral': row.get('neutral', 0)
        })

    # Devolver DataFrame con sentimientos diarios
    return pd.DataFrame(sentiment_data).set_index('Date').sort_index()
'''


# In[13]:


# Usar el mapeo nativo del modelo (evita suposiciones)
id2label = finbert.config.id2label  # e.g. {0:'neutral', 1:'positive', 2:'negative'}

@torch.no_grad()
def predict_sentiment_batch(texts, batch_size=16, max_length=512):
    all_probs = []
    all_labels = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        encoded_inputs = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        encoded_inputs = {k: v.to(device) for k, v in encoded_inputs.items()}
        outputs = finbert(**encoded_inputs)
        #axis - tensorflow -- dim - torch
        #probs = torch.softmax(outputs.logits, axis=-1).numpy()  # (B, num_labels)
        probs = torch.softmax(outputs.logits, dim=-1)
        all_probs.append(probs.detach().cpu().numpy())

        argmax = probs.argmax(dim=-1).detach().cpu().numpy()
        #all_labels.extend([id2label[int(p.argmax())] for p in probs])
        all_labels.extend([id2label[int(x)] for x in argmax])
    return np.vstack(all_probs), all_labels

def predict_sentiment_from_news(df_news):
    if df_news.empty:
        idx = pd.date_range(start=pd.to_datetime(start_date), end=pd.to_datetime(end_date), freq="D")
        return pd.DataFrame(0.0, index=idx, columns=["Neutral","Positive","Negative"])

    df_news['combined_text'] = df_news.apply(create_combined_text, axis=1)
    df_news['publishedAt'] = pd.to_datetime(df_news['publishedAt'], utc=True, errors='coerce').dt.tz_convert('America/New_York').dt.date
    probs, labels = predict_sentiment_batch(df_news['combined_text'].tolist(), batch_size=16)
    cols = [id2label[i] for i in range(probs.shape[1])]
    dfp = pd.DataFrame(probs, columns=cols)
    dfp['Date'] = pd.to_datetime(df_news['publishedAt'])
    # promedio diario de probabilidades
    daily = dfp.groupby('Date')[cols].mean().sort_index()
    # renombrar a mayúsculas inicial si quieres
    daily = daily.rename(columns=str.capitalize)
    return daily


# In[14]:


sentiment_counts = predict_sentiment_from_news(df_news)
print(sentiment_counts)


# In[18]:


def calculate_technical_indicators(data):
    df = df_netflix_yfinance.copy()
    df['MA5'] = df['Close'].rolling(window=5).mean() # Se queda, funciona para mercados volatiles y no volatiles
    df['MA20'] = df['Close'].rolling(window=20).mean() # Se queda, funciona para mercados volatiles y no volatiles
    df['MA50'] = df['Close'].rolling(window=50).mean() # Se queda, funciona para mercados volatiles y no volatiles

    # MACD y Signal_Line -- Se queda, funciona para mercados volatiles y no volatiles
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # RSI - Se queda, funciona para mercados volatiles y no volatiles
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / (loss + 1e-10)
    df['RSI'] = 100 - (100 / (1 + rs))

    # Otros indicadores
    df['Returns'] = df['Close'].pct_change()
    df['Volatility'] = df['Returns'].rolling(window=20).std()
    df['Volume_MA'] = df['Volume'].rolling(window=5).mean()

    # Rellenar valores faltantes
    #df = df.fillna(method='bfill').fillna(method='ffill')
    df = df.bfill().ffill()
    return df


# In[19]:


df_final = calculate_technical_indicators(df_netflix_yfinance)
df_final


# In[20]:


# Paso 3: Fusionar con sentimiento
sentiment_counts = sentiment_counts.reindex(df_final.index, method='ffill').fillna(0)
df_final_bigdata = pd.concat([df_final, sentiment_counts], axis=1).fillna(0)
df_final_bigdata


# In[21]:


features = ['Close', 'Volume', 'MA5', 'RSI', 'MACD', 'Signal_Line', 'Returns', 'Volatility', 'Volume_MA', 'Neutral','Positive','Negative']
#features = ['Close', 'Volume', 'MA5', 'RSI', 'MACD', 'Signal_Line', 'Returns', 'Volatility', 'Volume_MA']
#features = ['Close', 'Volume', 'High', 'Low', 'Open']


# In[ ]:


feature_scaler = RobustScaler()
target_scaler = RobustScaler()


# In[23]:


# Asegurar tipo float y rellenar faltantes
data = df_final_bigdata[features].astype(float).ffill().bfill()


# In[24]:


# Dividir en train/test 70/30
split = int(len(data) * 0.8)
train_data = data.iloc[:split]
test_data = data.iloc[split:]


# In[25]:


# Aca estaba tu error, en un modelo nunca se debe entrenar los datos de testeo, solo transformarlos ya que hay data-leak
X_train_scaled = feature_scaler.fit_transform(train_data)
y_train_scaled = target_scaler.fit_transform(train_data[['Close']])

X_test_scaled = feature_scaler.transform(test_data)
y_test_scaled = target_scaler.transform(test_data[['Close']])


# In[26]:


# Crear secuencias de entrada -- Modificacion ahora hacer la secuencia en base a los datos de entrenamiento y testing
lookback = 60
X_train, y_train = [], []
for i in range(lookback, len(X_train_scaled)):
    X_train.append(X_train_scaled[i - lookback:i])
    y_train.append(y_train_scaled[i, 0])

X_train = np.array(X_train)
y_train = np.array(y_train)

# Test
X_test, y_test = [], []
for i in range(lookback, len(X_test_scaled)):
    X_test.append(X_test_scaled[i - lookback:i])
    y_test.append(y_test_scaled[i, 0])

X_test = np.array(X_test)
y_test = np.array(y_test)


# ### Aqui acaba la prueba

# In[34]:


#del model_LSTM


# ### Modelo Complejo PyTorch

# Como en este apartado usaremos PyTorch y no TensorFlow 

# ### Modelo Simplificado (TensorFlow)

# In[36]:


class SeqDataset(Dataset):
    def __init__(self, X, y):
        X = X.astype(np.float32)
        y = y.astype(np.float32)
        self.X = torch.from_numpy(X)  # (N, T, F)
        self.y = torch.from_numpy(y).unsqueeze(-1)  # (N, 1)
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

BATCH_SIZE = 64

train_ds = SeqDataset(X_train, y_train)
test_ds  = SeqDataset(X_test,  y_test)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
val_loader   = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, drop_last=False)


# In[37]:


class LSTMRegressor(nn.Module):
    def __init__(self, input_dim, hidden=64, dense=32):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, batch_first=True)
        self.dropout1 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(hidden, dense)
        self.act = nn.ReLU()
        self.dropout2 = nn.Dropout(0.1)
        self.out = nn.Linear(dense, 1)
    def forward(self, x):
        # x: (B, T, F)
        out, _ = self.lstm(x)
        out = out[:, -1, :]            # última salida temporal
        out = self.dropout1(out)
        out = self.fc1(out)
        out = self.act(out)
        out = self.dropout2(out)
        return self.out(out)

class GRURegressor(nn.Module):
    def __init__(self, input_dim, hidden=64, dense=32):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden, batch_first=True)
        self.dropout1 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(hidden, dense)
        self.act = nn.ReLU()
        self.dropout2 = nn.Dropout(0.1)
        self.out = nn.Linear(dense, 1)
    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.dropout1(out)
        out = self.fc1(out)
        out = self.act(out)
        out = self.dropout2(out)
        return self.out(out)


# In[38]:


INPUT_DIM = X_train.shape[2]
lr = 1e-3


# In[39]:


model = LSTMRegressor(INPUT_DIM).to(device)
# model = GRURegressor(INPUT_DIM).to(device)


# In[40]:


criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


# In[41]:


EPOCHS = 100
PATIENCE = 20
best_val = float("inf")
pat = 0
history = {"train": [], "val": []}

for epoch in range(1, EPOCHS+1):
    # train
    model.train()
    train_losses = []
    for Xb, yb in train_loader:
        Xb = Xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad()
        preds = model(Xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
    train_loss = float(np.mean(train_losses))

    # val
    model.eval()
    val_losses = []
    with torch.no_grad():
        for Xb, yb in val_loader:
            Xb = Xb.to(device)
            yb = yb.to(device)
            preds = model(Xb)
            loss = criterion(preds, yb)
            val_losses.append(loss.item())
    val_loss = float(np.mean(val_losses))

    history["train"].append(train_loss)
    history["val"].append(val_loss)

    print(f"Epoch {epoch:03d} | train {train_loss:.6f} | val {val_loss:.6f}")

    if val_loss < best_val - 1e-8:
        best_val = val_loss
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        pat = 0
    else:
        pat += 1
        if pat >= PATIENCE:
            print("Early stopping!")
            break


# In[42]:


# Restaurar mejor estado
model.load_state_dict({k: v.to(device) for k, v in best_state.items()})


# In[43]:


model.eval()
with torch.no_grad():
    preds_scaled = []
    for Xb, _ in val_loader:
        Xb = Xb.to(device)
        out = model(Xb).detach().cpu().numpy()
        preds_scaled.append(out)
preds_scaled = np.vstack(preds_scaled)


# In[49]:


# Invertir escala
preds_actual = target_scaler.inverse_transform(preds_scaled)
y_test_actual = target_scaler.inverse_transform(y_test.reshape(-1, 1))

mse = mean_squared_error(y_test_actual, preds_actual)
mae = mean_absolute_error(y_test_actual, preds_actual)
rmse = math.sqrt(mse)
mape = np.mean(np.abs((y_test_actual - preds_actual) / (y_test_actual + 1e-12))) * 100

print("\nModel Performance:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Mean Absolute Percentage Error: {mape:.2f}%")


# In[33]:


model_LSTM = Sequential([
    Input(shape=(X_train.shape[1], X_train.shape[2])),

    LSTM(64, return_sequences=False),
    Dropout(0.2),

    Dense(32, activation='relu'),
    Dropout(0.1),

    Dense(1) 
])

optimizer = Adam(learning_rate=0.001)

model_LSTM.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae', 'mse'])


# In[34]:


early_stop = EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True,
    verbose=1
)


# In[35]:


hist = model_LSTM.fit(X_train, y_train, epochs=100, batch_size=16, validation_split=0.1, callbacks=[early_stop], verbose=1)


# In[36]:


# Predecir y evaluar
predicted_price = model_LSTM.predict(X_test)


# In[37]:


predictions_actual = target_scaler.inverse_transform(predicted_price)
y_test_actual = target_scaler.inverse_transform(y_test.reshape(-1, 1))


# In[38]:


# Calcular métricas de rendimiento
mse = np.mean((predictions_actual - y_test_actual) ** 2)
mae = np.mean(np.abs(predictions_actual - y_test_actual))
rmse = np.sqrt(mse)
mape = np.mean(np.abs((y_test_actual - predictions_actual) / y_test_actual)) * 100


# In[39]:


# Imprimir métricas
print(f'\nModel Performance:')
print(f'Mean Squared Error: {mse:.2f}')
print(f'Root Mean Squared Error: {rmse:.2f}')
print(f'Mean Absolute Error: {mae:.2f}')
print(f'Mean Absolute Percentage Error: {mape:.2f}%')


# In[40]:


# Crear un DataFrame para facilitar el gráfico
valid = pd.DataFrame()
valid['Real'] = y_test_actual.ravel()
valid['Predicción'] = predictions_actual.ravel()

# Graficar
plt.figure(figsize=(14, 6))
plt.plot(valid['Real'].values, label='Precio Real')
plt.plot(valid['Predicción'].values, label='Precio Predicho')
plt.title('Predicción de precios con LSTM + Sentimiento (Test Set)')
plt.xlabel('Tiempo')
plt.ylabel('Precio de Cierre (USD)')
plt.legend()
plt.show()


# #### Podemos concluir segun internet:
#     - MAPE < 10% = esta en un rango aceptable
#     - MAPE < 9% = esta en un rango aceptable
#     - RMSE 64 = Si es se predice si sube o baja la accion maximo habra una diferencia de 64$ aunque en algunos casos puede variar

# ## Vamos a probar con un modelo GRU

# In[41]:


#features = ['Close', 'Volume', 'High', 'Low', 'Open']
features = ['Close', 'Volume', 'MA5', 'RSI', 'MACD', 'Signal_Line', 'Returns', 'Volatility', 'Volume_MA', 'Negative','Neutral','Positive']
#features = ['Close', 'Volume', 'MA5', 'RSI', 'MACD', 'Signal_Line', 'Returns', 'Volatility', 'Volume_MA']


# In[42]:


data = df_final_bigdata[features].astype(float).ffill().bfill()


# In[43]:


# Escalado
split = int(len(data) * 0.8)
train = data.iloc[:split]
test  = data.iloc[split:]

# 2) scalers entrenados SOLO con train
feature_scaler = RobustScaler()
target_scaler  = RobustScaler()

X_train_scaled = feature_scaler.fit_transform(train)
X_test_scaled  = feature_scaler.transform(test)

y_train_scaled = target_scaler.fit_transform(train[['Close']])
y_test_scaled  = target_scaler.transform(test[['Close']])


# In[44]:


# Secuencias con lookback
lookback = 60

def create_sequences(X, y, lookback):
    Xs, ys = [], []
    for i in range(lookback, len(X)):
        Xs.append(X[i-lookback:i])
        ys.append(y[i, 0])
    return np.array(Xs), np.array(ys)

X_train, y_train = create_sequences(X_train_scaled, y_train_scaled, lookback)
X_test, y_test = create_sequences(X_test_scaled, y_test_scaled, lookback)


# In[61]:


#del model_GRU


# In[45]:


# Modelo GRU
model_GRU = Sequential([
    Input(shape=(X_train.shape[1], X_train.shape[2])),
    GRU(units=64, return_sequences=False),
    Dropout(0.2),
    Dense(units=32, activation='relu'),
    Dropout(0.1),
    Dense(units=1)
])

optimizer = Adam(learning_rate=0.001)
model_GRU.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae', 'mse'])

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True,
    verbose=1
)


# In[46]:


hist = model_GRU.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=100,
    batch_size=16,
    callbacks=[early_stop],
    verbose=2
)


# In[47]:


predicted_price = model_GRU.predict(X_test)
predictions_actual = target_scaler.inverse_transform(predicted_price)
y_test_actual = target_scaler.inverse_transform(y_test.reshape(-1, 1))


# In[48]:


# Métricas
mse = np.mean((predictions_actual - y_test_actual) ** 2)
mae = np.mean(np.abs(predictions_actual - y_test_actual))
rmse = np.sqrt(mse)
mape = np.mean(np.abs((y_test_actual - predictions_actual) / y_test_actual)) * 100
print(f"\nModel Performance:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Mean Absolute Percentage Error: {mape:.2f}%")


# In[49]:


# Crear un DataFrame para facilitar el gráfico
valid = pd.DataFrame()
valid['Real'] = y_test_actual.ravel()
valid['Predicción'] = predictions_actual.ravel()

# Graficar
plt.figure(figsize=(14, 6))
plt.plot(valid['Real'].values, label='Precio Real')
plt.plot(valid['Predicción'].values, label='Precio Predicho')
plt.title('Predicción de precios con LSTM + Sentimiento (Test Set)')
plt.xlabel('Tiempo')
plt.ylabel('Precio de Cierre (USD)')
plt.legend()
plt.show()

