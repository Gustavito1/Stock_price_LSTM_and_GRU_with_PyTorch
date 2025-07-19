# 📈 Predicción del Precio de Acciones con Deep Learning y Análisis de Sentimientos

Este proyecto predice el precio de cierre de las acciones de **Netflix (NFLX)** combinando análisis técnico, análisis de sentimientos de noticias financieras, y modelos de redes neuronales profundas (LSTM y GRU).

---

## 🧠 Tecnologías y Bibliotecas Usadas

- `Python`
- `Pandas`, `NumPy`, `Matplotlib`
- `Scikit-learn`
- `TensorFlow`, `Keras`
- `yfinance` (datos de mercado)
- `NewsAPI` (noticias)
- `FinBERT` (análisis de sentimientos financiero)
- `Transformers` de HuggingFace

---

## 📊 Datos Usados

- **Históricos del mercado:** 3 años de datos de precios de NFLX desde Yahoo Finance.
- **Noticias financieras:** Extraídas con NewsAPI, filtradas por popularidad y palabras clave relevantes (Netflix, NFLX, financial).
- **Sentimientos:** Clasificados como *positivo, negativo o neutral* usando el modelo FinBERT.

---

## ⚙️ Flujo del Proyecto

1. **Extracción de Datos Financieros y Noticias**
2. **Análisis de Sentimiento con FinBERT**
3. **Cálculo de Indicadores Técnicos**: MA, RSI, MACD, Volatilidad, etc.
4. **Preprocesamiento y Escalado**
5. **Construcción de Secuencias para Modelos Temporales**
6. **Entrenamiento y Evaluación de Modelos LSTM y GRU**
7. **Visualización de Predicciones**
8. **Cálculo de métricas**: MSE, RMSE, MAE, MAPE

---

## 🧪 Modelos Implementados

### 🔹 LSTM (Long Short-Term Memory)

- Estructura simple con capa LSTM y Dense.
- Mejora cuando se combina con sentimientos.

### 🔹 GRU (Gated Recurrent Unit)

- Variante más eficiente de RNN.
- Resultados similares a LSTM, con menos parámetros.

---

## 📈 Resultados

- Se mide el rendimiento con:
  - `MSE`, `RMSE`, `MAE`, `MAPE`
- Se obtiene una precisión aceptable (ej. MAPE < 10%)
- Se visualiza la predicción frente al precio real.

---
