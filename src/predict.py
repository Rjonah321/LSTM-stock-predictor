import matplotlib.pyplot as plt
import pandas as pd
import pickle
import torch
import glob

# Loads the model and sets it to evaluation mode
model = torch.jit.load("LSTM.pt")
model.to("cuda")
model.eval()

# Loads data from the specified directory
data_directory = glob.glob("../dataset/*")
stock_data = []
for file in data_directory:
    df = pd.read_csv(file, index_col="Date", parse_dates=True)
    stock_data.append(df)

# Creates dataframe and drops unnecessary data
df = pd.concat(stock_data)
df = df.drop("Dividends", axis=1).drop("Stock Splits", axis=1)

# Drops NaN values
if df.isna:
    df.dropna(inplace=True)

# Gets the last 500 training examples
df = df[len(df) - 500:]

# Creates X and y variables
X = df.drop("Close", axis=1)
y = df[["Close"]]

# Opens the saved scalers and applies them to the X and y
with open("scalers.pkl", "rb") as f:
    scalers = pickle.load(f)

scaler_X = scalers["scaler_X"]
scaler_y = scalers["scaler_y"]

X_scaled = scaler_X.transform(X)
y_scaled = scaler_y.transform(y)

# Creates tensors to make predictions
X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to("cuda").unsqueeze(1)
y_tensor = torch.tensor(y_scaled, dtype=torch.float32).to("cuda")

# Makes predictions on the data
with torch.no_grad():
    y_pred_scaled = model(X_tensor)

# Moves the predictions to the cpu, converts it to a NumPy array, and transforms it back to its original values
y_pred = y_pred_scaled.cpu().numpy()
y_pred = scaler_y.inverse_transform(y_pred)

# Creates a range of dates to display on the graph
date_range = pd.date_range(start=df.index[0], periods=len(df))

# Creates a graph showing the model's predictions vs. the target
plt.figure(figsize=(12, 6))
plt.plot(date_range, y, label="Actual", color="blue", alpha=0.7)
plt.plot(date_range, y_pred, label="Predicted", color="red", alpha=0.7)
plt.title("Actual vs. Predicted Stock Prices")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()

# Creates a graph showing the residuals
residuals = y - y_pred
plt.figure(figsize=(12, 6))
plt.plot(date_range, residuals, color="green", alpha=0.7)
plt.title("Residuals")
plt.xlabel("Date")
plt.ylabel("Error")
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
