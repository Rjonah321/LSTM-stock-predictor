from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import Dataset, random_split
import pandas as pd
import numpy as np
import pickle
import torch
import time


# Creates a Dataset class
class StockDataset(Dataset):
    def __init__(self, file_paths, target_col, preprocessing=None, normalize=True, save_scaler=True):
        # Initializes a list to store processed datasets and the target column
        self.stock_data = []
        self.target_col = target_col

        # Reads all files in the specified file paths and appends them to the stock_data list
        for file in file_paths:
            df = pd.read_csv(file, index_col="Date", parse_dates=True)
            self.stock_data.append(df)

        # Concatenates the list into a DataFrame and preprocesses it
        self.df = pd.concat(self.stock_data)
        if preprocessing:
            self.df = preprocessing(self.df)

        # Creates X and y variables
        self.X = self.df.drop(self.target_col, axis=1)
        self.y = self.df[[self.target_col]]

        # Normalizes the data
        if normalize:
            self.scaler_X = StandardScaler()
            self.scaler_y = MinMaxScaler()
            self.scaler_X.fit(self.X)
            self.scaler_y.fit(self.y)
            self.X = self.scaler_X.transform(self.X)
            self.y = self.scaler_y.transform(self.y)

            # Saves the scalers for later interfacing
            if save_scaler:
                with open("scalers.pkl", "wb") as f:
                    pickle.dump({"scaler_X": self.scaler_X, "scaler_y": self.scaler_y}, f)

        else:
            self.X = np.array(self.X)
            self.y = np.array(self.y)

        # Turns X and y into tensors for training
        self.X = torch.Tensor(self.X)
        self.y = torch.Tensor(self.y)

        # Reshapes X and y so that they are compatible with the LSTM model
        self.X = torch.reshape(self.X, (self.X.shape[0], 1, self.X.shape[1]))
        self.y = torch.reshape(self.y, (self.y.shape[0], 1, self.y.shape[1]))

    def __len__(self):
        # Gets the length of the DataFrame
        return len(self.df)

    def __getitem__(self, index):
        # Returns X and y values
        return self.X[index], self.y[index]


# Preprocesses the data
def preprocess(df):
    # Drops undesired features
    df = df.drop("Dividends", axis=1).drop("Stock Splits", axis=1)

    # Removes all NaN values
    if df.isna:
        df.dropna(inplace=True)

    return df


# Splits datasets into training testing, and validation sets
def split_dataset(dataset, create_validation_set=False, train_size=0.8, test_size=0.2, validation_size=0.0):
    # Gets the length of the dataset
    data_size = len(dataset)
    if len(dataset) == 0:
        raise ValueError("Dataset must have more than one value.")

    if create_validation_set:
        # Ensures that the dataset will split correctly
        if train_size + validation_size + test_size != 1.0:
            raise ValueError("The sum of train_size, validation_size, and test_size must be equal to 1.0.")

        # Gets the desired length of the datasets
        train_size = int(train_size * data_size)
        val_size = int(validation_size * data_size)
        test_size = data_size - train_size - val_size

        # Splits the datasets and returns them
        train_data, val_data, test_data = random_split(dataset, [train_size, val_size, test_size])
        return train_data, val_data, test_data

    else:
        # Ensures that the dataset will split correctly
        if train_size + test_size != 1.0:
            raise ValueError("The sum of train_size and test_size must be equal to 1.0.")

        # Gets the desired length of the datasets
        train_size = int(train_size * data_size)
        test_size = data_size - train_size

        # Splits the datasets and returns them
        train_data, test_data = random_split(dataset, [train_size, test_size])
        return train_data, test_data


# Training logic for the LSTM
def train(dataloader, model, criterion, optimizer, batch_size, device):
    # Gets the size of the dataset
    size = len(dataloader.dataset)
    # Turns the model to training mode
    model.train()
    start_time = time.time()

    for batch, (x, y) in enumerate(dataloader):
        # Trains the model on individual batches
        x, y = x.to(device), y.view(-1, 1).to(device)
        pred = model(x)

        optimizer.zero_grad()

        loss = criterion(pred, y)

        loss.backward()
        optimizer.step()

        if batch % 1000 == 0:
            loss, current = loss.item(), batch * batch_size + len(x)
            print(f"Loss: {loss:.4e} [{current}/{size}]")

    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)

    print(f"Training time: {minutes} minutes and {seconds} seconds")


# Testing logic for the LSTM
def test(dataloader, model, loss_fn, device, filename):
    # Gets the number of batches in the dataloader
    num_batches = len(dataloader)
    # Sets the model to evaluation mode
    model.eval()
    test_loss = 0

    # Compares the model's predictions to the target
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.view(-1, 1).to(device)
            pred = model(x)
            test_loss += loss_fn(pred, y).item()

    test_loss /= num_batches
    print(f"Test Error: Avg loss: {test_loss:.4e} \n")

    with open(filename, "a") as f:
        f.write(f"Test Error: Avg loss: {test_loss:.4e}\n")
