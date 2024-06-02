from src.utils import StockDataset, split_dataset, train, test, preprocess
from torch.utils.data import DataLoader
from torch import nn
import torch.onnx
import torch
import glob


# Create LSTM model
class LSTM(nn.Module):
    def __init__(self, output_size, input_size, hidden_size, num_layers, seq_length):
        super().__init__()
        # Initialize internal variables
        self.output_size = output_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_length = seq_length

        # Define model layers
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,
                            num_layers=self.num_layers, batch_first=True)
        self.fc_1 = nn.Linear(self.hidden_size, 128)
        self.fc = nn.Linear(128, self.output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Processs input through model
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Get LSTM layer output
        output, (hn, cn) = self.lstm(x, (h0, c0))
        hn = hn[-1]

        # Forward LSTM output through layers
        out = self.relu(hn)
        out = self.fc_1(out)
        out = self.relu(out)
        out = self.fc(out)

        return out


if __name__ == "__main__":
    # Loads all stock data paths from the specified directory and creates a dataset
    data_directory = glob.glob("./dataset/*")
    dataset = StockDataset(data_directory, "Close", preprocessing=preprocess, normalize=True, save_scaler=False)

    # Creates training, validation, and testing datasets
    train_dataset, test_dataset = split_dataset(dataset, train_size=0.9, test_size=0.1)

    # Converts datasets into dataloaders
    batch_size = 128
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # Define device to use
    device = "cuda"
    model = LSTM(1, 4, 400, 1, 1).to(device)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Train the model
    for t in range(20):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, criterion, optimizer, batch_size, device)
        test(test_dataloader, model, criterion, device, "log")

    print("Finished training.")
    scripted_model = torch.jit.script(model)
    scripted_model.save("LSTM.pt")
    print("Model saved!")
