import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch
import numpy as np
from torch import nn
import pandas as pd 
import torch.optim as optim
from .ann.neural_network import AnnForBinaryClassification
from defence.dataset.csv_dataset import CSVDataset
from utils.file_utils import find_all_file_names

def train_defence_model(): 
    # Folder containing CSV files
    folder_path = './../CICIoT2023/'
    batch_size = 2000

    # Create dataset instance
    dataset = CSVDataset()

    # Split dataset into training and validation sets
    train_indices, val_indices = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)
    

    # Create DataLoader for training set
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Create DataLoader for validation set
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


    #hyper parameters
    learning_rate = 0.0001
    num_epochs = 10

    csv_files = find_all_file_names()
    data_frame = pd.read_csv(csv_files[0])
    input_features = len(pd.read_csv(csv_files[0]).columns) - 1
    model = AnnForBinaryClassification(input_features)
    criterion = nn.BCELoss()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    
    for epoch in range(num_epochs):
        # Train:   
        for batch_index, (features, labels) in enumerate(train_loader):
   
            optimizer.zero_grad()
            outputs = model(features)
            outputs = torch.squeeze(outputs, -1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')
        
        model.eval()  # Set the model to evaluation mode
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                outputs = torch.squeeze(outputs, -1)
                total += labels.size(0)
                correct += (outputs == labels).sum().item()
               
        accuracy = 100 * correct / total
        print(f'Accuracy on validation set: {accuracy:.2f}%')