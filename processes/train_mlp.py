import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm


class MLPClassifier(nn.Module):
    def __init__(self, classifier_input_size=768):
        super(MLPClassifier, self).__init__()
        self.hidden_layers = nn.Sequential(
            nn.Linear(classifier_input_size, 300),  # l1
            nn.ReLU(),
            nn.LayerNorm(300),
            nn.Dropout(0.3),
            nn.Linear(300, 100),  # l2
            nn.ReLU(),
            nn.LayerNorm(100),
            nn.Dropout(0.3),
            nn.Linear(100, 50),  # l3
            nn.ReLU(),
            nn.LayerNorm(50),
        )
        self.output_layer = nn.Linear(50, 3)  # 3 classes

    def forward(self, x):
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x


def create_dataloader(x, y, batch_size=32, shuffle=True):
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader


def train_step(model, data_loader, loss_fn, optimizer, device):
    model.train()  # Set the model to training mode
    for inputs, targets in tqdm(data_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        # Zero the gradients
        optimizer.zero_grad()
        # Forward pass
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()


def validate_model(model, data_loader, loss_fn, device):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    correct_predictions = 0

    with torch.no_grad():
        for inputs, targets in tqdm(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            total_loss += loss.item()

            # For multi-class classification, use argmax to get the predicted class
            predicted_classes = outputs.argmax(dim=1)
            correct_predictions += (predicted_classes == targets).sum().item()

    average_loss = total_loss / len(data_loader)
    accuracy = correct_predictions / len(data_loader.dataset)
    return average_loss, accuracy


def train_model(x_train, y_train, x_val, y_val, model, epochs, checkpoints_dir_path,
                batch_size=32, learning_rate=1e-4,
                weight_decay=1e-2, save_checkpoints=True):
    # Move model to the appropriate compute device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    # Create training and validation data loaders
    train_loader = create_dataloader(x_train, y_train, batch_size, shuffle=True)
    val_loader = create_dataloader(x_val, y_val, batch_size, shuffle=False)

    # Define loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()  # Use CrossEntropyLoss for multi-class classification
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_losses, val_losses = [], []

    # Training loop
    for epoch in range(epochs):
        print(f'\n\n**Epoch #{epoch + 1}')
        train_step(model, train_loader, loss_fn, optimizer, device)

        print('Finished training step, checking loss on train')
        train_loss, train_accuracy = validate_model(model, train_loader, loss_fn, device)
        train_losses.append(train_loss)

        print('Checking loss on validation')
        val_loss, val_accuracy = validate_model(model, val_loader, loss_fn, device)
        val_losses.append(val_loss)

        print(
            f'Train Epoch {epoch + 1}/{epochs} Stats- Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}; Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

        # Save checkpoints
        if save_checkpoints:
            torch.save(model.state_dict(),
                       f'{checkpoints_dir_path}/epoch_{epoch + 1}_{round(val_loss, 2)}_val_loss.pth')
            np.save(f'{checkpoints_dir_path}/train_losses.npy', np.array(train_losses))
            np.save(f'{checkpoints_dir_path}/val_losses.npy', np.array(val_losses))

    print("Training complete!")


