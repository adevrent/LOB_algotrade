# Import necessary libraries
from matplotlib import pyplot as plt
import torch
from torch import nn
import numpy as np

import torch
import torch.nn as nn

def autopad(input_len, kernel_length, stride):
    # Computes the symmetric padding needed to preserve the input_len.
    P = ((stride - 1) * input_len - stride + kernel_length) // 2
    return P

class ConvBlock1x2_1(nn.Module):
    def __init__(self, leaky_slope=0.01):
        super().__init__()
        # Shape of a single input is (time, features) = (100, 40)
        self.m, self.n = (100, 40)
        # We add padding to keep the time dimension the same
        # First layer: 1x2@16 (stride=1x2)
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1, 2), stride=(1, 2), padding=(autopad(self.m, 1, 1), 0)), nn.LeakyReLU(leaky_slope))
        # Second layer: 4x1@16
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(4, 1), stride=(1, 1), padding=(autopad(self.m, 4, 1), 0)), nn.LeakyReLU(leaky_slope))
        # Third layer: 4x1@16
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(4, 1), stride=(1, 1), padding=(autopad(self.m, 4, 1), 0)), nn.LeakyReLU(leaky_slope))

    def forward(self, x):
        # Input shape: (batch, 1, 100, 40)
        x = self.conv1(x)  # → (batch, 16, 100, 20)
        x = self.conv2(x)  # → (batch, 16, 100, 20)
        x = self.conv3(x)  # → (batch, 16, 100, 20)
        return x
    
class ConvBlock1x2_2(nn.Module):
    def __init__(self, leaky_slope=0.01):
        super().__init__()
        # Shape of a single input is (time, features) = (100, 20)
        self.m, self.n = (100, 20)
        # We add padding to keep the time dimension the same
        # First layer: 1x2@16 (stride=1x2)
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 2), stride=(1, 2), padding=(autopad(self.m, 1, 1), 0)), nn.LeakyReLU(leaky_slope))
        # Second layer: 4x1@16
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(4, 1), stride=(1, 1), padding=(autopad(self.m, 4, 1), 0)), nn.LeakyReLU(leaky_slope))
        # Third layer: 4x1@16
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(4, 1), stride=(1, 1), padding=(autopad(self.m, 4, 1), 0)), nn.LeakyReLU(leaky_slope))

    def forward(self, x):
        # Input shape: (batch, 16, 100, 20)
        x = self.conv1(x)  # → (batch, 16, 100, 10)
        x = self.conv2(x)  # → (batch, 16, 100, 10)
        x = self.conv3(x)  # → (batch, 16, 100, 10)
        return x
    
class ConvBlock1x10(nn.Module):
    def __init__(self, leaky_slope=0.01):
        super().__init__()
        # Shape of a single input is (time, features) = (100, 10)
        self.m, self.n = (100, 10)
        # We add padding to keep the time dimension the same
        # First layer: 1x10@16 (stride=1x2)
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 10), stride=(1, 1), padding=(autopad(self.m, 1, 1), 0)), nn.LeakyReLU(leaky_slope))
        # Second layer: 4x1@16
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(4, 1), stride=(1, 1), padding=(autopad(self.m, 4, 1), 0)), nn.LeakyReLU(leaky_slope))
        # Third layer: 4x1@16
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(4, 1), stride=(1, 1), padding=(autopad(self.m, 4, 1), 0)), nn.LeakyReLU(leaky_slope))

    def forward(self, x):
        # Input shape: (batch, 16, 100, 10)
        x = self.conv1(x)  # → (batch, 16, 100, 1)
        x = self.conv2(x)  # → (batch, 16, 100, 1)
        x = self.conv3(x)  # → (batch, 16, 100, 1)
        return x
    
class ConvStack(nn.Module):
    def __init__(self, leaky_slope=0.01):
        super().__init__()
        self.block1 = ConvBlock1x2_1(leaky_slope)
        self.block2 = ConvBlock1x2_2(leaky_slope)
        self.block3 = ConvBlock1x10(leaky_slope)
    
    def forward(self, x):
        # Input shape: (batch, 1, 100, 40))
        x = self.block1(x)  # → (batch, 16, 100, 20)
        x = self.block2(x)  # → (batch, 16, 100, 10)
        x = self.block3(x)  # → (batch, 16, 100, 1)
        return x
    
class Inception(nn.Module):
    def __init__(self, leaky_slope=0.01):
        super(Inception, self).__init__()
        # Shape of a single input is (time, features) = (100, 1)
        self.m, self.n = (100, 1)
        # Branch 1
        self.b1_conv1 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 1), stride=(1, 1), padding=(autopad(self.m, 1, 1), 0)), nn.LeakyReLU(leaky_slope))
        self.b1_conv2 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 1), stride=(1, 1), padding=(autopad(self.m, 3, 1), 0)), nn.LeakyReLU(leaky_slope))
        # Branch 2
        self.b2_conv1 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 1), stride=(1, 1), padding=(autopad(self.m, 1, 1), 0)), nn.LeakyReLU(leaky_slope))
        self.b2_conv2 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(5, 1), stride=(1, 1), padding=(autopad(self.m, 5, 1), 0)), nn.LeakyReLU(leaky_slope))
        # Branch 3
        self.b3_pool = nn.Sequential(nn.MaxPool2d(kernel_size=(3, 1), stride=(1, 1), padding=(autopad(self.m, 3, 1), 0)))
        self.b3_conv = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 1), stride=(1, 1), padding=(autopad(self.m, 1, 1), 0)), nn.LeakyReLU(leaky_slope))
        
    def forward(self, x):
        # Branch 1
        b1 = self.b1_conv1(x)  # → (batch, 32, 100, 1)
        b1 = self.b1_conv2(b1)  # → (batch, 32, 100, 1)
        # Branch 2
        b2 = self.b2_conv1(x)  # → (batch, 32, 100, 1)
        b2 = self.b2_conv2(b2)  # → (batch, 32, 100, 1)
        # Branch 3
        b3 = self.b3_pool(x)  # → (batch, 16, 100, 1)
        b3 = self.b3_conv(b3)  # → (batch, 32, 100, 1)
        return torch.cat((b1, b2, b3), dim=1)  # → (batch, 96, 100, 1)
    
class LSTM(nn.Module):
    # TODO: Implement the LSTM class

##############################################################################################################

def train_NBEATS(train_loader, val_loader, model, loss_func, optimizer, device, num_epochs, feature="prices", patience=10):
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    epochs_without_improve = 0

    if feature == "prices":
        save_path_best = "best_nbeats_price_model.pth"
        save_path_final = "nbeats_price_model_weights_final.pth"
    else:
        save_path_best = "best_nbeats_logr_model.pth"
        save_path_final = "nbeats_logr_model_weights_final.pth"

    for epoch in range(num_epochs):
        # Training loop
        model.train()
        running_train_loss = 0.0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)   # shape: (batch_size, H*n)
            targets = targets.to(device)   # shape: (batch_size, H)
            
            optimizer.zero_grad()
            forecasts = model(inputs)      # forecast shape: (batch_size, H)
            loss = loss_func(forecasts, targets)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * inputs.size(0)
        
        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        
        # Validation loop
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                forecasts = model(inputs)
                loss = loss_func(forecasts, targets)
                running_val_loss += loss.item() * inputs.size(0)
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_train_loss:.4f} - Val Loss: {epoch_val_loss:.4f}")
        
        # Early stopping: check if validation loss improved
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_without_improve = 0
            # Optionally save the best model here
            torch.save(model.state_dict(), save_path_best)
            print(f"    Best model weights so far saved to {save_path_best}")
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    # Save the final model weights
    torch.save(model.state_dict(), save_path_final)
    print(f"    Final model weights saved to {save_path_final}")

    # Plot the training and validation loss curves
    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axs[0].plot(train_losses, marker='o', label="Training Loss")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("MSE Loss")
    axs[0].set_title("Training Loss Curve")
    axs[0].legend()
    axs[0].grid()

    axs[1].plot(val_losses, marker='x', color='orange', label="Validation Loss")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("MSE Loss")
    axs[1].set_title("Validation Loss Curve")
    axs[1].legend()
    axs[1].grid()

    fig.tight_layout()
    plt.show(block=True)

def test_NBEATS(test_loader, model, loss_func, device, weights_filepath):
    """
    Loads the optimum weights, evaluates the model on the test set, prints the sMAPE value,
    and plots forecast vs. true curves for 10 random samples (5x2 subplots). 
    Additionally, shows ratio difference (%) on a secondary y-axis.
    
    Parameters:
        test_loader (DataLoader): DataLoader for the test set.
        model (nn.Module): The NBEATS model instance.
        loss_func (callable): The loss function to compute sMAPE.
        device (torch.device): Device (cuda or cpu) on which to run the evaluation.
        weights_filepath (str): Path to the optimum weights file (.pth).
    """
    # Load the optimum weights
    model.load_state_dict(torch.load(weights_filepath, map_location=device))
    model.to(device)
    model.eval()
    
    forecasts_list = []
    targets_list = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            forecasts = model(inputs)
            forecasts_list.append(forecasts)
            targets_list.append(targets)
    
    # Concatenate all batches
    forecasts_all = torch.cat(forecasts_list, dim=0)  # shape: (num_samples, H)
    targets_all = torch.cat(targets_list, dim=0)      # shape: (num_samples, H)
    
    # Compute sMAPE on the entire test set
    test_smape = loss_func(forecasts_all, targets_all).item()
    print(f"Test sMAPE: {test_smape:.4f}")
    
    # Plot 10 random samples in a single figure with 5 rows, 2 columns
    num_samples = forecasts_all.shape[0]
    sample_indices = np.random.choice(num_samples, size=10, replace=False)
    
    fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(12, 15), sharex=False)
    axes = axes.flatten()  # so we can iterate easily

    for i, idx in enumerate(sample_indices):
        ax = axes[i]
        forecast_sample = forecasts_all[idx].cpu().numpy()
        target_sample = targets_all[idx].cpu().numpy()
        
        # Plot forecast vs. actual
        ax.plot(target_sample, marker='o', label="Actual")
        ax.plot(forecast_sample, marker='x', label="Forecast")
        ax.set_title(f"Sample {idx} - Forecast vs Actual", fontsize=10)
        ax.set_ylabel("Value")
        ax.grid(True)
        
        # Only label the x-axis for the bottom row (the last 2 plots)
        if i < 8:  
            ax.set_xlabel("")
        else:
            ax.set_xlabel("Time Step")

        # Create a twin axis for ratio difference
        ax2 = ax.twinx()
        # ratio_diff = 100 * (Forecast/Actual - 1)
        # Avoid division by zero by adding a small epsilon if necessary
        eps = 1e-8
        ratio_diff = 100.0 * ((forecast_sample + eps) / (target_sample + eps) - 1.0)
        ax2.plot(ratio_diff, color='tab:red', linestyle='--', label="Ratio Diff (%)")
        ax2.set_ylabel("Ratio Diff (%)", color='tab:red')
        ax2.tick_params(axis='y', labelcolor='tab:red')

        # Combine legends from both axes
        lines_1, labels_1 = ax.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")
    plt.subplots_adjust(hspace=0.4, wspace=0.3)  # increase spacing
    plt.tight_layout()
    plt.show(block=True)