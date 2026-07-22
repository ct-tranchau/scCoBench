import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Argument parser for command-line execution
parser = argparse.ArgumentParser(description="Autoencoder for single-cell RNA-seq imputation")
parser.add_argument("--input_path", type=str, required=True, help="Path to input CSV file")
parser.add_argument("--output_path", type=str, required=True, help="Path to save imputed output CSV file")
parser.add_argument("--loss_plot_path", type=str, required=True, help="Path to save loss plot PDF file")
parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
parser.add_argument("--mask_rate", type=float, default=0.1,
                    help="Fraction of entries to randomly mask each step (denoising objective)")
parser.add_argument("--val_split", type=float, default=0.1, help="Fraction of cells held out for validation")
parser.add_argument("--seed", type=int, default=42, help="Random seed (controls split, init, masking)")
args = parser.parse_args()

# Seed all RNGs for reproducibility across seeds.
torch.manual_seed(args.seed)
np.random.seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

# Step 1: Load and preprocess the single-cell RNA-seq data
# NOTE: input CSV is expected to be cells x genes (rows = cells, columns = genes).
print(f"Loading data from {args.input_path}...")
original_data = pd.read_csv(args.input_path, index_col=0)
if original_data.shape[0] < original_data.shape[1] // 10:
    print(f"WARNING: input has {original_data.shape[0]} rows and {original_data.shape[1]} columns; "
          "this script assumes cells x genes. Transpose if your file is genes x cells.")
transformed_data = np.log1p(original_data)  # Apply log1p transformation

# Convert data to PyTorch tensor
input_data = torch.tensor(transformed_data.values, dtype=torch.float32)

# Step 2: Train/validation split and DataLoaders
n_total = input_data.shape[0]
n_val = max(1, int(round(n_total * args.val_split)))
perm = torch.randperm(n_total, generator=torch.Generator().manual_seed(args.seed))
val_idx, train_idx = perm[:n_val], perm[n_val:]
train_data = input_data[train_idx]
val_data = input_data[val_idx]

train_loader = data.DataLoader(data.TensorDataset(train_data),
                               batch_size=args.batch_size, shuffle=True)
val_loader = data.DataLoader(data.TensorDataset(val_data), batch_size=args.batch_size)
infer_loader = data.DataLoader(data.TensorDataset(input_data), batch_size=args.batch_size)

# Step 3: Define the Autoencoder model
class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=400, latent_dim=200):
        super(Autoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, latent_dim),
            nn.LeakyReLU(0.2)
        )
        
        # ReLU on the final layer: outputs are non-negative AND can be exactly 0,
        # which matters for sparse single-cell data (most entries are true zeros).
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim),
            nn.ReLU()
        )
     
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Step 4: Define the loss function
def loss_function(x, x_hat):
    return nn.functional.mse_loss(x_hat, x, reduction='mean')

# Step 5: Initialize the model, optimizer, and device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_dim = input_data.shape[1]
model = Autoencoder(input_dim=input_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)

# Step 6: Train the model with a denoising / masked-reconstruction objective
def train(model, optimizer, train_loader, val_loader, epochs, device, mask_rate):
    train_history, val_history = [], []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        n_train_batches = 0
        for (x,) in train_loader:
            x = x.to(device)
            mask = torch.bernoulli(torch.full_like(x, 1.0 - mask_rate))
            x_corrupt = x * mask
            optimizer.zero_grad()
            x_hat = model(x_corrupt)
            loss = loss_function(x, x_hat)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_train_batches += 1

        avg_train_loss = train_loss / max(1, n_train_batches)
        train_history.append(avg_train_loss)

        model.eval()
        val_loss = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for (x,) in val_loader:
                x = x.to(device)
                x_hat = model(x)
                val_loss += loss_function(x, x_hat).item()
                n_val_batches += 1
        avg_val_loss = val_loss / max(1, n_val_batches)
        val_history.append(avg_val_loss)

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}")

    return train_history, val_history

# Train the model
print("Training Autoencoder...")
train_history, val_history = train(model, optimizer, train_loader, val_loader,
                                   args.epochs, device, args.mask_rate)

# Step 7: Plot and save loss
plt.figure(figsize=(8, 5))
plt.plot(range(1, args.epochs + 1), train_history, marker='o', linestyle='-', label='Train Loss')
plt.plot(range(1, args.epochs + 1), val_history, marker='s', linestyle='-', label='Val Loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss over Epochs")
plt.legend()
plt.grid(True)
plt.savefig(args.loss_plot_path)
print(f"Loss plot saved to {args.loss_plot_path}")

# Step 8: Reconstruct the data (batched to avoid OOM)
print("Reconstructing data...")
model.eval()
recon_chunks = []
with torch.no_grad():
    for (x,) in infer_loader:
        recon_chunks.append(model(x.to(device)).cpu().numpy())
reconstructed_data = np.concatenate(recon_chunks, axis=0)

# Inverse transformation (kept as float; downstream code can round if it needs counts)
reconstructed_data = np.expm1(reconstructed_data)
reconstructed_df = pd.DataFrame(reconstructed_data, columns=original_data.columns, index=original_data.index)

# Save output
print(f"Saving imputed data to {args.output_path}...")
reconstructed_df.to_csv(args.output_path)
print("Done!")
