import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---- Argument Parser ----
parser = argparse.ArgumentParser(description="VAE for single-cell RNA-seq imputation")
parser.add_argument("--input_path", type=str, required=True, help="Path to input CSV file")
parser.add_argument("--output_path", type=str, required=True, help="Path to save imputed output CSV file")
parser.add_argument("--loss_plot_path", type=str, required=True, help="Path to save loss plot PDF file")
parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
parser.add_argument("--lr", type=float, default=0.00005, help="Learning rate")
parser.add_argument("--mask_rate", type=float, default=0.1,
                    help="Fraction of entries to randomly mask each step (denoising objective)")
parser.add_argument("--seed", type=int, default=42, help="Random seed (controls split, init, masking)")
args = parser.parse_args()

# Seed all RNGs for reproducibility across seeds.
torch.manual_seed(args.seed)
np.random.seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

# ---- Load and Transform Data ----
# NOTE: input CSV is expected to be cells x genes (rows = cells, columns = genes).
print(f"Loading data from {args.input_path}...")
original_data = pd.read_csv(args.input_path, index_col=0)
if original_data.shape[0] < original_data.shape[1] // 10:
    print(f"WARNING: input has {original_data.shape[0]} rows and {original_data.shape[1]} columns; "
          "this script assumes cells x genes. Transpose if your file is genes x cells.")
transformed_data = np.log1p(original_data)
input_data = torch.tensor(transformed_data.values, dtype=torch.float32)

# ---- Train/Dev/Test Split (80/10/10 by row) ----
n_total = input_data.shape[0]
perm = torch.randperm(n_total, generator=torch.Generator().manual_seed(args.seed))
n_test = max(1, int(round(n_total * 0.10)))
n_dev = max(1, int(round(n_total * 0.10)))
test_idx = perm[:n_test]
dev_idx = perm[n_test:n_test + n_dev]
train_idx = perm[n_test + n_dev:]
train_data = input_data[train_idx]
dev_data = input_data[dev_idx]
test_data = input_data[test_idx]

train_loader = data.DataLoader(data.TensorDataset(train_data), batch_size=args.batch_size, shuffle=True)
dev_loader = data.DataLoader(data.TensorDataset(dev_data), batch_size=args.batch_size)
test_loader = data.DataLoader(data.TensorDataset(test_data), batch_size=args.batch_size)
infer_loader = data.DataLoader(data.TensorDataset(input_data), batch_size=args.batch_size)

# ---- VAE Model ----
class scVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=400, latent_dim=200):
        super(scVAE, self).__init__()

        # Two-layer encoder, symmetric to the decoder.
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
        )
        self.mean_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)

        # ReLU on the final layer: outputs are non-negative AND can be exactly 0,
        # which matters for sparse single-cell data (most entries are true zeros).
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim),
            nn.ReLU()
        )

    def encode(self, x):
        h = self.encoder(x)
        mean = self.mean_layer(h)
        # Clamp logvar to keep exp() and KL numerically stable.
        logvar = torch.clamp(self.logvar_layer(h), min=-10.0, max=10.0)
        return mean, logvar

    def reparameterization(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        return mean + std * epsilon

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterization(mean, logvar)
        x_hat = self.decode(z)
        return x_hat, mean, logvar

# ---- Loss Function ----
def loss_function(x, x_hat, mean, log_var, kl_weight=1.0):
    recon_loss = nn.functional.mse_loss(x_hat, x, reduction='none').sum(dim=-1).mean()
    kl_div = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), dim=-1).mean()
    return recon_loss + kl_weight * kl_div

# ---- Training Function ----
def train(model, optimizer, train_loader, dev_loader, epochs, device, mask_rate):
    train_losses = []
    dev_losses = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_train_batches = 0
        kl_weight = min(1.0, epoch / 10.0)

        for batch in train_loader:
            x = batch[0].to(device)
            mask = torch.bernoulli(torch.full_like(x, 1.0 - mask_rate))
            x_corrupt = x * mask
            optimizer.zero_grad()
            x_hat, mean, log_var = model(x_corrupt)
            loss = loss_function(x, x_hat, mean, log_var, kl_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_train_batches += 1

        avg_train_loss = total_loss / max(1, n_train_batches)
        train_losses.append(avg_train_loss)

        # Dev loss (use the same kl_weight as training so the curves are comparable)
        model.eval()
        with torch.no_grad():
            dev_loss = 0.0
            n_dev_batches = 0
            for batch in dev_loader:
                x = batch[0].to(device)
                x_hat, mean, log_var = model(x)
                dev_loss += loss_function(x, x_hat, mean, log_var, kl_weight).item()
                n_dev_batches += 1
            avg_dev_loss = dev_loss / max(1, n_dev_batches)
            dev_losses.append(avg_dev_loss)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Dev Loss: {avg_dev_loss:.4f}")

    return train_losses, dev_losses

# ---- Setup ----
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_dim = input_data.shape[1]
model = scVAE(input_dim=input_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)

# ---- Run Training ----
print("Training VAE...")
train_losses, dev_losses = train(model, optimizer, train_loader, dev_loader,
                                 args.epochs, device, args.mask_rate)

# ---- Evaluate on Test Set ----
# Use kl_weight=1.0 (the fully-annealed value); only meaningful if epochs >= 10.
print("Evaluating on test set...")
model.eval()
with torch.no_grad():
    test_loss = 0.0
    n_test_batches = 0
    for batch in test_loader:
        x = batch[0].to(device)
        x_hat, mean, log_var = model(x)
        test_loss += loss_function(x, x_hat, mean, log_var, kl_weight=1.0).item()
        n_test_batches += 1
    test_loss /= max(1, n_test_batches)
print(f"Test Loss: {test_loss:.4f}")

# ---- Plot Training and Dev Loss ----
plt.figure(figsize=(8, 5))
plt.plot(range(1, args.epochs + 1), train_losses, marker='o', label='Train Loss')
plt.plot(range(1, args.epochs + 1), dev_losses, marker='s', label='Dev Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Dev Loss over Epochs")
plt.legend()
plt.grid(True)
plt.savefig(args.loss_plot_path)
print(f"Loss plot saved to {args.loss_plot_path}")

# ---- Impute Full Data (batched to avoid OOM) ----
# NOTE: this reconstructs every cell, including dev/test cells that were never trained on.
# Decode from the posterior MEAN (not a sample) so imputation is deterministic.
print("Reconstructing full input data...")
model.eval()
recon_chunks = []
with torch.no_grad():
    for batch in infer_loader:
        x = batch[0].to(device)
        mean, _ = model.encode(x)
        x_hat = model.decode(mean)
        recon_chunks.append(x_hat.cpu().numpy())
reconstructed_data = np.concatenate(recon_chunks, axis=0)

# ---- Inverse Transform (kept as float; downstream code can round if it needs counts) ----
reconstructed_data = np.expm1(reconstructed_data)
reconstructed_df = pd.DataFrame(reconstructed_data, columns=original_data.columns, index=original_data.index)

# ---- Save Output ----
print(f"Saving imputed data to {args.output_path}...")
reconstructed_df.to_csv(args.output_path)
print("Done!")