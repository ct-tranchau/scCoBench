import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Argument parser for command-line execution
parser = argparse.ArgumentParser(description="GAN for single-cell RNA-seq imputation")
parser.add_argument("--input_path", type=str, required=True, help="Path to input CSV file")
parser.add_argument("--output_path", type=str, required=True, help="Path to save imputed output CSV file")
parser.add_argument("--loss_plot_path", type=str, required=True, help="Path to save loss plot PDF file")
parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
parser.add_argument("--lr_g", type=float, default=0.0001, help="Learning rate for Generator")
parser.add_argument("--lr_d", type=float, default=0.00001, help="Learning rate for Discriminator")
parser.add_argument("--lambda_adv", type=float, default=0.1,
                    help="Weight on the adversarial loss for the generator")
parser.add_argument("--mask_rate", type=float, default=0.1,
                    help="Fraction of entries to randomly mask each step (denoising objective)")
parser.add_argument("--seed", type=int, default=42, help="Random seed (controls init, masking)")
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

# Apply log1p transformation
transformed_data = np.log1p(original_data)

# Convert the data to a PyTorch tensor
input_data = torch.tensor(transformed_data.values, dtype=torch.float32)

# Step 2: Create PyTorch DataLoaders (training + non-shuffled inference loader)
train_dataset = data.TensorDataset(input_data)
train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
infer_loader = data.DataLoader(train_dataset, batch_size=args.batch_size)

# Step 3: Define the GAN architecture
class Generator(nn.Module):
    def __init__(self, input_dim, hidden_dim=400, latent_dim=200):
        super(Generator, self).__init__()
        # ReLU on the final layer: outputs are non-negative AND can be exactly 0,
        # which matters for sparse single-cell data (most entries are true zeros).
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim=400):
        super(Discriminator, self).__init__()
        # Outputs raw logits; pair with BCEWithLogitsLoss for numerical stability.
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        return self.model(x)

# Step 4: Initialize the models, optimizer, and device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_dim = input_data.shape[1]

generator = Generator(input_dim).to(device)
discriminator = Discriminator(input_dim).to(device)

optimizer_G = optim.Adam(generator.parameters(), lr=args.lr_g)
optimizer_D = optim.Adam(discriminator.parameters(), lr=args.lr_d)

criterion_G = nn.MSELoss()                # reconstruction term
criterion_D = nn.BCEWithLogitsLoss()      # numerically stable BCE on raw logits

# Step 5: Train the GAN with loss tracking
def train_gan(generator, discriminator, train_loader, epochs, device, lambda_adv, mask_rate):
    g_losses = []
    d_losses = []

    for epoch in range(epochs):
        generator.train()
        discriminator.train()
        g_loss_total = 0.0
        d_loss_total = 0.0
        n_batches = 0

        for (x,) in train_loader:
            x = x.to(device)

            real_labels = torch.ones((x.size(0), 1), device=device)
            fake_labels = torch.zeros((x.size(0), 1), device=device)

            # Corrupt the input so the generator learns to recover masked entries
            mask = torch.bernoulli(torch.full_like(x, 1.0 - mask_rate))
            x_corrupt = x * mask

            # ---- Train the Discriminator on real vs. detached fakes ----
            optimizer_D.zero_grad()
            generated_data = generator(x_corrupt)
            real_pred = discriminator(x)
            fake_pred = discriminator(generated_data.detach())
            d_loss_real = criterion_D(real_pred, real_labels)
            d_loss_fake = criterion_D(fake_pred, fake_labels)
            d_loss = (d_loss_real + d_loss_fake) / 2
            d_loss.backward()
            optimizer_D.step()
            d_loss_total += d_loss.item()

            # ---- Train the Generator: reconstruction + adversarial ----
            optimizer_G.zero_grad()
            generated_data = generator(x_corrupt)
            g_recon = criterion_G(generated_data, x)
            g_adv = criterion_D(discriminator(generated_data), real_labels)
            g_loss = g_recon + lambda_adv * g_adv
            g_loss.backward()
            optimizer_G.step()
            g_loss_total += g_loss.item()

            n_batches += 1

        avg_g_loss = g_loss_total / max(1, n_batches)
        avg_d_loss = d_loss_total / max(1, n_batches)
        g_losses.append(avg_g_loss)
        d_losses.append(avg_d_loss)

        print(f"Epoch {epoch+1}/{epochs}, Generator Loss: {avg_g_loss:.4f}, "
              f"Discriminator Loss: {avg_d_loss:.4f}")

    return generator, g_losses, d_losses

# Train the GAN
print("Training GAN...")
generator, g_losses, d_losses = train_gan(generator, discriminator, train_loader,
                                          args.epochs, device, args.lambda_adv, args.mask_rate)

# Step 6: Plot and save Generator and Discriminator Loss
plt.figure(figsize=(8, 5))
plt.plot(range(1, args.epochs + 1), g_losses, label="Generator Loss", marker='o')
plt.plot(range(1, args.epochs + 1), d_losses, label="Discriminator Loss", marker='s')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Generator and Discriminator Loss During Training")
plt.legend()
plt.grid()
plt.savefig(args.loss_plot_path, format="pdf")
plt.close()
print(f"Loss plot saved to {args.loss_plot_path}")

# Step 7: Generate imputed data (batched to avoid OOM)
print("Generating imputed data...")
generator.eval()
imputed_chunks = []
with torch.no_grad():
    for (x,) in infer_loader:
        imputed_chunks.append(generator(x.to(device)).cpu().numpy())
imputed_data = np.concatenate(imputed_chunks, axis=0)

# Convert back to original scale (kept as float; downstream code can round if it needs counts)
imputed_data = np.expm1(imputed_data)

imputed_df = pd.DataFrame(imputed_data, columns=original_data.columns, index=original_data.index)

# Step 8: Save the imputed DataFrame
print(f"Saving imputed data to {args.output_path}...")
imputed_df.to_csv(args.output_path)
print("Done!")
