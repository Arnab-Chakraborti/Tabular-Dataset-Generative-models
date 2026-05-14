import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import fetch_california_housing
from src.model import Generator 
from src.utils import fast_mmd, boundary_loss
from src.eval_metrics import plot_kde,lat_long_spread, plot_tsne_comparison, plot_histograms

DATA_PATH = os.path.join(os.getcwd(), "data", "california_housing.csv")
if os.path.exists(DATA_PATH):
    print("Loading data from local /data folder...")
    df = pd.read_csv(DATA_PATH)
else:
    print("Local data not found. Fetching from Scikit-Learn")
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame
    df.to_csv(DATA_PATH, index=False)
    print(f"Data saved to {DATA_PATH} for future use.")


print(df.head())


X_train, X_test = train_test_split(
    df,
    test_size=0.3,
    random_state=42,
    shuffle=True,
)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)

real_min = X_train_scaled.min(axis=0).values
real_max = X_train_scaled.max(axis=0).values

X_train_scaled = torch.tensor(X_train_scaled.values, dtype=torch.float32)
real_min = torch.tensor(real_min, dtype=torch.float32)
real_max = torch.tensor(real_max, dtype=torch.float32)

train_dataset = TensorDataset(X_train_scaled) 
train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True, drop_last=True)

noise_dim = 14
output_dim = X_train.shape[1]

generator_model = Generator(noise_dim=noise_dim, output_dim=output_dim)
z_test = torch.randn(
    X_test.shape[0],
    noise_dim
)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    generator_model.parameters(),
    lr=0.001,
    weight_decay=1e-4
)


epochs = 200
lambda_gen = 50 
lambda_boundary = 1000 
patience = 25
best_loss = float('inf')
counter = 0
checkpoint_path = 'best_generator.pth'

generator_model.train()

for epoch in range(epochs):
    total_mse = 0
    total_mmd = 0
    total_b_loss = 0
    total_loss_val = 0
    for batch_X_real_list in train_loader:
        batch_X_real = batch_X_real_list[0]
        current_batch_size = batch_X_real.size(0)
        
        z_fresh = torch.randn(current_batch_size, noise_dim)
        z_fresh = z_fresh + (torch.randn_like(z_fresh) * 0.05)
        
        batch_X_gen = generator_model(z_fresh)
        
        mse_loss = criterion(batch_X_gen, batch_X_real)
        mmd_loss_val = fast_mmd(batch_X_real, batch_X_gen)
        b_loss = boundary_loss(batch_X_gen, real_min, real_max)

        loss = mse_loss + (lambda_gen * mmd_loss_val) + (lambda_boundary * b_loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_mse += mse_loss.item()
        total_mmd += mmd_loss_val.item()
        total_b_loss += b_loss.item()
        total_loss_val += loss.item()
        

    avg_loss = total_loss_val / len(train_loader)

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(generator_model.state_dict(), checkpoint_path)
        counter = 0
    else:
        counter += 1

    if epoch % 10 == 0:
        avg_mse = total_mse / len(train_loader)
        avg_mmd = total_mmd / len(train_loader)
        avg_b_loss = total_b_loss / len(train_loader)
        print(f"Epoch [{epoch}/{epochs}] | MSE: {avg_mse:.4f} | MMD: {avg_mmd:.4f} | Boundary: {avg_b_loss:.4f} | Total: {avg_loss:.4f}")

    if counter >= patience:
        print(f"Early stopping at epoch {epoch}")
        break
generator_model.load_state_dict(torch.load(checkpoint_path))

generator_model.eval()

with torch.no_grad():
    c = generator_model(z_test)
    c = c.cpu().numpy()

# Transform back to original scale
reconstr_data = scaler.inverse_transform(c)
reconstr_df = pd.DataFrame(reconstr_data, columns=X_test.columns)

# Statistics
print("Generated Data vs Real Data Description:")
print(reconstr_df.describe(), X_test.describe())

# Run the plots
print("Generating Latitude-Longitude Spread of Real Data vs Fake Data...")
lat_long_spread(reconstr_df,X_test)

print("Generating Distribution comparisons:")
plot_kde(reconstr_df,X_test)

#2D visualisation
plot_tsne_comparison(X_test, reconstr_df)
