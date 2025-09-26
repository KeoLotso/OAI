import os
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.utils as vutils
import shutil

class ImageDataset(Dataset):
    def __init__(self, folder, img_size=64):
        self.files = [os.path.join(folder, f) for f in os.listdir(folder)
                      if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        return self.transform(img)

class Autoencoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128*8*8, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128*8*8),
            nn.Unflatten(1, (128, 8, 8)),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid()
        )
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

def train():
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, "data")
    if not os.path.exists(data_dir):
        print("No 'data' folder found. Add images first.")
        return

    datasets = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
    if not datasets:
        print("No subfolders found in 'data'. Add a folder with images.")
        return

    print("Available datasets:")
    for i, d in enumerate(datasets, 1):
        print(f"{i}. {d}")
    choice = int(input("Select dataset number to train on: ")) - 1
    data_folder = os.path.join(data_dir, datasets[choice])
    dataset = ImageDataset(data_folder)
    if len(dataset) == 0:
        print(f"No images found in {data_folder}.")
        return

    model_name = input("Enter model name: ").strip() or datasets[choice]
    latent_dim = int(input("Latent dimension (default 128): ") or 128)
    batch_size = int(input("Batch size (default 16): ") or 16)
    epochs = int(input("Epochs (default 20): ") or 20)
    lr = float(input("Learning rate (default 1e-3): ") or 1e-3)

    models_dir = os.path.join(base_dir, "Models")
    os.makedirs(models_dir, exist_ok=True)
    model_dir = os.path.join(models_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    generated_dir = os.path.join(model_dir, "generated")
    os.makedirs(generated_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = Autoencoder(latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(1, epochs+1):
        total_loss = 0
        for imgs in loader:
            imgs = imgs.to(device)
            optimizer.zero_grad()
            recon = model(imgs)
            loss = criterion(recon, imgs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch}/{epochs}, Loss: {total_loss/len(loader):.4f}")

        with torch.no_grad():
            sample = model.decoder(torch.randn(batch_size, latent_dim, device=device))
            vutils.save_image(sample, os.path.join(generated_dir, f"sample_epoch{epoch}.png"), normalize=True, nrow=4)

    torch.save(model.state_dict(), os.path.join(model_dir, f"{model_name}.pth"))
    print(f"Training finished. Model saved in '{model_dir}'")

def generate():
    base_dir = os.path.dirname(__file__)
    models_dir = os.path.join(base_dir, "Models")
    if not os.path.exists(models_dir):
        print("No Models folder found. Train a model first.")
        return

    models = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
    if not models:
        print("No models found. Train a model first.")
        return

    print("Available models:")
    for i, m in enumerate(models, 1):
        print(f"{i}. {m}")
    choice = int(input("Select model number: ")) - 1
    model_name = models[choice]

    latent_dim = int(input("Latent dimension (default 128): ") or 128)
    num_images = int(input("Number of images to generate (default 4): ") or 4)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_dir = os.path.join(models_dir, model_name)
    model_path = os.path.join(model_dir, f"{model_name}.pth")
    if not os.path.exists(model_path):
        print("Model file not found!")
        return

    model = Autoencoder(latent_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    generated_dir = os.path.join(model_dir, "generated_gen")
    os.makedirs(generated_dir, exist_ok=True)

    with torch.no_grad():
        z = torch.randn(num_images, latent_dim, device=device)
        imgs = model.decoder(z)
        for i, img in enumerate(imgs):
            vutils.save_image(img, os.path.join(generated_dir, f"gen_{i}.png"), normalize=True)

    print(f"Generated {num_images} images in '{generated_dir}'")

def remove_model():
    base_dir = os.path.dirname(__file__)
    models_dir = os.path.join(base_dir, "Models")
    if not os.path.exists(models_dir):
        print("No Models folder found.")
        return

    models = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
    if not models:
        print("No models to remove.")
        return

    print("Available models:")
    for i, m in enumerate(models, 1):
        print(f"{i}. {m}")
    choice = int(input("Select model number to remove: ")) - 1
    model_name = models[choice]
    shutil.rmtree(os.path.join(models_dir, model_name))
    print(f"Model '{model_name}' removed successfully.")

if __name__ == "__main__":
    choice = input("[t]rain, [g]enerate or [r]emove model? ").strip().lower()
    if choice.startswith("t"):
        train()
    elif choice.startswith("g"):
        generate()
    elif choice.startswith("r"):
        remove_model()
