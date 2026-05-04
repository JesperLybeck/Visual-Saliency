import kagglehub
from pathlib import Path
import torch
from torch.utils.data import Dataset, random_split, DataLoader
import numpy as np
from PIL import Image

def downloadDataset():
    project_dir = Path(__file__).resolve().parent
    dataset_dir = project_dir / "dataset"
    download_dir = dataset_dir / "salicon"

    dataset_dir.mkdir(parents=True, exist_ok=True)
    download_dir.mkdir(parents=True, exist_ok=True)

    gitignore = dataset_dir / ".gitignore"
    if not gitignore.exists():
        gitignore.write_text("*\n!.gitignore\n")

    path = kagglehub.dataset_download(
        "roshan401/salicon",
        output_dir=str(download_dir),
        force_download=True
    )

    print("Path to dataset files:", path)


class SaliconDataset(Dataset):
    def __init__(self, split='train', transform=None):
       
        self.transform = transform
        project_dir = Path(__file__).resolve().parent
        dataset_dir = project_dir / "dataset" / "salicon"
        image_dir = dataset_dir / "images" /"images"/ split
        heatmap_dir = dataset_dir / "maps" / split

        print(f"Image dir: {image_dir}")
        print(f"Image dir exists: {image_dir.exists()}")
        print(f"Heatmap dir: {heatmap_dir}")
        print(f"Heatmap dir exists: {heatmap_dir.exists()}")
        
        # Build list of (image_path, heatmap_path) tuples
        self.data = []
        image_files = sorted(image_dir.glob("*.jpg"))
        for img_path in image_files:
            # Use stem (filename without extension) directly
            heatmap_path = heatmap_dir / (img_path.stem + ".png")
            if heatmap_path.exists():
                print(f"Matched: {img_path.name} <-> {heatmap_path.name}")
                self.data.append((str(img_path), str(heatmap_path)))
        print(f"Matched {len(self.data)} pairs\n")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, heatmap_path = self.data[idx]
        
        # Load image
        image = np.array(Image.open(img_path).convert('RGB')).astype(np.float32) / 255.0
        image = torch.tensor(image).permute(2, 0, 1).float()
        
        # Load heatmap (label)
        heatmap = np.array(Image.open(heatmap_path).convert('L')).astype(np.float32) / 255.0
        heatmap = torch.tensor(heatmap).unsqueeze(0).float()
        
        sample = (image, heatmap)
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample


def get_dataloaders(batch_size=32, num_workers=0):
    
    # Load full validation dataset
    val_dataset = SaliconDataset(split='val')
    
    # Split validation into val and test (50/50)
    val_size = len(val_dataset) // 2
    test_size = len(val_dataset) - val_size
    val_dataset, test_dataset = random_split(
        val_dataset,
        [val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Load training dataset
    train_dataset = SaliconDataset(split='train')
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    #downloadDataset()
    
    # Test dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=4)
    
    image, heatmap = next(iter(train_loader))
    print(f"Image batch shape: {image.shape}")      # (4, 3, H, W)
    print(f"Heatmap batch shape: {heatmap.shape}")  # (4, 1, H, W)
    print(f"Train size: {len(train_loader.dataset)}")
    print(f"Val size: {len(val_loader.dataset)}")
    print(f"Test size: {len(test_loader.dataset)}")


