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
    def __init__(self, split='train', transform=None, dataset_root=None):
        self.transform = transform

        if dataset_root is None:
            project_dir = Path(__file__).resolve().parent
            dataset_root = project_dir / "dataset" / "salicon"
        else:
            dataset_root = Path(dataset_root)

        image_dir = dataset_root / "images" / "images" / split
        heatmap_dir = dataset_root / "maps" / split

        print(f"Image dir: {image_dir}")
        print(f"Image dir exists: {image_dir.exists()}")
        print(f"Heatmap dir: {heatmap_dir}")
        print(f"Heatmap dir exists: {heatmap_dir.exists()}")

        self.data = []
        image_files = sorted(image_dir.glob("*.jpg"))
        for img_path in image_files:
            heatmap_path = heatmap_dir / f"{img_path.stem}.png"
            if heatmap_path.exists():
                self.data.append((str(img_path), str(heatmap_path)))

        print(f"Matched {len(self.data)} pairs\n")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, heatmap_path = self.data[idx]

        image = np.array(Image.open(img_path).convert('RGB')).astype(np.float32) / 255.0
        image = torch.tensor(image).permute(2, 0, 1).float()

        heatmap = np.array(Image.open(heatmap_path).convert('L')).astype(np.float32) / 255.0
        heatmap = torch.tensor(heatmap).unsqueeze(0).float()

        sample = (image, heatmap)

        if self.transform:
            sample = self.transform(sample)

        return sample


def get_dataloaders(batch_size=32, num_workers=0, dataset_root=None):
    print("Getting dataloaders...")
    train_dataset = SaliconDataset(split='train', dataset_root=dataset_root)
    val_dataset = SaliconDataset(split='val', dataset_root=dataset_root)

    val_size = len(val_dataset) // 2
    test_size = len(val_dataset) - val_size
    val_dataset, test_dataset = random_split(
        val_dataset,
        [val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_dataset = SaliconDataset(split='train', dataset_root=dataset_root)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=4)
    image, heatmap = next(iter(train_loader))
    print(f"Image batch shape: {image.shape}")
    print(f"Heatmap batch shape: {heatmap.shape}")