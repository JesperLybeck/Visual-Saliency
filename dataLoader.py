import kagglehub
from pathlib import Path
import torch
import torchvision
from torch.utils.data import Dataset, random_split, DataLoader
import numpy as np
import random
import torchvision.transforms.functional as TF
from scipy.io import loadmat
import requests
import zipfile

from PIL import Image

def download_and_extract(url, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    zip_path = output_dir / "dataset.zip"

    print(f"Downloading {url}")

    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(zip_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    print("Extracting...")

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(output_dir)

    zip_path.unlink()

    print(f"Dataset extracted to: {output_dir}")


def buildFixationMap(fixation_data, image_size=(480, 640)):
    gaze = fixation_data["gaze"]

    fixation_map = np.zeros(
            (image_size[0], image_size[1]),
            dtype=np.float32
        )
    for observer in range(gaze.shape[0]):

        fix_points = gaze[observer, 0][2]

        for x, y in fix_points:

            x = int(x)
            y = int(y)

            if (
                0 <= x < image_size[1] and
                0 <= y < image_size[0]
            ):
                fixation_map[y, x] = 1.0
    fixation_map = torch.tensor(
    fixation_map
    ).unsqueeze(0).float()
    return fixation_map
        
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


class SaliencyAugmentation:

    def __init__(
        self,
        hflip_prob=0.5,
        brightness=0.08,
        contrast=0.08,
        saturation=0.08,
        hue=0.01,
           
    ):
        self.blur_prob = 0.15
        self.blur = torchvision.transforms.GaussianBlur(
            kernel_size=3,
            sigma=(0.1, 1.0)
)

        self.hflip_prob = hflip_prob
        
        self.color_jitter = torchvision.transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue
        )
        self.crop_scale = (0.95, 1.0)
        self.crop_ratio = (0.95, 1.05)

    def __call__(self, sample):

        image, heatmap, fixation_map = sample
        if random.random() < self.hflip_prob:

            image = TF.hflip(image)
            heatmap = TF.hflip(heatmap)
            fixation_map = TF.hflip(fixation_map)

        i, j, h, w = torchvision.transforms.RandomResizedCrop.get_params(
            image,
            scale=self.crop_scale,
            ratio=self.crop_ratio
        )

        image = TF.resized_crop(
            image,
            i,
            j,
            h,
            w,
            size=image.shape[-2:],
            interpolation=TF.InterpolationMode.BILINEAR
        )

        heatmap = TF.resized_crop(
            heatmap,
            i,
            j,
            h,
            w,
            size=heatmap.shape[-2:],
            interpolation=TF.InterpolationMode.BILINEAR
        )
        fixation_map = TF.resized_crop(
            fixation_map,
            i,
            j,
            h,
            w,
            size=fixation_map.shape[-2:],
            interpolation=TF.InterpolationMode.NEAREST
        )
        if random.random() < self.blur_prob:
            image = self.blur(image)

        image = self.color_jitter(image)

        return image, heatmap, fixation_map

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
        fixation_dir = dataset_root / "fixations" / split

        print(f"Image dir: {image_dir}")
        print(f"Image dir exists: {image_dir.exists()}")
        print(f"Heatmap dir: {heatmap_dir}")
        print(f"Heatmap dir exists: {heatmap_dir.exists()}")

        self.data = []
        image_files = sorted(image_dir.glob("*.jpg"))
        for img_path in image_files:
            heatmap_path = heatmap_dir / f"{img_path.stem}.png"
            fixation_path = fixation_dir / f"{img_path.stem}.mat"
            if heatmap_path.exists() and fixation_path.exists():
                self.data.append((str(img_path), str(heatmap_path), str(fixation_path)))

        print(f"Matched {len(self.data)} pairs\n")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, heatmap_path, fixation_path = self.data[idx]

        image = np.array(Image.open(img_path).convert('RGB')).astype(np.float32) / 255.0
        image = torch.tensor(image).permute(2, 0, 1).float()
        
        heatmap = np.array(Image.open(heatmap_path).convert('L')).astype(np.float32) / 255.0
        heatmap = torch.tensor(heatmap).unsqueeze(0).float()

        fixation_data = loadmat(fixation_path)
        fixation_map = buildFixationMap(fixation_data)

        image = TF.resize(
        image,
        [480, 640],
        interpolation=TF.InterpolationMode.BILINEAR
        )

        heatmap = TF.resize(
            heatmap,
            [480, 640],
            interpolation=TF.InterpolationMode.BILINEAR
        )
        fixation_map = TF.resize(
            fixation_map,
            [480,640],
            interpolation=TF.InterpolationMode.NEAREST
        )

        sample = (image, heatmap, fixation_map)

        if self.transform:
            sample = self.transform(sample)

        return sample


def get_dataloaders(batch_size=32, num_workers=0, dataset_root=None):
    print("Getting dataloaders...")
    
    train_transform = SaliencyAugmentation()

    train_dataset = SaliconDataset(
        split='train',
        transform=train_transform,
        dataset_root=dataset_root
    )

    val_dataset = SaliconDataset(
        split='val',
        dataset_root=dataset_root
    )

    val_size = len(val_dataset) // 2
    test_size = len(val_dataset) - val_size
    val_dataset, test_dataset = random_split(
        val_dataset,
        [val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
   
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


