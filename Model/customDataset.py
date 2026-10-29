import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# Pytorch does not like string labels. real = 0 fake = 1
class ImageDataset(Dataset):
    def __init__(self, csv_file, img_dir):

        self.annotations = pd.read_csv(csv_file)
        self.img_dir = Path(img_dir)

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):

        img_path = self.img_dir / str(
            self.annotations.iloc[index, 0]
        )

        label = torch.tensor(
            int(self.annotations.iloc[index, 1]),
            dtype=torch.long
        )

        with Image.open(img_path) as image:
            image = image.convert("RGB")
            image = self.transform(image)

        return image, label

