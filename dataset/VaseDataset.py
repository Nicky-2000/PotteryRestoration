import pandas as pd
import os
from PIL import Image
from torch.utils.data import Dataset
import torch

class VaseDataset(Dataset):
    def __init__(self, root_dir: str, captions_file: str = "captions.csv", transform=None):
        self.root_dir = root_dir # This should either be /dataset/train or /dataset/val
        self.captions = pd.read_csv(captions_file)
        self.transform = transform

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Load masked image
        masked_img_name = os.path.join(
            self.root_dir, "masked", self.captions.iloc[idx, 0]
        )
        masked_image = Image.open(masked_img_name).convert("RGB")

        # Load full image
        full_img_name = os.path.join(
            self.root_dir,
            "full",
            self.captions.iloc[idx, 0].replace("_masked", "_full"),
        )
        full_image = Image.open(full_img_name).convert("RGB")

        # Get caption
        caption = self.captions.iloc[idx, 1]

        # Apply transformations if specified
        if self.transform:
            masked_image = self.transform(masked_image)
            full_image = self.transform(full_image)

        return {
            "masked_images": masked_image,
            "full_images": full_image,
            "text": caption,
        }
